# %%
import gymnasium as gym
import tmrl
from collections import deque, namedtuple
from jax import random, grad, jit, tree_util, lax, nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
import pickle


# %% Setup replay buffer and environment
entry = namedtuple("Memory", ["obs_data", "obs_images", "action", "reward", "next_obs_data", "next_obs_images", "done"])
memory = deque(maxlen=1000)  # Replay buffer
gamma = 0.99  # Discount factor
learning_rate = 0.01
env = tmrl.get_environment()
filename = "params.pkl"
filenameTarget = "targetParams.pkl"

def conv(input, kernel, stride = 1, padding="SAME"):
    return lax.conv(input, kernel, (stride, stride), padding=padding)

# %% Initialize network parameters
def init_weights(shape, rng_key):
    fan_in = jnp.prod(jnp.asarray(shape[:-1]))
    stddev = jnp.sqrt(2.0 / fan_in)
    return jax.random.normal(rng_key, shape) * stddev

#Initialize params
def initialize_params(rng):
    rngs = jax.random.split(rng, 5)
    conv1 = init_weights((16, 4, 8, 8), rngs[0])
    conv2 = init_weights((32, 16, 4, 4), rngs[1])
    fc_layer_one = init_weights((2051, 256), rngs[2])
    fc_layer_two = init_weights((256, 3), rngs[3])
    params = conv1, conv2, fc_layer_one, fc_layer_two
    return params

# Forward pass
def model(params, obs_data, obs_images):
    if obs_images.ndim == 3:
        obs_images = obs_images[None,...]
    conv1, conv2, fc_layer_one, fc_layer_two = params
    x = nn.leaky_relu(conv(obs_images, conv1, 4))
    x = nn.leaky_relu(conv(x, conv2, 2))
    x = x.reshape(x.shape[0], -1)
    x = jnp.concatenate([x, obs_data.reshape(x.shape[0], -1)], axis=1)
    x = jnp.tanh(jnp.dot(x, fc_layer_one))
    x = jnp.tanh(jnp.dot(x, fc_layer_two))
    return x

# Loss function using Bellman equation
def bellman_loss(params, target_params, batch):
    obs_data, obs_images, actions, rewards, next_obs_data, next_obs_images, dones = batch

    # Forward pass to get q-values for current observations and next observations
    q_values = model(params, obs_data, obs_images)           # Shape: (batch_size, num_actions)
    next_q_values = model(target_params, next_obs_data, next_obs_images)  # Shape: (batch_size, num_actions)

    # Index into q_values for each action dimension: assume actions[:, 0] = accelerate, actions[:, 1] = brake, etc.
    q_values_accelerate = q_values[:, 0] * actions[:, 0]
    q_values_brake = q_values[:, 1] * actions[:, 1]
    q_values_steer = q_values[:, 2] * actions[:, 2]
    
    # Calculate the combined q_value for the selected actions
    q_values_selected = q_values_accelerate + q_values_brake + q_values_steer

    # For target Q-values, get the max over next Q-values for each dimension
    max_next_q_values = jnp.max(next_q_values, axis=1)
    target = rewards + gamma * max_next_q_values * (1.0 - dones)
    
    # Calculate the Bellman loss
    loss = jnp.mean((q_values_selected - target) ** 2)
    return loss


# Policy with epsilon-greedy for continuous actions
def policy_fn(params, rng, obs_data, obs_images, epsilon=0.1):
    rng, key = random.split(rng)
    if random.uniform(key) < epsilon:
        # Random action
        return jnp.array([np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(-1, 1)])
    else:
        action = model(params, obs_data, obs_images)
        return action[0]

# Training step with gradient descent
@jit
def update(params, target_params, opt_state, batch):
    loss, gradients = jax.value_and_grad(bellman_loss)(params, target_params, batch)
    updates, opt_state = optimizer.update(gradients, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Sampling from memory
def sample_batch(memory):
    batch = memory
    obs_data_batch = jnp.array([b.obs_data for b in batch])
    obs_images_batch = jnp.array([b.obs_images for b in batch])
    action_batch = jnp.array([b.action for b in batch])
    reward_batch = jnp.array([b.reward for b in batch])
    next_obs_data_batch = jnp.array([b.next_obs_data for b in batch])
    next_obs_images_batch = jnp.array([b.next_obs_images for b in batch])
    done_batch = jnp.array([b.done for b in batch])
    batch = obs_data_batch, obs_images_batch, action_batch, reward_batch, next_obs_data_batch, next_obs_images_batch, done_batch
    return batch

def preprocess_obs(obs):
    """
    Preprocessor for TM2020 with images, converting images back to uint8
    """
    grayscale_images = obs[3]
    grayscale_images = grayscale_images.astype(np.float32) / 256.0
    obs_data = jnp.concatenate([obs[0], obs[1], obs[2]], axis=0)
    return obs_data, grayscale_images

# Soft update of target network
def update_target_network(params, target_params, tau=0.95):
    return tree_util.tree_map(lambda p, tp: tau * p + (1 - tau) * tp, params, target_params)

# Main training loop
rng = random.PRNGKey(0)
output_size = 3
params = initialize_params(rng)
target_params = params

# Set up optimizer
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)
rewards = []
highestReward = 0


# # %%
# obs, info = env.reset()
# total_reward = 0

# # Print the observation structure to understand it better
# print("Observation structure:", type(obs), "Content:", obs)

# # If obs is a dictionary or has nested components, handle accordingly:
# if isinstance(obs, dict):
#     # Assuming the actual observation is in a key like 'observation' (adjust as per env)
#     obs_array = jnp.asarray(obs.get("observation", obs)).flatten()
# elif isinstance(obs, (tuple, list)):
#     # If obs is a tuple or list, access a specific part that’s the main observation
#     obs_array = jnp.asarray(obs[0]).flatten()
# else:
#     # If obs is already an array-like structure, directly flatten it
#     obs_array = jnp.asarray(obs).flatten()

# %%

for episode in range(1000):  # rtgym ensures this runs at 20Hz by default
    obs, info = env.reset()
    total_reward = 0
    obs_data, obs_images = preprocess_obs(obs)
    t = 0
    terminated = False
    truncated = False
    first = True
    while not (terminated | truncated):
        t += 1
        rng, key = random.split(rng)
        epsilon = max(0.01, 1.0 - episode / 1000)
        action = policy_fn(params, key, obs_data, obs_images, epsilon)
        action = action.at[0].set(1)
        action = action.at[1].set(0)

        next_obs, reward, terminated, truncated, info = env.step(action)
        if first:
            reward = 0
        first = False
        done = terminated or truncated
        next_obs_data, next_obs_images = preprocess_obs(next_obs)
        reward = reward + (next_obs_data[0] / 100)
        # Store transition in replay buffer
        memory.append(entry(obs_data, obs_images, action, reward, next_obs_data, next_obs_images, done))
        total_reward += reward
        obs_data, obs_images = next_obs_data, next_obs_images

        if done:
            
            batch = sample_batch(memory)
            params, opt_state, loss = update(params, target_params, opt_state, batch)

            if t % 50 == 0:
                target_params = update_target_network(params, target_params)
                rewards.append(total_reward)
            
            if(highestReward < total_reward):
                    best_params = params
                    best_target_params = target_params
                    highestReward = total_reward
                    print("New record:" , highestReward)
            
            memory.clear()
            break

# %% ------------------- 4. Graph -------------------
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Progress')
plt.show()
# %% -------------------  Params Saver -------------------
# import pickle
# with open(filename, 'wb') as f:
#     pickle.dump(best_params, f)
# with open(filenameTarget, 'wb') as f:
#     pickle.dump(best_target_params, f)

# # %% ------------------- Params Loader -------------------
# with open(filename, 'rb') as f:
#     loaded_params = pickle.load(f)
# with open(filenameTarget, 'rb') as f:
#     loaded_target_params = pickle.load(f)

# # %%
# params = loaded_params
# target_params = loaded_target_params