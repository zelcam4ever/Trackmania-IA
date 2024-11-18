# %%
import gymnasium as gym
from tmrl import get_environment
from collections import deque, namedtuple
from jax import random, grad, jit, tree_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
import pickle


# %% Setup replay buffer and environment
entry = namedtuple("Memory", ["obs", "action", "reward", "next_obs", "done"])
memory = deque(maxlen=1000)  # Replay buffer
gamma = 0.99  # Discount factor
learning_rate = 0.01
env = get_environment()
filename = "params.pkl"
filenameTarget = "targetParams.pkl"

# %% Initialize network parameters
def initialize_mlp_params(rng, input_size, hidden_sizes, output_size):
    def init_layer_params(m, n, rng):
        w_key, b_key = random.split(rng)
        weight = random.normal(w_key, (m, n)) * jnp.sqrt(2.0 / m)
        bias = jnp.zeros(n)
        return weight, bias

    params = []
    keys = random.split(rng, len(hidden_sizes) + 1)

    # Input layer
    params.append(init_layer_params(input_size, hidden_sizes[0], keys[0]))
    # Hidden layers
    for i in range(len(hidden_sizes) - 1):
        params.append(init_layer_params(hidden_sizes[i], hidden_sizes[i + 1], keys[i + 1]))
    # Output layer
    params.append(init_layer_params(hidden_sizes[-1], output_size, keys[-1]))
    return params

# Forward pass
def forward_mlp(params, x):
    activations = x
    for w, b in params[:-1]:
        activations = jnp.tanh(jnp.dot(activations, w) + b)  # For bounded actions
    final_w, final_b = params[-1]
    return jnp.tanh(jnp.dot(activations, final_w) + final_b)

# Loss function using Bellman equation
def bellman_loss(params, target_params, batch):
    obs, actions, rewards, next_obs, dones = batch

    # Forward pass to get q-values for current observations and next observations
    q_values = forward_mlp(params, obs)           # Shape: (batch_size, num_actions)
    next_q_values = forward_mlp(target_params, next_obs)  # Shape: (batch_size, num_actions)

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
def policy_fn(params, rng, obs, epsilon=0.1):
    rng, key = random.split(rng)
    if random.uniform(key) < epsilon:
        # Random action
        return jnp.array([np.random.uniform(-1, 1), np.random.uniform(0, 1), np.random.uniform(-1, 1)])
    else:
        action = forward_mlp(params, obs)
        return jnp.array([1.0, 0.0, action[2]])  # Keeping gas and brake constant here

# Training step with gradient descent
@jit
def update(params, target_params, opt_state, batch):
    loss, gradients = jax.value_and_grad(bellman_loss)(params, target_params, batch)
    updates, opt_state = optimizer.update(gradients, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Sampling from memory
def sample_batch(rng, memory, batch_size):
    indices = random.choice(rng, jnp.arange(len(memory)), shape=(batch_size,), replace=False)
    batch = [memory[i] for i in indices]
    batch = entry(*zip(*batch))
    obs, actions, rewards, next_obs, dones = map(jnp.array, (batch.obs, batch.action, batch.reward, batch.next_obs, batch.done))
    return obs, actions, rewards, next_obs, dones

def preprocess_obs(obs):
    speed = obs[0].flatten()          # Flatten speed array
    lidar = obs[1].flatten()           # Flatten LIDAR array
    prev_actions = obs[2].flatten()    # Flatten previous action arrays
    prev_actions_2 = obs[3].flatten()  # Flatten second previous action array

    # Concatenate all flattened arrays into a single 1D array
    processed_obs = jnp.concatenate([speed, lidar, prev_actions, prev_actions_2])
    return processed_obs

# Soft update of target network
def update_target_network(params, target_params, tau=0.95):
    return tree_util.tree_map(lambda p, tp: tau * p + (1 - tau) * tp, params, target_params)

# Main training loop
rng = random.PRNGKey(0)
input_size = 83  # Based on Trackmania observations (flattened)
hidden_sizes = [128, 64]
output_size = 3
params = initialize_mlp_params(rng, input_size, hidden_sizes, output_size)
target_params = params.copy()

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

for episode in range(200):  # rtgym ensures this runs at 20Hz by default
    obs, info = env.reset()
    total_reward = 0
    obs = preprocess_obs(obs)
    t = 0
    terminated = False
    truncated = False
    first = True
    while not (terminated | truncated):
        t += 1
        rng, key = random.split(rng)
        epsilon = max(0.1, 1.0 - episode / 100)
        action = policy_fn(params, key, obs, epsilon)

        next_obs, reward, terminated, truncated, info = env.step(action)
        if first:
            reward = 0
        first = False
        done = terminated or truncated
        next_obs = preprocess_obs(next_obs)
        # Store transition in replay buffer
        memory.append(entry(obs, action, reward, next_obs, done))
        total_reward += reward
        obs = next_obs

        # Train after enough samples
        if len(memory) >= 64 and t % 4 == 0:
            batch = sample_batch(rng, memory, 64)
            params, opt_state, loss = update(params, target_params, opt_state, batch)

        if t % 50 == 0:
            target_params = update_target_network(params, target_params)

        if done:
            rewards.append(total_reward)
            if(highestReward < total_reward):
                    best_params = params.copy()
                    best_target_params = target_params.copy()
                    highestReward = total_reward
                    print("New record:" , highestReward)
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