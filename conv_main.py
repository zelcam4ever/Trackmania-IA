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
import tqdm


# %% Setup replay buffer and environment
entry = namedtuple("Memory", ["obs_data", "obs_images", "action", "reward", "next_obs_data", "next_obs_images", "done"])
memory = deque(maxlen=1000)  # Replay buffer
gamma = 0.99  # Discount factor
learning_rate = 0.01
env = tmrl.get_environment()
filenameCritic = "criticParams.pkl"
filenameCriticTarget = "criticTargetParams.pkl"
filenameActor = "actorParams.pkl"

def conv(input, kernel, stride = 1, padding="SAME"):
    return lax.conv(input, kernel, (stride, stride), padding=padding)

# %% Initialize network parameters
def init_weights(shape, rng_key):
    fan_in = jnp.prod(jnp.asarray(shape[:-1]))
    stddev = jnp.sqrt(2.0 / fan_in)
    return jax.random.normal(rng_key, shape) * stddev

#Initialize params
def initialize_params_actor(rng):
    rngs = jax.random.split(rng, 5)
    conv1 = init_weights((16, 4, 8, 8), rngs[0])
    conv2 = init_weights((32, 16, 4, 4), rngs[1])
    fc_layer_one = init_weights((2051, 256), rngs[2])
    fc_layer_two = init_weights((256, 3), rngs[3])
    params = conv1, conv2, fc_layer_one, fc_layer_two
    return params

#Initialize params
def initialize_params_critic(rng):
    rngs = jax.random.split(rng, 5)
    conv1 = init_weights((16, 4, 8, 8), rngs[0])
    conv2 = init_weights((32, 16, 4, 4), rngs[1])
    fc_layer_one = init_weights((2054, 256), rngs[2])
    fc_layer_two = init_weights((256, 1), rngs[3])
    params = conv1, conv2, fc_layer_one, fc_layer_two
    return params

# Actor Network
def actor_model(actor_params, obs_data, obs_images):
    """
    Actor model predicts continuous actions for the current state.
    """
    if obs_images.ndim == 3:
        obs_images = obs_images[None,...]
    conv1, conv2, fc_layer_one, fc_layer_two = actor_params
    x = nn.relu(conv(obs_images, conv1, 4))
    x = nn.relu(conv(x, conv2, 2))
    x = x.reshape(x.shape[0], -1)  # Flatten convolutional features
    x = jnp.concatenate([x, obs_data.reshape(x.shape[0], -1)], axis=1)
    x = jnp.tanh(jnp.dot(x, fc_layer_one))
    output = jnp.dot(x, fc_layer_two)
    # Binary outputs for forward and backward
    forward = nn.sigmoid(output[:, 0])  # Apply sigmoid
    backward = nn.sigmoid(output[:, 1])  # Apply sigmoid

    # Continuous output for steering
    steer = nn.tanh(output[:, 2])  # Range -1 to 1

    # Threshold forward and backward to convert to binary (0 or 1)
    forward = (forward > 0.5).astype(jnp.float32)
    backward = (backward > 0.5).astype(jnp.float32)

    return jnp.stack([forward, backward, steer], axis=-1)
    #action = output = jnp.tanh(jnp.dot(x, fc_layer_two))
    #return action

# Critic Network
def critic_model(critic_params, obs_data, obs_images, actions):
    """
    Critic model predicts Q-values for a state-action pair (s, a).
    """
    if obs_images.ndim == 3:
        obs_images = obs_images[None,...]
    conv1, conv2, fc_layer_one, fc_layer_two = critic_params
    x = nn.relu(conv(obs_images, conv1, 4))
    x = nn.relu(conv(x, conv2, 2))
    x = x.reshape(x.shape[0], -1)  # Flatten convolutional features
    x = jnp.concatenate([x, obs_data.reshape(x.shape[0], -1), actions.reshape(x.shape[0], -1)], axis=1)
    x = nn.tanh(jnp.dot(x, fc_layer_one))
    q_value = jnp.dot(x, fc_layer_two)  # Output Q-value
    return q_value  # Shape: (batch_size, 1)

# Loss function for the critic
def bellman_loss(critic_params, target_critic_params, actor_params, batch):
    """
    Bellman loss for the critic network.
    """
    # Unpack batch
    obs_data, obs_images, actions, rewards, next_obs_data, next_obs_images, dones = batch

    # Predicted Q-values for current state-action pairs
    q_values = critic_model(critic_params, obs_data, obs_images, actions)
    
    # Predict next actions using the actor
    next_actions = actor_model(actor_params, next_obs_data, next_obs_images)

    # Target Q-values using the target critic and next state-action pairs
    target_q_values = critic_model(target_critic_params, next_obs_data, next_obs_images, next_actions)

    # Compute the Bellman targets
    targets = rewards + gamma * target_q_values.squeeze() * (1.0 - dones)

    # Critic loss: Mean squared error between predicted and target Q-values
    loss = jnp.mean((q_values.squeeze() - targets) ** 2)

    return loss

# Loss function for the actor
def policy_loss(actor_params, critic_params, batch, entropy_weight=0.01, steer_reg_weight=0.01):
    """
    Policy loss for the actor network with entropy and steering regularization.
    """
    # Unpack batch
    obs_data, obs_images, _, _, _, _, _ = batch

    # Predict actions using the actor
    predicted_actions = actor_model(actor_params, obs_data, obs_images)

    # Critic evaluates Q-values for the current policy
    q_values = critic_model(critic_params, obs_data, obs_images, predicted_actions)

    # Advantage calculation: Q-value - mean Q-value (or baseline)
    advantage = q_values - jnp.mean(q_values)

    # Policy loss: Maximize advantage by minimizing the negative advantage
    policy_loss = -jnp.mean(advantage)

    # Entropy regularization: Encourage exploration
    entropy = -jnp.sum(predicted_actions * jnp.log(jnp.clip(predicted_actions, 1e-8, 1)), axis=-1)
    entropy_loss = -entropy_weight * jnp.mean(entropy)

    # Steering regularization: Penalize extreme steering values
    steer_reg_loss = steer_reg_weight * jnp.mean((predicted_actions[:, 2]) ** 2)

    # Combine losses
    loss = policy_loss + entropy_loss + steer_reg_loss
    return loss


# Policy with epsilon-greedy for continuous actions
def policy_fn(params, rng, obs_data, obs_images, epsilon=0.1):
    rng, key = random.split(rng)
    if random.uniform(key) < epsilon:
        # Random action
        return jnp.array([np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(-1, 1)])
    else:
        action = actor_model(params, obs_data, obs_images)
        return action[0]

# Training step with gradient descent
@jit
def update_actor(actor_params, critic_params, actor_opt_state, batch, entropy_weight = 0.01, steer_reg_weight = 0.01):
    loss, gradients = jax.value_and_grad(policy_loss)(actor_params, critic_params, batch, entropy_weight, steer_reg_weight)
    updates, actor_opt_state = actor_optimizer.update(gradients, actor_opt_state, actor_params)
    actor_params = optax.apply_updates(actor_params, updates)
    return actor_params, actor_opt_state, loss

@jit
def update_critic(critic_params, target_critic_params, actor_params, critic_opt_state, batch):
    loss, gradients = jax.value_and_grad(bellman_loss)(critic_params, target_critic_params, actor_params, batch)
    updates, critic_opt_state = critic_optimizer.update(gradients, critic_opt_state, critic_params)
    critic_params = optax.apply_updates(critic_params, updates)
    return critic_params, critic_opt_state, loss

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
def update_target_network(params, target_params, tau=0.005):
    return tree_util.tree_map(lambda p, tp: tau * p + (1 - tau) * tp, params, target_params)

# %%
# Main training loop
rng = random.PRNGKey(0)
output_size = 3
# Initialize actor, critic, and their target networks
actor_params = initialize_params_actor(rng)
critic_params = initialize_params_critic(rng)
target_critic_params = critic_params

# Set up optimizers for actor and critic
actor_optimizer = optax.adam(learning_rate * 0.1)
critic_optimizer = optax.adam(learning_rate)
actor_opt_state = actor_optimizer.init(actor_params)
critic_opt_state = critic_optimizer.init(critic_params)
rewards = []
highestReward = 0
tau = 0.05

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
    obs_data, obs_images = preprocess_obs(obs)
    terminated = False
    truncated = False
    first = True
    print(f"Episode: {episode}")
    while not (terminated | truncated):
        rng, key = random.split(rng)
        epsilon = max(0.01, 1.0 - episode / 100)
        entropy_weight = max(0.01, 0.1 * (1 - episode / 200))
        action = policy_fn(actor_params, key, obs_data, obs_images, epsilon)
        #action = action.at[0].set(1)
        #action = action.at[1].set(0)
        next_obs, reward, terminated, truncated, info = env.step(action)
        if first:
            reward = 0
        first = False
        done = terminated or truncated
        next_obs_data, next_obs_images = preprocess_obs(next_obs)
        reward = reward + (next_obs_data[0])
        # Store transition in replay buffer
        memory.append(entry(obs_data, obs_images, action, reward, next_obs_data, next_obs_images, done))
        total_reward += reward
        obs_data, obs_images = next_obs_data, next_obs_images

        if done:
            
            batch = sample_batch(memory)
            
            # Update critic
            critic_params, critic_opt_state, critic_loss = update_critic(
                critic_params, target_critic_params, actor_params, critic_opt_state, batch
            )
            
            print(f"Critic Loss: {critic_loss:.6f}")
            
            # Update actor
            actor_params, actor_opt_state, actor_loss = update_actor(
                actor_params, critic_params, actor_opt_state, batch, 0.1, 0.1
            )
            
            print(f"Actor Loss: {actor_loss:.6f}")
            
            # Soft update target networks
            target_critic_params = update_target_network(critic_params, target_critic_params, tau)
            
            if(highestReward < total_reward):
                    best_actor_params = actor_params
                    best_target_critic_params = target_critic_params
                    best_critic_params = critic_params
                    highestReward = total_reward
                    print("New record:" , highestReward)
            
            memory.clear()
            rewards.append(total_reward)
            
            q_value = critic_model(critic_params, obs_data, obs_images, action)
            print(f"Reward: {reward}, Q-values: {q_value}")
            print(f"Action: {action}")
            model_action = actor_model(actor_params, obs_data, obs_images)
            print(f"Model Action: {model_action}")
            break

# %% ------------------- 4. Graph -------------------
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Progress')
plt.show()
# %% -------------------  Params Saver ------------------
with open(filenameCritic, 'wb') as f:
    pickle.dump(best_critic_params, f)
with open(filenameCriticTarget, 'wb') as f:
    pickle.dump(best_target_critic_params, f)
with open(filenameActor, 'wb') as f:
    pickle.dump(best_actor_params, f)

# # %% ------------------- Params Loader -------------------
with open(filenameCritic, 'rb') as f:
     loaded_critic_params = pickle.load(f)
with open(filenameCriticTarget, 'rb') as f:
     loaded_critic_target_params = pickle.load(f)
with open(filenameActor, 'rb') as f:
     loaded_actor_params = pickle.load(f)     

# %%
critic_params = loaded_critic_params
target_critic_params = loaded_critic_target_params
actor_params = loaded_actor_params

# %% use best params
critic_params = best_critic_params
target_critic_params = best_target_critic_params
actor_params = best_actor_params