# %%
import gymnasium as gym
from tmrl import get_environment
from collections import deque, namedtuple
from jax import random, grad, jit, tree_util, nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
import pickle
import tqdm as tqdm


# %% Setup replay buffer and environment
entry = namedtuple("Memory", ["obs", "action", "reward", "next_obs", "done"])
memory = deque(maxlen=50000)  # Replay buffer
gamma = 0.99  # Discount factor
learning_rate = 0.0001
env = get_environment()
filename = "BestParams.pkl"
filenameTarget = "BestTargetParams.pkl"
filename_completed = "LastCompletedParams.pkl"
filenameTarget_completed = "LastCompletedTargetParams.pkl"

# %% Initialize network parameters
def initialize_mlp_params(rng, input_size, hidden_sizes, output_size):
    def init_layer_params(m, n, rng):
        w_key, b_key = random.split(rng)
        weight = random.normal(w_key, (m, n)) * jnp.sqrt(2.0 / m)
        bias = jnp.ones(n) * 0.1
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

def forward_mlp_actor(params, x):
    """
    Forward pass for actor network. Produces mean and log_std for 3 action dimensions:
    - Forward: [0, 1]
    - Backward: [0, 1]
    - Steer: [-1, 1]
    """
    activations = x
    for w, b in params[:-1]:
        activations = nn.relu(jnp.dot(activations, w) + b)  # Hidden layers

    final_w, final_b = params[-1]
    output = jnp.dot(activations, final_w) + final_b

    return output


def compute_l2_regularization(params, lambda_reg = 0.1):
    l2_reg = 0.0
    for i in [0,1,2,3]:
        w, _ = params[i]  # Weights are stored in these indices
        l2_reg += jnp.sum(w ** 2)  # L2 regularization for weights
    return l2_reg * lambda_reg

def huber_loss(error, delta=10.0):
    return jnp.where(jnp.abs(error) <= delta,
                     0.5 * error**2,
                     delta * (jnp.abs(error) - 0.5 * delta))

def bellman_loss(params, target_params, batch, gamma=0.95):
    """
    Compute the Bellman loss for Deep Q-Learning.

    Parameters:
        params: The parameters of the current Q-network.
        target_params: The parameters of the target Q-network.
        batch: A tuple (obs, actions, rewards, next_obs, dones).
               - obs: Observations of shape (batch_size, observation_dim).
               - actions: One-hot encoded actions of shape (batch_size, num_actions).
               - rewards: Rewards of shape (batch_size,).
               - next_obs: Next observations of shape (batch_size, observation_dim).
               - dones: Done flags of shape (batch_size,).
        gamma: Discount factor (default: 0.99).

    Returns:
        Scalar loss value.
    """
    obs, actions, rewards, next_obs, dones = batch

    # Compute Q-values for the current state
    q_values = forward_mlp_actor(params, obs)  # Shape: (batch_size, num_actions)
    
    # Extract Q-values for the chosen actions
    action_indices = jnp.argmax(actions, axis=1)  # Convert one-hot to indices
    chosen_q_values = jnp.take_along_axis(q_values, action_indices[:, None], axis=1).squeeze()  # Shape: (batch_size,)

    # Compute Q-values for the next state using the target network
    next_q_values = forward_mlp_actor(target_params, next_obs)  # Shape: (batch_size, num_actions)

    # Clip Q-values to avoid negatives
    next_q_values = jnp.maximum(next_q_values, 0.0)
    
    # Compute the target Q-values
    target_q_values = rewards + gamma * jnp.max(next_q_values, axis=1) * (1.0 - dones)  # Shape: (batch_size,)

    # Compute Bellman error
    loss = jnp.mean(huber_loss(chosen_q_values, target_q_values))

    loss += compute_l2_regularization(params, lambda_reg= 0.001)
    return loss




# Policy with epsilon-greedy for continuous actions
def policy_fn(params, rng, obs, epsilon=0.1):
    rng, key = random.split(rng)
    if random.uniform(key) < epsilon:
        
        # Random action
        action = jnp.eye(11)[random.randint(rng, (), 0, 11)]
        return action
    else:
        action = forward_mlp_actor(params, obs)
        action = convert_action(action)
        return action

# Training step with gradient descent
@jit
def update_actor(params, target_params, opt_state, batch):
    loss, gradients = jax.value_and_grad(bellman_loss)(params, target_params, batch)
    updates, opt_state = actor_optimizer.update(gradients, opt_state, params)
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
    speed = obs[0].flatten()         # Flatten speed array
    lidar = obs[1].flatten()            # Flatten LIDAR array
    prev_actions = obs[2].flatten()    # Flatten previous action arrays
    prev_actions_2 = obs[3].flatten()  # Flatten second previous action array

    # Concatenate all flattened arrays into a single 1D array
    processed_obs = jnp.concatenate([speed, lidar, prev_actions, prev_actions_2])
    return processed_obs

def soft_update(target_params, source_params, polyak=0.995):
    return jax.tree_util.tree_map(lambda t, s: polyak * t + (1 - polyak) * s, target_params, source_params)


import numpy as np

def map_action(array):
    """
    Maps a one-hot encoded action array to the game's action space.
    
    Parameters:
        array (np.ndarray): A 1D array with a single `1` and the rest `0`s.
        
    Returns:
        tuple: A tuple (forward, backward, steer) representing the action.
               - forward (int): 0 or 1
               - backward (int): 0 or 1
               - steer (int): -1, 0, or 1
    """
    if array.sum() != 1 or array.ndim != 1 or len(array) != 11:
        raise ValueError("Input array must be 1D, of length 11, and contain exactly one `1`.")
    
    # Define the mapping: index -> (forward, backward, steer)
    action_map = jnp.array([
        [1, 0,  0],  # Forward, no turning
        [0, 1,  0],  # Backward, no turning
        [0, 0,  1],  # No movement, turn right
        [0, 0, -1],  # No movement, turn left
        [1, 0,  1],  # Forward, turn right
        [1, 0, -1],  # Forward, turn left
        [0, 1,  1],  # Backward, turn right
        [0, 1, -1],  # Backward, turn left
        [1, 1,  1],   #Drift right
        [1, 1,  0],   #Drift no steer
        [1, 1, -1]   #Drift Left
    ])
    
    # Get the index of the `1` in the array
    action_index = np.argmax(array)
    
    # Map the index to the corresponding action
    return action_map[action_index]

def convert_action(action):
    """
    Converts a Q-value array into a one-hot encoded action.
    
    Parameters:
        action (jnp.ndarray): Array of Q-values (1D).
        
    Returns:
        jnp.ndarray: One-hot encoded action (same shape as input).
    """
    # Find the index of the maximum Q-value
    index = jnp.argmax(action)
    
    # Create a one-hot encoded array
    one_hot_action = jnp.zeros_like(action).at[index].set(1)
    
    return one_hot_action

# %%
# Main training loop
rng = random.PRNGKey(0)
input_size = 83  # Based on Trackmania observations (flattened)
hidden_sizes = [265, 256, 128]
output_size = 11
actor_params = initialize_mlp_params(rng, input_size, hidden_sizes, output_size)
target_actor_params = actor_params.copy()
# Set up optimizer
actor_optimizer = optax.adam(0.001)
actor_opt_state = actor_optimizer.init(actor_params)


rewards = []
highestReward = 0
max_speed = 20
max_lidar = 70

last_completed_params = actor_params.copy
last_completed_target_params = target_actor_params.copy

# %%

memory.clear()

for episode in range(101):  # rtgym ensures this runs at 20Hz by default
    obs, info = env.reset()
    total_reward = 0
    obs = preprocess_obs(obs)
    t = 0
    terminated = False
    truncated = False
    loss = 0
    print(f"Episode: {episode}")
    if episode % 10 == 0 and episode != 0:
        # Save bets params
        with open(filename, 'wb') as f:
            pickle.dump(best_actor_params, f)
        with open(filenameTarget, 'wb') as f:
            pickle.dump(best_target_actor_params, f)
        with open(filename_completed, 'wb') as f:
            pickle.dump(last_completed_params, f)
        with open(filenameTarget_completed, 'wb') as f:
            pickle.dump(last_completed_target_params, f)
        print("Params saved")

    if episode % 50 == 0 and episode != 0:
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Training Progress')
        plt.show()

    while not (terminated | truncated):
        t += 1
        rng, key = random.split(rng)
        epsilon = max(0.1, 1.0 - episode / 1000)
        action = policy_fn(actor_params, key, obs, epsilon)
        game_action = map_action(action)
        
        next_obs, reward, terminated, truncated, info = env.step(game_action)
        done = terminated or truncated
        next_obs = preprocess_obs(next_obs)
        if reward < 100:
            reward *= 10
        if reward >= 100:
            last_completed_params = actor_params.copy()
            last_completed_target_params = target_actor_params.copy()

        # Store transition in replay buffer
        memory.append(entry(obs, action, reward, next_obs, done))
        total_reward += reward
        obs = next_obs

        if done:
            rewards.append(total_reward)
            if(highestReward <= total_reward):
                    best_actor_params = actor_params.copy()
                    best_target_actor_params = target_actor_params.copy()
                    highestReward = total_reward
                    print("New record:" , highestReward)
            if len(memory) >= 1000:
                #Train if more than 1000 samples collected.
                for i in range(100):
                    rngs = random.split(rng, 3)
                    batch = sample_batch(rngs[0], memory, 256)
                    actor_params, actor_opt_state, loss = update_actor(actor_params, target_actor_params, actor_opt_state, batch)
                    target_actor_params = soft_update(target_actor_params, actor_params)
                print(f"Loss: {loss:.6f}")
                print(f"q_values:", forward_mlp_actor(actor_params, obs))
                model_action = convert_action(forward_mlp_actor(actor_params, obs))
                model_action = map_action(model_action)
                print(f"Model Action: {model_action}")
                print(f"Total Reward: {total_reward}")
            break
        
# %% Train model on Memory

for i in range(10000):
    rngs = random.split(rng, 3)
    batch = sample_batch(rngs[0], memory, 256)
    actor_params, actor_opt_state, loss = update_actor(actor_params, target_actor_params, actor_opt_state, batch)
    target_actor_params = soft_update(target_actor_params, actor_params)

print(f"Loss: {loss:.6f}")
print(f"q_values:", forward_mlp_actor(actor_params, obs))
model_action = convert_action(forward_mlp_actor(actor_params, obs))
model_action = map_action(model_action)
print(f"Model Action: {model_action}")

#Save params after training
with open(filename, 'wb') as f:
    pickle.dump(actor_params, f)
with open(filenameTarget, 'wb') as f:
    pickle.dump(target_actor_params, f)


# %% ------------------- 4. Graph -------------------
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Progress')
plt.show()


# %% ------------------- Params Loader -------------------
with open(filename, 'rb') as f:
     loaded_actor_params = pickle.load(f)
with open(filenameTarget, 'rb') as f:
     loaded_target_actor_params = pickle.load(f)

# %%

actor_params = loaded_actor_params
target_actor_params = loaded_target_actor_params
actor_opt_state = actor_optimizer.init(actor_params)
# %% Params loader for last completed

with open(filename_completed, 'rb') as f:
     loaded_actor_params = pickle.load(f)
with open(filenameTarget_completed, 'rb') as f:
     loaded_target_actor_params = pickle.load(f)

# %%

actor_params = loaded_actor_params
target_actor_params = loaded_target_actor_params
actor_opt_state = actor_optimizer.init(actor_params)
# %%
