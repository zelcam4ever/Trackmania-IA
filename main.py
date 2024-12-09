# %%
import gymnasium as gym
from tmrl import get_environment
from collections import deque, namedtuple
from jax import random, grad, jit, tree_util, nn, lax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
import pickle


# %% Setup replay buffer and environment
entry = namedtuple("Memory", ["obs", "action", "reward", "next_obs", "done"])
memory = deque(maxlen=10000)  # Replay buffer
gamma = 0.99  # Discount factor
learning_rate = 0.001
env = get_environment()
filenameCritic = "criticParams.pkl"
filenameCriticTarget = "criticTargetParams.pkl"
filenameActor = "actorParams.pkl"

# %% Initialize network parameters

def conv(input, kernel, stride = 1, padding="SAME"):
    return lax.conv(input, kernel, (stride, stride), padding=padding)

def initialize_mlp_params(rng, input_size, hidden_sizes, output_size):
    def init_layer_params(m, n, rng):
        w_key, b_key = random.split(rng)
        weight = random.normal(w_key, (m, n)) * jnp.sqrt(0.1 / m)
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

def forward_mlp_actor(params, x):
    """
    Forward pass for actor network. Produces mean and log_std for 3 action dimensions:
    - Forward: [0, 1]
    - Backward: [0, 1]
    - Steer: [-1, 1]
    """
    activations = x
    for w, b in params[:-1]:
        activations = nn.leaky_relu(jnp.dot(activations, w) + b)  # Hidden layers

    final_w, final_b = params[-1]
    output = jnp.dot(activations, final_w) + final_b

    if output.ndim == 1:
        output = output[None,...]
    
    # Split into mean and log_std (3 for each dimension)
    mean = output[:, :3]  # First 3 are the mean values for the actions
    log_std = jnp.clip(output[:, 3:], -5, 2)  # Last 3 are log_std, clipped for stability

    return mean, log_std

def forward_mlp_critic(params, x, action):
    if action.ndim == 1:
        action = action[None,...]
    if x.ndim == 1:
        x = x[None,...]
    activations = jnp.concatenate([x, action], axis=1) 
    for w, b in params[:-1]:
        activations = nn.leaky_relu(jnp.dot(activations, w) + b)  # For bounded actions
    final_w, final_b = params[-1]
    return jnp.dot(activations, final_w) + final_b

def huber_loss(predicted, target, delta=1.0):
    """
    Compute the Huber loss.
    
    Arguments:
    - predicted: Predicted values.
    - target: Target values.
    - delta: Huber loss delta parameter.
    
    Returns:
    - Mean Huber loss.
    """
    error = predicted - target
    abs_error = jnp.abs(error)
    quadratic = jnp.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return jnp.mean(0.5 * quadratic ** 2 + delta * linear)

def compute_l2_regularization(params, lambda_reg = 0.1):
    l2_reg = 0.0
    for i in [0,1,2]:
        w, _ = params[i]  # Weights are stored in these indices
        l2_reg += jnp.sum(w ** 2)  # L2 regularization for weights
    return l2_reg * lambda_reg

def bellman_loss(critic_params, target_critic_params_list, actor_params, batch, gamma=0.99):
    """
    Compute Bellman loss for a single critic.
    
    Arguments:
    - critic_params: Parameters of the critic network.
    - target_critic_params_list: List of target critic parameters.
    - actor_params: Parameters of the actor network.
    - batch: Batch of experience (obs, actions, rewards, next_obs, dones).
    - gamma: Discount factor.
    
    Returns:
    - Bellman loss.
    """
    obs, actions, rewards, next_obs, dones = batch

    # Predict Q-values
    predicted_q_values = forward_mlp_critic(critic_params, obs, actions)

    # Compute target Q-values
    next_actions = get_action_for_inference(actor_params, next_obs)
    target_q_values = jnp.min(
        jnp.stack([
            forward_mlp_critic(target_params, next_obs, next_actions)
            for target_params in target_critic_params_list
        ]),
        axis=0
    )
    targets = rewards + gamma * target_q_values * (1.0 - dones)

    # Compute Huber loss
    loss = huber_loss(predicted_q_values.squeeze(), targets)
    loss += compute_l2_regularization(critic_params)
    return loss


def compute_log_probs(mean, log_std, sampled_actions):
    """
    Compute the log probability of the sampled actions given the Gaussian distribution
    parameterized by the mean and log_std.

    Arguments:
    - mean: Mean of the action distribution (batch_size, action_dim).
    - log_std: Log standard deviation of the action distribution (batch_size, action_dim).
    - sampled_actions: Sampled actions taken by the agent (batch_size, action_dim).

    Returns:
    - log_probs: Log probability of each action in the batch (batch_size,).
    """
    # Calculate the standard deviation from log_std
    std = jnp.exp(log_std)

    # Compute the log probability of the sampled actions under a Gaussian distribution
    log_probs = -0.5 * jnp.sum(
        jnp.log(2 * jnp.pi * std**2) + ((sampled_actions - mean)**2) / (std**2), axis=-1
    )

    return log_probs



def policy_loss(actor_params, critic_params_list, batch, key, alpha=0.1):
    """
    Compute the actor's policy loss.
    
    Arguments:
    - actor_params: Parameters of the actor network.
    - critic_params_list: List of critic parameters.
    - batch: Batch of experience (obs, actions, rewards, next_obs, dones).
    - key: Random number generator key.
    - alpha: Entropy regularization coefficient.
    
    Returns:
    - Policy loss.
    """
    obs, _, _, _, _ = batch

    # Actor forward pass
    mean, log_std = forward_mlp_actor(actor_params, obs)
    std = jnp.exp(log_std)

    # Reparameterization trick
    sampled_actions = mean + std * random.normal(key, mean.shape)

    # Apply action bounds (e.g., sigmoid and tanh)
    bounded_actions = sampled_actions.at[:, :2].set(nn.sigmoid(sampled_actions[:, :2]))
    bounded_actions = bounded_actions.at[:, 2].set(nn.tanh(sampled_actions[:, 2]))

    # Compute log probabilities
    log_probs = compute_log_probs(mean, log_std, sampled_actions)

    # Compute Q-values from all critics
    q_values = jnp.min(
        jnp.stack([forward_mlp_critic(critic_params, obs, bounded_actions) for critic_params in critic_params_list]),
        axis=0
    )

    # Policy loss: maximize Q-values and entropy
    loss = jnp.mean(alpha * log_probs - q_values)
    loss += compute_l2_regularization(actor_params)
    return loss



def get_action_for_inference(actor_params, obs):
    """
    Get the deterministic action from the actor for inference.
    """
    mean, _ = forward_mlp_actor(actor_params, obs)  # Only the mean is needed for deterministic inference

    # Apply sigmoid/tanh constraints
    forward = nn.sigmoid(mean[:, 0])  # Forward: [0, 1]
    backward = nn.sigmoid(mean[:, 1])  # Backward: [0, 1]
    steer = nn.tanh(mean[:, 2])  # Steer: [-1, 1]

    action = jnp.stack([forward, backward, steer], axis=-1)
    return action


# Policy with epsilon-greedy for continuous actions
def policy_fn(params, rng, obs, epsilon=0.1):
    rng, key = random.split(rng)
    if random.uniform(key) < epsilon:
        
        # Random action
        action = jnp.array([np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(-1, 1)])
        return action
    else:
        action = get_action_for_inference(params, obs)
        return action[0]  # Keeping gas and brake constant here

# Training step with gradient descent
@jit
def update_actor(actor_params, critic_params_list, opt_state, batch, key):
    loss, gradients = jax.value_and_grad(policy_loss)(actor_params, critic_params_list, batch, key)
    updates, opt_state = actor_optimizer.update(gradients, opt_state, actor_params)
    actor_params = optax.apply_updates(actor_params, updates)
    return actor_params, opt_state, loss

@jit
def update_critic(
    critic_params_list,
    target_critic_params_list,
    actor_params,
    opt_state_critic_list,
    batch,
    tau=0.005,
):
    """
    Update critics individually using separate gradient and loss computations.
    """
    updated_critic_params_list = []
    updated_opt_state_critic_list = []
    critic_losses = []

    for critic_idx, (critic_params, opt_state) in enumerate(
        zip(critic_params_list, opt_state_critic_list)
    ):
        # Compute gradients and loss for this critic
        loss, grads = jax.value_and_grad(bellman_loss)(
            critic_params, target_critic_params_list, actor_params, batch
        )

        # Update critic parameters
        updates, new_opt_state = critic_optimizer.update(grads, opt_state, critic_params)
        updated_critic_params = optax.apply_updates(critic_params, updates)

        # Store results
        updated_critic_params_list.append(updated_critic_params)
        updated_opt_state_critic_list.append(new_opt_state)
        critic_losses.append(loss)

    # Update target critics
    updated_target_critic_params_list = update_target_critic_networks(
        updated_critic_params_list, target_critic_params_list, tau
    )

    # Return results
    return (
        updated_critic_params_list,
        updated_opt_state_critic_list,
        jnp.mean(jnp.array(critic_losses)),  # Average critic loss
        updated_target_critic_params_list,
    )


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

def update_target_critic_networks(critic_params_list, target_critic_params_list, tau=0.005):
    """
    Soft update for all critic target networks using JAX tree_map.
    
    Arguments:
    - critic_params_list: List of parameters for all critics.
    - target_critic_params_list: List of parameters for all target critics.
    - tau: Soft update coefficient.
    
    Returns:
    - Updated target critic parameters list.
    """
    # Perform soft updates using tree_map for each critic
    updated_target_critic_params_list = [
        jax.tree.map(lambda tp, cp: tau * cp + (1 - tau) * tp, target_params, critic_params)
        for target_params, critic_params in zip(target_critic_params_list, critic_params_list)
    ]
    
    return updated_target_critic_params_list


# %%
# Main training loop
rng = random.PRNGKey(0)
input_size = 83  # Based on Trackmania observations (flattened)
hidden_sizes = [256, 256, 128]
output_size = 6
actor_params = initialize_mlp_params(rng, input_size, hidden_sizes, output_size)

# Set up optimizer
actor_optimizer = optax.adam(0.00001)
actor_opt_state = actor_optimizer.init(actor_params)

# Variable for the number of critics
num_critics = 10  # Change this value as needed

# Initialize a single optimizer
critic_optimizer = optax.adam(0.00005)

# Initialize lists to hold parameters, target parameters, and optimizer states
critic_params_list = []
target_critic_params_list = []
critic_opt_state_list = []

# Loop to initialize each critic's parameters and optimizer state
for i in range(num_critics):
    # Initialize critic parameters
    critic_params = initialize_mlp_params(rng, input_size + 3, hidden_sizes, 1)
    target_critic_params = critic_params.copy()

    # Initialize optimizer state for each critic
    critic_opt_state = critic_optimizer.init(critic_params)

    # Append to lists
    critic_params_list.append(critic_params)
    target_critic_params_list.append(target_critic_params)
    critic_opt_state_list.append(critic_opt_state)

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

memory.clear()

for episode in range(1):  # rtgym ensures this runs at 20Hz by default
    obs, info = env.reset()
    total_reward = 0
    obs = preprocess_obs(obs)
    t = 0
    terminated = False
    truncated = False
    actor_loss = 0
    critic_loss = 0
    print(f"Episode: {episode}")
    while not (terminated | truncated):
        t += 1
        rng, key = random.split(rng)
        #epsilon = max(0.8, 1.0 - episode / 500)
        epsilon = 0
        action = policy_fn(actor_params, key, obs, epsilon)
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_obs = preprocess_obs(next_obs)
        # Store transition in replay buffer
        memory.append(entry(obs, action, reward, next_obs, done))
        total_reward += reward
        obs = next_obs
        
        if len(memory) >= 1000 and t % 4 == 0:
            rngs = random.split(rng, 3)
            batch = sample_batch(rngs[0], memory, 256)
            critic_params_list, critic_opt_state_list, critic_loss, target_critic_params_list = update_critic(critic_params_list, target_critic_params_list, actor_params, critic_opt_state_list, batch)
            if t % 40 == 0:
                actor_params, actor_opt_state, actor_loss = update_actor(actor_params, critic_params_list, actor_opt_state, batch, rngs[1])

        if done:
            rewards.append(total_reward)
            if(highestReward < total_reward):
                    best_actor_params = actor_params.copy()
                    best_critic_params_list = critic_params_list.copy()
                    best_target_critic_params_list = target_critic_params_list.copy()
                    highestReward = total_reward
                    print("New record:" , highestReward)
            print(f"Critic Loss: {critic_loss:.6f}")
            print(f"Actor Loss: {actor_loss:.6f}")
            q_value = forward_mlp_critic(critic_params_list[0], obs, action)
            print(f"Reward: {reward}, Q-values: {q_value}")
            print(f"Action: {action}")
            model_action = get_action_for_inference(actor_params, obs)
            print(f"Model Action: {model_action[0]}")
            break

# %% ------------------- 4. Graph -------------------
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Progress')
plt.show()
# %% -------------------  Params Saver -------------------
with open(filenameCritic, 'wb') as f:
    pickle.dump(critic_params_list, f)
with open(filenameCriticTarget, 'wb') as f:
    pickle.dump(target_critic_params_list, f)
with open(filenameActor, 'wb') as f:
    pickle.dump(actor_params, f)

# %% ------------------- Params Loader -------------------
with open(filenameCritic, 'rb') as f:
     loaded_critic_params = pickle.load(f)
with open(filenameCriticTarget, 'rb') as f:
     loaded_critic_target_params = pickle.load(f)
with open(filenameActor, 'rb') as f:
     loaded_actor_params = pickle.load(f)   

# %% use best params
critic_params = best_critic_params_list
target_critic_params = best_target_critic_params_list
actor_params = best_actor_params


# %%
critic_params_list = loaded_critic_params
target_critic_params_list = loaded_critic_target_params
actor_params = loaded_actor_params
# %% Clear memory

memory.clear()

# %%
filenameCritic = "criticParams_v2.pkl"
filenameCriticTarget = "criticTargetParams_v2.pkl"
filenameActor = "actorParams_v2.pkl"