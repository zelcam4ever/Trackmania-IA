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


# %% Setup replay buffer and environment
entry = namedtuple("Memory", ["obs", "action", "reward", "next_obs", "done"])
memory = deque(maxlen=1000000)  # Replay buffer
gamma = 0.95  # Discount factor
learning_rate = 0.001
env = get_environment()
filenameCritic = "criticParams.pkl"
filenameCriticTarget = "criticTargetParams.pkl"
filenameActor = "actorParams.pkl"

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

def initialize_mlp_critic_params(rng, input_size, hidden_sizes, output_size):
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

def forward_mlp(params, x):
    """
    Forward pass for actor network. Produces mean and log_std for 3 action dimensions:
    - Forward: [0, 1]
    - Backward: [0, 1]
    - Steer: [-1, 1]
    """
    activations = x
    for w, b in params[:-1]:
        activations = nn.tanh(jnp.dot(activations, w) + b)  # Hidden layers

    final_w, final_b = params[-1]
    output = jnp.dot(activations, final_w) + final_b

    if output.ndim == 1:
        output = output[None,...]
    
    # Split into mean and log_std (3 for each dimension)
    mean = output[:, :3]  # First 3 are the mean values for the actions
    log_std = jnp.clip(output[:, 3:], -20, 2)  # Last 3 are log_std, clipped for stability

    # Apply constraints to the mean
    mean_forward = nn.sigmoid(mean[:, 0])  # Forward: [0, 1]
    mean_backward = nn.sigmoid(mean[:, 1])  # Backward: [0, 1]
    mean_steer = nn.tanh(mean[:, 2])  # Steer: [-1, 1]

    mean = jnp.stack([mean_forward, mean_backward, mean_steer], axis=-1)

    return mean, log_std

def forward_mlp_critic(params, x, action):
    if action.ndim == 1:
        action = action[None,...]
    if x.ndim == 1:
        x = x[None,...]
    activations = jnp.concatenate([x, action], axis=1) 
    for w, b in params[:-1]:
        activations = nn.tanh(jnp.dot(activations, w) + b)  # For bounded actions
    final_w, final_b = params[-1]
    return jnp.dot(activations, final_w) + final_b

def huber_loss(x, delta=1.0):
    """Compute the Huber loss between predicted Q-values and targets."""
    abs_error = jnp.abs(x)
    loss = jnp.where(abs_error <= delta, 0.5 * abs_error ** 2, delta * (abs_error - 0.5 * delta))
    return jnp.mean(loss)

def compute_l2_regularization(params):
    l2_reg = 0.0
    # Weights are located at indices 1, 3, 5 (0-based indexing)
    for i in [len(params)]:
        w, _ = params[i]  # Weights are stored in these indices
        l2_reg += jnp.sum(w ** 2)  # L2 regularization for weights
    return l2_reg

def bellman_loss(
    critic_params_list,
    target_critic_params_list,
    actor_params,
    batch,
    gamma=0.99,
    tau=0.005,
):
    """
    Bellman loss for multiple critics in SAC, returning a loss for each critic.
    """
    # Unpack batch
    obs, actions, rewards, next_obs, dones = batch

    # Compute predicted Q-values from all critics
    q_values_list = [forward_mlp_critic(params, obs, actions) for params in critic_params_list]

    # Predict next actions using the actor (for next state)
    next_actions = get_action_for_inference(actor_params, next_obs)

    # Compute target Q-values from all target critics
    target_q_values_list = [
        forward_mlp_critic(params, next_obs, next_actions)
        for params in target_critic_params_list
    ]

    # Take the minimum of the target Q-values across all critics (clipped double Q-learning)
    target_q_values_min = jnp.min(jnp.stack(target_q_values_list, axis=0), axis=0)

    # Clip target Q-values to prevent instability
    target_q_values_min = jnp.clip(target_q_values_min, -1e2, 1e2)

    # Compute the Bellman targets
    targets = rewards + gamma * target_q_values_min * (1.0 - dones)

    # Compute Huber loss for each critic
    
    critic_losses = []
    
    for q_values in q_values_list:
        loss = huber_loss(q_values.squeeze() - targets)
        critic_losses.append(loss)


    # Return a list of losses, one for each critic
    return jnp.mean(jnp.array(critic_losses))




def compute_log_probs(mean, log_std, actions):
    """
    Compute log probabilities of actions under the transformed Gaussian policy.
    Handles sigmoid and tanh transformations for constrained action spaces.
    """
    epsilon = 1e-6  # Small epsilon to avoid division by zero
    std = jnp.exp(log_std) + epsilon  # Ensure std is not zero
    pre_squash_actions = (actions - mean) / std  # Actions before activation functions

    # Log-probabilities for Gaussian distribution
    log_probs_gaussian = -0.5 * (pre_squash_actions ** 2 + 2 * log_std + jnp.log(2 * jnp.pi))

    # Sum log probabilities across action dimensions
    log_probs = jnp.sum(log_probs_gaussian, axis=-1)

    # Adjustment for the transformations:
    # Sigmoid for forward/backward
    actions_clipped = jnp.clip(actions[:, :2], epsilon, 1 - epsilon)
    log_probs -= jnp.sum(jnp.log(actions_clipped * (1 - actions_clipped) + epsilon), axis=-1)
    
    # Tanh for steer
    actions_clipped_steer = jnp.clip(actions[:, 2], -1 + epsilon, 1 - epsilon)
    log_probs -= jnp.log(1 - actions_clipped_steer ** 2 + epsilon)

    return log_probs

def policy_loss(actor_params, critic_params_list, batch, key, alpha = 0.1):
    """
    Policy loss for the actor with entropy regularization, using multiple critics.
    
    Arguments:
    - actor_params: Parameters for the actor network.
    - critic_params_list: List of parameters for all critics.
    - batch: Batch of experience (obs, actions, rewards, next_obs, dones).
    - key: JAX PRNG key for sampling.
    - alpha: Entropy coefficient.
    
    Returns:
    - Policy loss for the actor.
    """
    # Unpack batch
    obs, _, _, _, _ = batch

    # Actor outputs mean and log_std
    mean, log_std = forward_mlp(actor_params, obs)
    std = jnp.exp(log_std)

    # Sample actions using reparametrization trick
    sampled_actions = mean + std * jax.random.normal(key, mean.shape)

    # Compute log probabilities of sampled actions
    log_probs = compute_log_probs(mean, log_std, sampled_actions)

    # Compute Q-values for sampled actions from all critics
    q_values_list = [forward_mlp_critic(critic_params, obs, sampled_actions) for critic_params in critic_params_list]

    # Average the Q-values from all critics
    avg_q_values = jnp.mean(jnp.stack(q_values_list, axis=0), axis=0)

    # Policy loss: maximize Q-values and entropy (minimize negative Q-values and log_probs)
    policy_loss = jnp.mean(alpha * log_probs - avg_q_values)

    return policy_loss

def get_action_for_inference(actor_params, obs):
    """
    Get the action from the actor for inference (no sampling, use the mean).
    """
    # Get the mean and log_std from the actor (mean is deterministic for testing)
    mean, _ = forward_mlp(actor_params, obs)  # We only need the mean, not the log_std
    
    # For inference, use the mean directly (no sampling, no noise)
    forward = mean[:, 0]  # Action for forward (range [0, 1])
    backward = mean[:, 1]  # Action for backward (range [0, 1])
    steer = mean[:, 2]  # Action for steer (range [-1, 1])

    # Return the deterministic actions
    action = jnp.stack([forward, backward, steer], axis=-1)
    return action


# Policy with epsilon-greedy for continuous actions
def policy_fn(params, rng, obs, epsilon=0.1):
    rng, key = random.split(rng)
    if random.uniform(key) < epsilon:
        # Random action
        action = jnp.array([np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(-1, 1)])
        #forward = action[0]
        #backward = action[1]
        #forward = (forward > 0.5).astype(jnp.float32)
        #backward = (backward > 0.5).astype(jnp.float32)
        #action = action.at[0].set(forward)
        #action = action.at[1].set(backward)
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
def update_critic(critic_params_list, target_critic_params_list, actor_params, opt_state_critic_list, batch, tau = 0.005):
    # Update critics
    updated_critic_params_list = []
    updated_opt_state_critic_list = []
    loss, gradients = jax.value_and_grad(bellman_loss)(
        critic_params_list, target_critic_params_list, actor_params, batch
    )
    for i, (critic_params, critic_opt_state, critic_grads) in enumerate(zip(
        critic_params_list, opt_state_critic_list, gradients)):
        
        # Apply gradient to update the critic parameters
        updates, new_opt_state = critic_optimizer.update(critic_grads, critic_opt_state, critic_params)
        updated_critic_params = optax.apply_updates(critic_params, updates)
        
        # Append updated parameters and opt state for each critic
        updated_critic_params_list.append(updated_critic_params)
        updated_opt_state_critic_list.append(new_opt_state)

    # Update target critics
    updated_target_critic_params_list = update_target_critic_networks(updated_critic_params_list, target_critic_params_list, tau)

    return (
        updated_critic_params_list,
        updated_opt_state_critic_list,
        loss,  # total loss of critics
        updated_target_critic_params_list
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
    Soft update for all critic target networks.
    
    Arguments:
    - critic_params_list: List of parameters for all critics.
    - target_critic_params_list: List of parameters for all target critics.
    - tau: Soft update coefficient.
    
    Returns:
    - Updated target critic parameters list.
    """
    updated_target_critic_params_list = []

    for critic_params, target_critic_params in zip(critic_params_list, target_critic_params_list):
        # Perform soft update for each layer in the critic's parameter list
        updated_target_critic_params = [
            (tau * w + (1 - tau) * tw, tau * b + (1 - tau) * tb)
            for (w, b), (tw, tb) in zip(critic_params, target_critic_params)
        ]
        updated_target_critic_params_list.append(updated_target_critic_params)
    
    return updated_target_critic_params_list


# %%
# Main training loop
rng = random.PRNGKey(0)
input_size = 83  # Based on Trackmania observations (flattened)
hidden_sizes = [256, 256, 128]
output_size = 6
actor_params = initialize_mlp_params(rng, input_size, hidden_sizes, output_size)

# Set up optimizer
actor_optimizer = optax.adam(learning_rate)
actor_opt_state = actor_optimizer.init(actor_params)

# Variable for the number of critics
num_critics = 10  # Change this value as needed

# Initialize a single optimizer
critic_optimizer = optax.adam(learning_rate)

# Initialize lists to hold parameters, target parameters, and optimizer states
critic_params_list = []
target_critic_params_list = []
critic_opt_state_list = []

# Loop to initialize each critic's parameters and optimizer state
for i in range(num_critics):
    # Initialize critic parameters
    critic_params = initialize_mlp_critic_params(rng, input_size + 3, hidden_sizes, 1)
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

for episode in range(10):  # rtgym ensures this runs at 20Hz by default
    obs, info = env.reset()
    total_reward = 0
    obs = preprocess_obs(obs)
    t = 0
    terminated = False
    truncated = False
    first = True
    actor_loss = 0
    critic_loss = 0
    print(f"Episode: {episode}")
    while not (terminated | truncated):
        t += 1
        rng, key = random.split(rng)
        #epsilon = max(0.1, 1.0 - episode / 100)
        epsilon = 0
        action = policy_fn(actor_params, key, obs, epsilon)

        #action = action.at[0].set(1)
        #action = action.at[1].set(0)
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        if first:
            reward = 0
        first = False
        reward = reward + (obs[0] / 20)
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
            if t % 80 == 0:
                actor_params, actor_opt_state, actor_loss = update_actor(actor_params, critic_params_list, actor_opt_state, batch, rngs[1])

        if done:
            rngs = random.split(rng, 3)
            #batch = sample_batch(rngs[0], memory, len(memory))
            #critic_params, critic_opt_state, critic_loss = update_critic(critic_params, target_critic_params, actor_params, critic_opt_state, batch)
            #actor_params, actor_opt_state, actor_loss = update_actor(actor_params, critic_params, actor_opt_state, batch, rngs[1])
            
            #target_critic_params = update_target_network(critic_params, target_critic_params)
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
            #memory.clear()
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
filenameCritic = "criticParams.pkl"
filenameCriticTarget = "criticTargetParams.pkl"
filenameActor = "actorParams.pkl"