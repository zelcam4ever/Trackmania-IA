# %% Imports
import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
from jax import random, value_and_grad
from collections import deque
import optax
import jax.random as random
from flax.core import frozen_dict
from flax.linen import Dense, relu, Module
from typing import Sequence 
import optax
from tmrl import get_environment


env = get_environment()
gamma = 0.99

class Critic(Module):
    obs_dim: int
    action_dim: int
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, obs, action):
        """
        Compute Q-value for a given state-action pair.
        """
        x = jnp.concatenate([obs, action], axis=-1)  # Concatenate state and action
        for dim in self.hidden_dims:
            x = Dense(dim)(x)
            x = relu(x)
        q_value = Dense(1)(x)  # Output a single Q-value
        return jnp.squeeze(q_value, axis=-1)  # Remove unnecessary dimensions

class MLP(nn.Module):
    hidden_dims: list
    output_dim: int

    def setup(self):
        # Create a list of dense layers
        self.hidden_layers = [nn.Dense(dim) for dim in self.hidden_dims]
        self.output_layer = nn.Dense(self.output_dim)  # Output layer

    def __call__(self, x):
        # Pass through the hidden layers
        for layer in self.hidden_layers:
            x = nn.relu(layer(x))
        # Output layer (mean and log_std)
        x = self.output_layer(x)
        return x
    
class Actor(nn.Module):
    obs_dim: int
    action_dim: int

    @nn.compact
    def __call__(self, obs):
        # Hidden layers
        x = nn.Dense(256)(obs)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)

        # Output layers for mean and log_std
        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)

        # Apply activation to ensure correct action ranges
        acceleration_mean = nn.sigmoid(mean[:, 0])  # [0, 1]
        brake_mean = nn.sigmoid(mean[:, 1])        # [0, 1]
        turning_mean = jnp.tanh(mean[:, 2])        # [-1, 1]

        # Combine the actions into a single tensor
        mean = jnp.stack([acceleration_mean, brake_mean, turning_mean], axis=-1)
        
        # Ensure log_std remains unconstrained for stability
        log_std = jnp.clip(log_std, -20, 2)

        return mean, log_std


def forward_mlp(actor_params, obs):
    """
    Pass observation through actor MLP to get the mean and log_std for action sampling.
    """
    # Define the actor network
    actor_network = MLP(hidden_dims=[256, 256], output_dim=6)  # 3 actions, each with mean and log_std
    mean_log_std = actor_network.apply({'params': actor_params}, obs)


    # Split output into mean and log_std for each action
    mean, log_std = mean_log_std[:, :3], mean_log_std[:, 3:]
    
    return mean, log_std


def compute_log_probs(mean, log_std, actions):
    """
    Compute log probabilities of actions given the mean and log_std from the actor network.
    
    Arguments:
    - mean: The mean of the action distribution (from the actor network).
    - log_std: The log standard deviation of the action distribution (from the actor network).
    - actions: The actions sampled from the actor network.
    
    Returns:
    - log probabilities of the sampled actions under the current policy.
    """
    # Compute the standard deviation from the log_std
    std = jnp.exp(log_std)
    
    # Compute the log probability for each action dimension
    log_probs = -0.5 * jnp.sum(jnp.log(2 * jnp.pi * std ** 2) + ((actions - mean) ** 2) / (std ** 2), axis=-1)
    
    return log_probs

def forward_mlp_critic(critic_params, obs, action):
    """
    Pass observation and action through critic MLP to get Q-value.
    
    Arguments:
    - critic_params: Parameters of the critic network.
    - obs: The observation from the environment.
    - action: The action taken by the actor.
    
    Returns:
    - Q-value prediction for the given state-action pair.
    """
    # Concatenate the observation and action along the feature dimension
    x = jnp.concatenate([obs, action], axis=-1)
    
    # Define the critic network (same as actor, but with a different output dimension)
    critic_network = MLP(hidden_dims=[256, 256], output_dim=1)  # Output is a scalar Q-value
    
    # Apply the critic network
    q_value = critic_network.apply({'params': critic_params['params']}, x)
    
    return q_value

def policy_loss(actor_params, critic_params_list, batch, key, alpha=0.1):
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
    obs, actions, _, _, _ = batch

    # Actor outputs mean and log_std
    mean, log_std = forward_mlp(actor_params, obs)
    
    # Sample actions using reparameterization trick
    std = jnp.exp(log_std)
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


def actor_loss(actor_params, critic_params_1, critic_params_2, batch, key, alpha=0.1):
    """
    Compute the policy loss for the actor, with entropy regularization.
    
    Arguments:
    - actor_params: Parameters of the actor (policy).
    - critic_params_1: Parameters of the first critic.
    - critic_params_2: Parameters of the second critic.
    - batch: Batch of experience (observations, actions, rewards, next_obs, dones).
    - key: Random key for sampling actions.
    - alpha: Entropy regularization coefficient.

    Returns:
    - The actor loss (to minimize).
    """
    obs, actions, rewards, next_obs, dones = batch

    # Compute the mean and log_std from the actor network
    mean, log_std = forward_mlp(actor_params, obs)
    std = jnp.exp(log_std)

    # Sample actions from the policy (using the reparameterization trick)
    sampled_actions = mean + std * jax.random.normal(key, mean.shape)

    # Compute the log probability of the sampled actions
    log_probs = compute_log_probs(mean, log_std, sampled_actions)

    # Compute the Q-values for the sampled actions from both critics
    q_values_1 = forward_mlp_critic(critic_params_1, obs, sampled_actions)
    q_values_2 = forward_mlp_critic(critic_params_2, obs, sampled_actions)

    # Average Q-values from both critics
    q_values = jnp.minimum(q_values_1, q_values_2)

    # Policy loss: maximize Q-values and entropy (entropy regularization is added)
    policy_loss = jnp.mean(alpha * log_probs - q_values)

    return policy_loss

def critic_loss(critic_params, target_critic_params, batch, gamma=0.99):
    """
    Compute the critic loss using Bellman backup.
    
    Arguments:
    - critic_params: Parameters of the critic network.
    - target_critic_params: Parameters of the target critic network.
    - batch: Batch of experience (observations, actions, rewards, next_obs, dones).
    - gamma: Discount factor.

    Returns:
    - The critic loss (to minimize).
    """
    obs, actions, rewards, next_obs, dones = batch

    # Get the Q-values for the current state-action pair from the critic network
    q_values = forward_mlp_critic(critic_params, obs, actions)

    # Get the target Q-values from the target critic network (Bellman backup)
    next_q_values = forward_mlp_critic(target_critic_params, next_obs, actions)
    target_q_values = rewards + (1 - dones) * gamma * next_q_values

    # Critic loss: mean squared error between predicted Q-values and target Q-values
    loss = jnp.mean((q_values - target_q_values) ** 2)

    return loss

from collections import namedtuple
import jax
import jax.numpy as jnp
import numpy as np

# Define a named tuple to store experience
Experience = namedtuple('Experience', ['obs', 'action', 'reward', 'next_obs', 'done'])

def collect_batch_from_env(env, actor_params, key, batch_size=10, epsilon=0.1, memory_size=100):
    """
    Collect a batch of experience from the environment.
    
    Arguments:
    - env: The environment.
    - actor_params: The parameters of the actor (policy) network.
    - key: Random key for sampling.
    - batch_size: Number of experiences to collect.
    - epsilon: Exploration rate for epsilon-greedy policy.
    - memory_size: Size of the memory buffer to store experiences.
    
    Returns:
    - A batch of experiences (observations, actions, rewards, next_obs, dones).
    """
    obs = env.reset()  # Reset environment and get the initial observation
    
    # Check the structure of the observation
    print(f"obs before processing: {obs}")
    
    # If obs is a tuple, unpack it and flatten individual components
    if isinstance(obs, tuple):
        print(f"obs is a tuple with length {len(obs)}")
        
        # Check the contents of each element in the tuple
        for idx, o in enumerate(obs):
            print(f"Element {idx} of obs: {o}, type: {type(o)}")
        
        # Flatten the first element (tuple) of obs if it's an ndarray
        if isinstance(obs[0], (np.ndarray, jnp.ndarray)):
            flattened_obs_0 = jnp.ravel(obs[0])
            print(f"Flattened obs[0]: {flattened_obs_0}")
        else:
            flattened_obs_0 = jnp.array([])  # If it's not ndarray, leave it empty

        # Handle obs[1], which is a dictionary, and flatten its contents if needed
        flattened_obs_1 = jnp.array([])  # Initialize as an empty array
        if isinstance(obs[1], dict) and obs[1]:  # If it's a non-empty dictionary
            print(f"Flattening contents of obs[1] (dictionary): {obs[1]}")
            # Extract items from the dictionary and flatten any arrays inside
            flattened_obs_1 = jnp.concatenate([jnp.ravel(v) for v in obs[1].values() if isinstance(v, (np.ndarray, jnp.ndarray))])
        
        # Combine the flattened parts of obs[0] and obs[1] into one observation
        obs = jnp.concatenate([flattened_obs_0, flattened_obs_1]) if flattened_obs_0.size > 0 or flattened_obs_1.size > 0 else jnp.array([])

    else:
        # If obs is not a tuple, simply flatten it
        obs = jnp.ravel(obs)
    
    # Ensure obs is a JAX array (to avoid mixed types later)
    obs = jnp.array(obs)
    print(f"obs after processing: {obs}")
    
    memory = []  # Memory to store experiences
    
    for _ in range(batch_size):
        # Use epsilon-greedy policy to decide on the action
        action = policy_fn(actor_params, key, obs, epsilon)
        
        # Take action in the environment
        next_obs, reward, done, _ = env.step(action)
        
        # Check the structure of next_obs
        print(f"next_obs before processing: {next_obs}")
        
        # If next_obs is a tuple, extract and flatten its components
        if isinstance(next_obs, tuple):
            next_obs_flattened = [jnp.ravel(o) for o in next_obs if isinstance(o, (jnp.ndarray, np.ndarray))]
            print(f"Flattened next_obs: {next_obs_flattened}")
            next_obs = jnp.concatenate(next_obs_flattened)
        else:
            next_obs = jnp.ravel(next_obs)
        
        # Ensure next_obs is a JAX array
        next_obs = jnp.array(next_obs)
        print(f"next_obs after processing: {next_obs}")
        
        # Store the experience in memory
        memory.append(Experience(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done))
        
        # Set the next observation as the current one for the next step
        obs = next_obs
        
        if done:
            obs = env.reset()  # Reset if episode ends
        
        # If memory exceeds the buffer size, pop the oldest experience
        if len(memory) > memory_size:
            memory.pop(0)
    
    # Sample a batch from memory
    batch = sample_batch_from_memory(memory, batch_size)
    
    return batch




def sample_batch_from_memory(memory, batch_size):
    """
    Sample a batch of experiences from memory.
    
    Arguments:
    - memory: List of experiences (obs, action, reward, next_obs, done).
    - batch_size: Number of experiences to sample.
    
    Returns:
    - A batch of experiences (observations, actions, rewards, next_obs, dones).
    """
    # Randomly sample batch indices
    rng = jax.random.PRNGKey(0)
    indices = jax.random.choice(rng, jnp.arange(len(memory)), shape=(batch_size,), replace=False)
    
    # Sample the batch of experiences
    batch = [memory[i] for i in indices]
    
    # Unzip the batch into separate components
    batch = Experience(*zip(*batch))
    
    # Convert the batch components into JAX arrays
    obs, actions, rewards, next_obs, dones = map(jnp.array, (batch.obs, batch.action, batch.reward, batch.next_obs, batch.done))
    
    # Ensure all arrays have the same shape
    obs = jnp.array(obs)
    next_obs = jnp.array(next_obs)
    
    return obs, actions, rewards, next_obs, dones




def critic_loss_fn(critic_params, obs, actions, rewards, next_obs, dones, target_critic_params):
    """
    Calculate the loss for the critic network.
    
    Arguments:
    - critic_params: Parameters for the critic network.
    - obs: Observations.
    - actions: Actions taken by the agent.
    - rewards: Rewards received from the environment.
    - next_obs: Next observations after taking the actions.
    - dones: Done flags indicating the end of episodes.
    - target_critic_params: Parameters for the target critic network.
    
    Returns:
    - Critic loss.
    """
    # Compute the Q-values for the current critic and target critic
    q_values = forward_mlp_critic(critic_params, obs, actions)
    target_q_values_1 = forward_mlp_critic(target_critic_params, next_obs, actions)
    target_q_values_2 = forward_mlp_critic(target_critic_params, next_obs, actions)
    
    # Bellman backup for both critics
    target_q_values = rewards + (1 - dones) * gamma * jnp.minimum(target_q_values_1, target_q_values_2)
    
    # MSE loss for the critic
    critic_loss = jnp.mean((q_values - target_q_values) ** 2)
    
    return critic_loss


def soft_update(target_params, params, tau):
    """
    Soft update the target network parameters using Polyak averaging.
    
    Arguments:
    - target_params: Parameters of the target network.
    - params: Parameters of the current network.
    - tau: Soft update coefficient.
    
    Returns:
    - Updated target parameters.
    """
    return jax.tree_util.tree_map(
        lambda target, param: target * (1 - tau) + param * tau,
        target_params, params)

def update_networks(actor_params, critic_params_1, critic_params_2,
                    target_critic_params_1, target_critic_params_2,
                    batch, key, actor_optimizer, critic_optimizer_1, critic_optimizer_2, 
                    alpha=0.1, tau=0.005):
    """
    Update the actor and critic networks using the collected batch.
    
    Arguments:
    - actor_params: Parameters for the actor network.
    - critic_params_1: Parameters for the first critic network.
    - critic_params_2: Parameters for the second critic network.
    - target_critic_params_1: Parameters for the target critic network (first critic).
    - target_critic_params_2: Parameters for the target critic network (second critic).
    - batch: Batch of experience (obs, actions, rewards, next_obs, dones).
    - key: JAX PRNG key for sampling.
    - actor_optimizer: Optimizer for the actor network.
    - critic_optimizer_1: Optimizer for the first critic network.
    - critic_optimizer_2: Optimizer for the second critic network.
    - alpha: Entropy regularization coefficient.
    - tau: Soft target update coefficient.
    
    Returns:
    - Updated parameters for actor and critics.
    """
    
    # Unpack the batch
    obs, actions, rewards, next_obs, dones = batch

    # Compute actor and critic losses
    actor_loss, actor_grads = value_and_grad(policy_loss, has_aux=True)(
        actor_params, [critic_params_1, critic_params_2], batch, key, alpha)
    
    critic_loss_1, critic_grads_1 = value_and_grad(critic_loss_fn, has_aux=True)(
        critic_params_1, obs, actions, rewards, next_obs, dones, target_critic_params_1)
    
    critic_loss_2, critic_grads_2 = value_and_grad(critic_loss_fn, has_aux=True)(
        critic_params_2, obs, actions, rewards, next_obs, dones, target_critic_params_2)
    
    # Update the networks
    actor_params = actor_optimizer.update(actor_params, actor_grads)
    critic_params_1 = critic_optimizer_1.update(critic_params_1, critic_grads_1)
    critic_params_2 = critic_optimizer_2.update(critic_params_2, critic_grads_2)
    
    # Soft update of target critics using Polyak averaging
    target_critic_params_1 = soft_update(target_critic_params_1, critic_params_1, tau)
    target_critic_params_2 = soft_update(target_critic_params_2, critic_params_2, tau)
    
    return actor_params, critic_params_1, critic_params_2, target_critic_params_1, target_critic_params_2



def update_target_networks(critic_params, target_critic_params, tau=0.005):
    """
    Update the target network parameters using Polyak averaging.
    
    Arguments:
    - critic_params: Current critic network parameters.
    - target_critic_params: Current target critic network parameters.
    - tau: Update rate for target networks.
    
    Returns:
    - Updated target critic parameters.
    """
    updated_target_params_1 = jax.tree_multimap(
        lambda t, p: (1 - tau) * t + tau * p, target_critic_params, critic_params)
    return updated_target_params_1


# Replay buffer
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return map(np.array, zip(*batch))


class MLP(nn.Module):
        hidden_dims: Sequence[int]
        output_dim: int
        
        def setup(self):
            # Define the layers of the MLP
            self.hidden_layers = [
                Dense(dim) for dim in self.hidden_dims
            ]
            self.output_layer = Dense(self.output_dim)
        
        def __call__(self, x):
            # Apply hidden layers
            for layer in self.hidden_layers:
                x = relu(layer(x))
            # Output layer
            return self.output_layer(x)  
    
def initialize_mlp(input_dim, output_dim, hidden_dims=[256, 256]):
    """
    Initialize an MLP model with the given input/output dimensions and hidden layer sizes.
    Args:
        input_dim: The dimension of the input.
        output_dim: The dimension of the output.
        hidden_dims: A list of integers representing the number of units in each hidden layer.
    """
    # Ensure hidden_dims is a list
    if not isinstance(hidden_dims, (list, tuple)):
        raise ValueError("hidden_dims must be a list or tuple.")
    
    # Define the MLP model (assuming MLP class accepts hidden_dims as a list)
    model = MLP(hidden_dims=hidden_dims, output_dim=output_dim)
    
    # Initialize model parameters
    key = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, input_dim))  # Example input to initialize the model
    
    # Initialize parameters with the dummy input
    params = model.init(key, dummy_input)
    
    # Return only the parameters (not the entire state dict)
    return params['params']


def policy_fn(actor_params, rng, obs, epsilon=0.1):
    """
    Return actions using an epsilon-greedy policy or deterministic (actor network-based) policy.
    """
    rng, key = jax.random.split(rng)

    # With epsilon-greedy policy, choose random action
    if jax.random.uniform(key) < epsilon:
        # Sample random actions within valid ranges:
        # Acceleration and Brake between 0 and 1, Steering between -1 and 1
        action = jnp.array([
            np.random.uniform(0, 1),  # Acceleration
            np.random.uniform(0, 1),  # Brake
            np.random.uniform(-1, 1)  # Steering
        ])
    else:
        # Otherwise, use the actor network to get actions
        mean, log_std = forward_mlp(actor_params, obs)

        # Check for NaN or Inf in mean/log_std
        if jnp.any(jnp.isnan(mean)) or jnp.any(jnp.isnan(log_std)):
            print("NaN detected in mean or log_std")
            return jnp.array([0.0, 0.0, 0.0])  # Return a default action

        # Ensure log_std is not too small or large
        std = jnp.exp(log_std) + 1e-6  # Prevent division by zero

        # Sample actions using reparameterization trick
        sampled_actions = mean + std * jax.random.normal(key, mean.shape)

        # Clip actions to the valid range: Acceleration and Brake [0,1], Steering [-1,1]
        sampled_actions = jnp.clip(sampled_actions, jnp.array([0., 0., -1.]), jnp.array([1., 1., 1.]))

        action = sampled_actions  # The action for this timestep

    return action


def get_action_for_inference(actor_params, obs):
    """
    Get the action from the actor for inference (no sampling, use the mean).
    """
    # Get the mean and log_std from the actor (mean is deterministic for testing)
    mean, _ = forward_mlp(actor_params, obs)  # We only need the mean, not the log_std
    
    # For inference, use the mean directly (no sampling, no noise)
    action = jnp.clip(mean, jnp.array([0., 0., -1.]), jnp.array([1., 1., 1.]))

    return action


# Initialize actor and critic networks
def initialize_networks(obs_dim, action_dim, hidden_dim=256):
    # Make sure hidden_dim is a sequence (tuple or list)
    hidden_dims = [hidden_dim, hidden_dim]  # Default to two hidden layers, each of size `hidden_dim`
    
    # Initialize actor and critic networks
    actor_params = initialize_mlp(obs_dim, action_dim * 2, hidden_dims)  # Actor with 2 output dimensions per action (mean, log_std)
    critic_params_1 = initialize_mlp(obs_dim + action_dim, 1, hidden_dims)  # Critic 1
    critic_params_2 = initialize_mlp(obs_dim + action_dim, 1, hidden_dims)  
    target_critic_params_1 = critic_params_1  # Initialize the target critics as the same as the original critics
    target_critic_params_2 = critic_params_2
    return actor_params, critic_params_1, critic_params_2, target_critic_params_1, target_critic_params_2


# Optimizers
def initialize_optimizers(params, lr):
    return optax.adam(lr).init(params)

# Entropy temperature parameter
log_alpha = jnp.array(0.0)  # Log of alpha
alpha_optimizer = optax.adam(1e-3).init(log_alpha)

# Replay buffer
buffer = ReplayBuffer(max_size=100000)

# Initialize PRNG key
rng = random.PRNGKey(0)

# Environment dimensions
obs_dim = 83 # Replace with your environment's observation space size
action_dim = 3   # Replace with your environment's action space size

# SAC networks and optimizers
actor_params, critic_params_1, critic_params_2, target_critic_params_1, target_critic_params_2 = initialize_networks(obs_dim, action_dim)
actor_optimizer = initialize_optimizers(actor_params, lr=3e-4)
critic_optimizer_1 = initialize_optimizers(critic_params_1, lr=3e-4)
critic_optimizer_2 = initialize_optimizers(critic_params_2, lr=3e-4)

# %% Initiliazice the actor
# Initialize the actor
obs_dim = 83
action_dim = 3
key = random.PRNGKey(42)

# Create the Actor module
actor = Actor(obs_dim=obs_dim, action_dim=action_dim)

# Create dummy observations
dummy_obs = jnp.ones((10, obs_dim))  # Batch of 10 observations

# Initialize the actor parameters
init_variables = actor.init(key, dummy_obs)

# Get the mean and log_std for the dummy observations
mean, log_std = actor.apply(init_variables, dummy_obs)

# Test output
print("Mean:", mean)
print("Log Std:", log_std)
print("Mean shape:", mean.shape)  # Should be (10, action_dim)
print("Log Std shape:", log_std.shape)  # Should be (10, action_dim)

# Initialize the critic
critic = Critic(obs_dim=obs_dim, action_dim=action_dim)

# Create dummy observations and actions
dummy_obs = jnp.ones((10, obs_dim))  # Batch of 10 observations
dummy_action = jnp.array([[0.5, 0.0, 0.0]] * 10)  # Batch of 10 actions

# Initialize the critic parameters
critic_params = critic.init(key, dummy_obs, dummy_action)

# Get Q-values for dummy data
q_values = critic.apply(critic_params, dummy_obs, dummy_action)

# Test output
print("Q-values:", q_values)
print("Q-values shape:", q_values.shape)  # Should be (10,)

# %%
num_episodes = 10

# Initialize PRNG key for random number generation
key = jax.random.PRNGKey(0)

# Example training loop for multiple episodes
for episode in range(num_episodes):
    # Collect experience in the environment (obs, actions, rewards, next_obs, dones)
    
    batch = collect_batch_from_env(env, actor_params, key)

    
    # Update networks using the collected batch
    actor_params, critic_params_1, critic_params_2, target_critic_params_1, target_critic_params_2 = update_networks(
        actor_params, critic_params_1, critic_params_2, target_critic_params_1, target_critic_params_2, 
        batch, key, actor_optimizer, critic_optimizer_1, critic_optimizer_2)


# %%
