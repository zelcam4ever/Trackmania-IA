# %% 
# Import necessary libraries and define constants
from tmrl import get_environment
from time import sleep
from jax import random, grad, nn
import jax.numpy as jnp
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Define constants
ACTION_AMOUNT = 20
EPISODES = 550
EPSILON = 0.2
param_scaling = 0.02

# %% 
# Helper functions: Flatten nested lists/arrays and initialize model parameters
def flatten(args):
    """Flatten nested lists or arrays into a tuple."""
    try:
        iter(args)
        final = []
        for arg in args:
            final += flatten(arg)
        return tuple(final)
    except TypeError:
        return (args, )

def custom_setup_params(rng):
    """Initialize neural network parameters with scaled random values."""
    w1 = random.normal(rng, (16393, 16)) * param_scaling
    b1 = random.normal(rng, (16,)) * param_scaling
    w2 = random.normal(rng, (16, 16)) * param_scaling
    b2 = random.normal(rng, (16,)) * param_scaling
    w3 = random.normal(rng, (16, ACTION_AMOUNT * 3)) * param_scaling
    b3 = random.normal(rng, (ACTION_AMOUNT * 3,)) * param_scaling
    params = w1, b1, w2, b2, w3, b3
    return params

# %%
# Define the neural network model and policies
def custom_model(params, x_data):
    """Custom neural network for predicting actions."""
    w1, b1, w2, b2, w3, b3 = params
    z = jnp.dot(x_data, w1) + b1
    z = nn.relu(z)
    z = jnp.dot(z, w2) + b2
    z = nn.relu(z)
    z = jnp.dot(z, w3) + b3
    z = jnp.reshape(z, (ACTION_AMOUNT, 3))
    gas = jnp.clip(z[:, 0], 0.8, 1.0)
    brake = jnp.zeros_like(gas)
    steering = jnp.clip(z[:, 2], -1.0, 1.0)
    gas = jnp.round(gas / 0.2) * 0.2
    steering = jnp.round(steering / 0.1) * 0.1
    z = jnp.stack([gas, brake, steering], axis=-1)
    return z

def policy(params, obs):
    """Generate actions based on the policy."""
    action = custom_model(params, obs)
    return action

# %%
# Loss function and parameter update function
def loss_fn(params, obs, action, reward, next_obs, done, gamma):
    """Compute the loss for the neural network."""
    obs = obs.reshape(-1, obs.shape[-1])
    next_obs = next_obs.reshape(-1, next_obs.shape[-1])
    q_values = custom_model(params, obs)
    future_max_q_value = custom_model(params, next_obs)
    q_values = jnp.max(q_values[..., 0], axis=-1)
    future_max_q_value = jnp.max(future_max_q_value[..., 0], axis=-1)
    target_q_value = reward + gamma * future_max_q_value * (1.0 - done)
    loss = jnp.mean((q_values - target_q_value) ** 2)
    return loss

grad_fn = grad(loss_fn)

def update_params(params, batch, learning_rate):
    """Update parameters using gradient descent."""
    gamma = 0.9
    obs, actions, rewards, next_obs, dones = batch
    grads = grad_fn(params, obs, actions, rewards, next_obs, dones, gamma)
    params = [param - learning_rate * grad for param, grad in zip(params, grads)]
    return params

# %%
# Custom epsilon-greedy policy
def custom_epsilon_policy(params, obs, epsilon, rng):
    """Epsilon-greedy policy for exploration."""
    if random.uniform(rng) < epsilon:
        return custom_model(custom_setup_params(rng), obs)
    actions = custom_model(params, obs)
    return actions

# %%
# Setup environment and initialize variables
env = get_environment()
sleep(1.0)
rng = random.PRNGKey(43535)
params = custom_setup_params(rng)
learning_rate = 0.1
reward_history = []

# %%
# Main training loop
for i in range(EPISODES):
    obs, info = env.reset()
    obs = jnp.asarray(flatten(obs))
    terminated, truncated = False, False
    counter = 0
    start_obs = obs
    rng = random.PRNGKey(45 * i)
    actions = custom_epsilon_policy(params, obs, EPSILON, rng)
    actions = jnp.reshape(actions, (ACTION_AMOUNT, 3))
    total_reward = 0.0
    obs_list = []
    action_list = []

    while not (terminated | truncated):
        act = actions[counter % 20]
        action_list.append(act)
        next_obs, rew, terminated, truncated, info = env.step(act)
        next_obs = jnp.asarray(flatten(next_obs))
        speed_threshold = 20
        if obs[0] > speed_threshold:
            rew = 1
            print(f"#### Speed above! -- {obs[0]} ####")
        else:
            rew = 0
        total_reward += rew
        obs_list.append(obs)
        obs = next_obs
        counter += 1
        if counter >= ACTION_AMOUNT:
            break

    # Update parameters based on rewards
    def reward_loss_fn(params):
        q_values = custom_model(params, start_obs)
        predicted_rewards = jnp.sum(q_values * actions, axis=-1)
        return -jnp.sum(predicted_rewards * total_reward)

    grads = grad(reward_loss_fn)(params)
    params = [param - learning_rate * grad for param, grad in zip(params, grads)]
    reward_history.append(total_reward)

    # Adjust exploration and learning rate
    if total_reward > 6:
        EPSILON = 0.01
        learning_rate = 0.05
    elif total_reward > 10:
        EPSILON = 0.001
        learning_rate = 0.01

    # Periodically display results and plot rewards
    if i % 10 == 0:
        print(f"Current Epsilon: {EPSILON}")
        print("Actions for this episode:")
        for j in range(len(action_list)):
            print(f"Action: {action_list[j]}")
        plt.figure(figsize=(10, 6))
        plt.plot(reward_history, label="Reward per Episode", color="b", marker="o", markersize=4, linewidth=1)
        plt.title("Reward History Over Episodes", fontsize=16)
        plt.xlabel("Episode", fontsize=14)
        plt.ylabel("Total Reward", fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()

    print("################################################")
    print("#")
    print(f"# Total Reward after Episode {i}: {total_reward}")
    print("#")
    print("################################################")

# %%
# Final visualization of reward history
plt.figure(figsize=(10, 6))
plt.plot(reward_history, label="Reward per Episode", color="b", marker="o", markersize=4, linewidth=1)
plt.title("Reward History Over Episodes", fontsize=16)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Total Reward", fontsize=14)
plt.grid(alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
