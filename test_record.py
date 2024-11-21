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
EPISODES = 1
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
import keyboard

# Key mapping function
def check_keyboard_input():
    """Check for WASD key presses and return the corresponding action adjustment."""
    action_adjustment = jnp.array([0.0, 0.0, 0.0])  # Default: no adjustment
    if keyboard.is_pressed("w"):
        action_adjustment = action_adjustment.at[0].set(1.0)  # Increase gas
    if keyboard.is_pressed("s"):
        action_adjustment = action_adjustment.at[1].set(1.0)  # Apply brake
    if keyboard.is_pressed("a"):
        action_adjustment = action_adjustment.at[2].set(-1.0)  # Steer left
    if keyboard.is_pressed("d"):
        action_adjustment = action_adjustment.at[2].set(1.0)  # Steer right
    return action_adjustment

for i in range(EPISODES):
    obs, info = env.reset()
    obs = jnp.asarray(flatten(obs))
    terminated, truncated = False, False
    counter = 0
    start_obs = obs
    rng = random.PRNGKey(45 * i)

    recording = []
    while not (terminated | truncated):
        keyboard_adjustment = check_keyboard_input()
        act = keyboard_adjustment
        
        next_obs, rew, terminated, truncated, info = env.step(act)
        next_obs = jnp.asarray(flatten(next_obs))
        obs = next_obs

        recording.append((obs, act))


for i in range(len(recording)):
    print(f"Observation: {recording[i][0][0]}")
    print(f"Action: {recording[i][1]}")


# %%
# Export the recording to a file

import json

# Save the recording array to a text file
def save_recording_to_file(recording, filename):
    with open(filename, 'w') as file:
        # Convert the recording to a serializable format
        serializable_recording = [
            {'obs': obs.tolist(), 'act': act.tolist()} for obs, act in recording
        ]
        json.dump(serializable_recording, file)

# Example usage
save_recording_to_file(recording, "recording.txt")

# %%
