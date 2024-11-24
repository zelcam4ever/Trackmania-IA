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

# %%
# Define constants

ACTION_AMOUNT = 20
EPISODES = 550
EPSILON = 0.2
param_scaling = 0.02

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


# %%
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


# %% Record run to imitate
import time
from threading import Thread, Event

# Initialize the environment and variables
env = get_environment()
obs, info = env.reset()
obs = jnp.asarray(flatten(obs))
terminated, truncated = False, False
action_list = []

start_time = time.time()

# Event to control the recording thread
stop_recording = Event()

# Function to record actions with timestamps
def record_actions(interval=0.05):
    """Record keyboard actions with timestamps at regular intervals."""
    while not stop_recording.is_set():
        action_done = check_keyboard_input()
        timestamp = time.time() - start_time
        action_list.append((action_done, timestamp))
        time.sleep(interval)

# Start the recording thread
recording_thread = Thread(target=record_actions, args=(0.05,))
recording_thread.start()

# Main environment loop
try:
    while not (terminated | truncated):
        next_obs, rew, terminated, truncated, info = env.step(jnp.array([0.0, 0.0, 0.0]))
        next_obs = jnp.asarray(flatten(next_obs))

        # Check for termination
        if terminated or truncated:
            print("Recording complete!")
            break
finally:
    # Stop the recording thread and wait for it to finish
    stop_recording.set()
    recording_thread.join()

# Display recorded actions and timestamps
for action, timestamp in action_list:
    print(f"Action: {action}, Timestamp: {timestamp}")


# %%
# Find the closest action to a given timestamp
import bisect

# Function to find the closest timestamp
def find_closest_action(timestamp, action_list):
    """
    Find the closest action to a given timestamp.
    Uses binary search for efficient lookup.
    """
    timestamps = [entry[1] for entry in action_list]  # Extract timestamps
    idx = bisect.bisect_left(timestamps, timestamp)  # Find insertion position

    # Edge cases: check boundaries
    if idx == 0:
        return action_list[0]  # Closest is the first element
    elif idx == len(timestamps):
        return action_list[-1]  # Closest is the last element

    # Check neighbors to find the closest timestamp
    before = action_list[idx - 1]
    after = action_list[idx]
    if abs(before[1] - timestamp) <= abs(after[1] - timestamp):
        return before
    else:
        return after


# %%
# Main repeating loop
obs, info = env.reset()
terminated, truncated = False, False

# Reset the clock
start_time = time.time()

# Main loop
while not (terminated | truncated):
    # Calculate elapsed time since reset
    elapsed_time = time.time() - start_time

    # Find the closest action for the current elapsed time
    closest_action = find_closest_action(elapsed_time, action_list)
    act = closest_action[0]  # Extract the action part

    next_obs, rew, terminated, truncated, info = env.step(act)
    next_obs = jnp.asarray(flatten(next_obs))
    
    print(f"Time: {elapsed_time:.2f}s, Action: {act}")

    if terminated or truncated:
        print("End of simulation!")
        break