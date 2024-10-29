# %% TEST TMRL

from tmrl import get_environment
from time import sleep
from jax import random, grad, nn
import jax.numpy as jnp
from collections import namedtuple
import numpy as np

# %% Setup params

def flatten(args):
    try:
        iter(args)
        final = []
        for arg in args:
            final += flatten(arg)
        return tuple(final)
    except TypeError:
        return (args, )

Params = namedtuple('Params', ['w1', 'b1', 'w2', 'b2', 'w3', 'b3'])

def setup_params(rng):
    
    obs, info = env.reset()
    obs = jnp.asarray(flatten(obs))
    params = Params(
        w1 = random.normal(rng, (jnp.size(obs), 16)),
        b1 = random.normal(rng, (16,)),
        w2 = random.normal(rng, (16, 16)),
        b2 = random.normal(rng, (16,)),
        w3 = random.normal(rng, (16, 3)),
        b3 = random.normal(rng, (3,))
    )
    return params

    
# LIDAR observations are of shape: ((1,), (4, 19), (3,), (3,))
# representing: (speed, 4 last LIDARs, 2 previous actions)
# actions are [gas, break, steer], analog between -1.0 and +1.0
def model(params, x_data):
    z = x_data @ params.w1 + params.b1
    z = nn.relu(z)
    z = z @ params.w2 + params.b2
    z = nn.relu(z)
    z = z @ params.w3 + params.b3
    return z

def policy(params, obs):
    action = model(params, obs)
    return action

def loss_fn(params, obs, action, reward, next_obs, done, gamma):
        q_values = model(params, obs)
        future_max_q_value = model(params, next_obs)
        target_q_value = reward + gamma * future_max_q_value * (1.0 - done)
        loss = jnp.mean((q_values - target_q_value) ** 2)
        return loss

grad_fn = grad(loss_fn)

def update_params(params, obs, action, reward, next_obs, done, learning_rate):
    gamma = 0.9
    # Compute gradients with respect to the parameters
    grads = grad_fn(params, obs, action, reward, next_obs, done, gamma)
    # Update parameters using gradient descent
    params = [param - learning_rate * grad for param, grad in zip(params, grads)]

    return params

# %%
# Let us retrieve the TMRL Gymnasium environment.
# The environment you get from get_environment() depends on the content of config.json
env = get_environment()
rng = random.PRNGKey(0)
params = setup_params(rng)
learning_rate = 0.01

sleep(1.0)  # just so we have time to focus the TM20 window after starting the script


for _ in range(10):  # rtgym ensures this runs at 20Hz by default
    obs, info = env.reset()  # reset environment
    obs = jnp.asarray(flatten(obs))
    terminated, truncated = False, False
    while not (terminated | truncated):
        act = policy(params, obs)  # compute action
        next_obs, rew, terminated, truncated, info = env.step(act)  # step (rtgym ensures healthy time-steps)
        next_obs = jnp.asarray(flatten(next_obs))
        params = update_params(params, obs, act, rew, next_obs, (terminated | truncated), learning_rate)
        obs = next_obs

# %%
