# %%
import gymnasium as gym
from gymnasium import spaces
from tmrl import get_environment
import jax.numpy as jnp
from sbx import SAC
from gym import spaces
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
import numpy as np
from gymnasium import spaces

# %%

class CustomEnvWrapper(gym.Wrapper):
    def __init__(self, env, cnn_environment = None):
        super().__init__(env)
        self.cnn_environment = cnn_environment
        
        # Step 1: Adjust observation space to be flat
        self.observation_space = self._flatten_observation_space(env.observation_space)
        
        # Step 2: Adjust the action space if needed
        self.action_space = self._adjust_action_space(env.action_space)

    def _flatten_observation_space(self, space):
        if isinstance(space, spaces.Tuple):
            # Calculate the flattened size
            flat_dim = sum(np.prod(s.shape) for s in space.spaces)
            return spaces.Box(low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32)
        else:
            return space  # No modification if the observation space is not a Tuple

    def _adjust_action_space(self, space):
        if isinstance(space, spaces.Box):
            # Ensure the action space is within the expected bounds
            # Check if the action space is in range [-1, 1]
            if np.all(space.low == -1.0) and np.all(space.high == 1.0):
                return spaces.Box(low=-1.0, high=1.0, shape=space.shape, dtype=np.float32)
            else:
                # Adjust bounds as necessary (for example, scale actions within [-1, 1])
                return space  # You can modify it here if needed
        return space  # No modification if it's not a Box

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Flatten the observation immediately after resetting
        if self.cnn_environment is not None:
            self.cnn_environment
        return self.flatten_observation(obs), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        # Flatten the observation immediately after taking a step
        if self.cnn_environment is not None:
            self.cnn_environment
        return self.flatten_observation(obs), reward, done, truncated, info
    
    def flatten_observation(self, obs):
        speed = obs[0].flatten() /1000      # Flatten speed array
        lidar = obs[1].flatten() /1000         # Flatten LIDAR array
        prev_actions = obs[2].flatten()    # Flatten previous action arrays
        prev_actions_2 = obs[3].flatten()  # Flatten second previous action array

        # Concatenate all flattened arrays into a single 1D array
        processed_obs = jnp.concatenate([speed, lidar, prev_actions, prev_actions_2])
        return processed_obs



# %%
env = get_environment()
wrapped_env = CustomEnvWrapper(env)

# %%

model = SAC("MlpPolicy", wrapped_env, verbose=1) 

#%%
model.learn(total_timesteps=30_000, progress_bar=True)

# %% Test the model using deterministic policy
vec_env = model.get_env()

for episode in range(1):  # rtgym ensures this runs at 20Hz by default
    obs = vec_env.reset()
    total_reward = 0
    done = False

    while not (done):
        vec_env.render()
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        
# %% Save model

model.save("sac_trackmania_lidar")

# %% Load model

loaded_model = SAC.load("sac_trackmania_lidar")
loaded_model.set_env(wrapped_env)

model = loaded_model