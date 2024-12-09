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
import torch
from torch import nn
from torchvision.models import mobilenet_v3_small
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights

# %%

class CustomEnvWrapper(gym.Wrapper):
    def __init__(self, env, cnn_model):
        super().__init__(env)
        self.cnn_model = cnn_model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn_model.to(self.device)   
        
        # Step 1: Adjust observation space to be flat
        self.observation_space = self._flatten_observation_space(env.observation_space)
        
        # Step 2: Adjust the action space if needed
        self.action_space = self._adjust_action_space(env.action_space)

    def _flatten_observation_space(self, origional_space):
        dummy_input = torch.randn(1, 3, 64, 64).to(self.device)
        with torch.no_grad():
            dummy_output = self.cnn_model(dummy_input)
        
        # Combine CNN output with 9 non-image features
        cnn_output_shape = dummy_output.shape[1:]  # Get the CNN output shape
        flattened_cnn_size = int(np.prod(cnn_output_shape))  # Flatten size
        combined_size = flattened_cnn_size + 9  # Add 9 non-image features
        
        return gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(combined_size,),
            dtype=np.float32
        )
        

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
        obs_data, obs_images = self.preprocess_obs(obs)
        obs_tensor = torch.tensor(obs_images, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.cnn_model(obs_tensor)
        features = features.cpu().numpy().squeeze()
        obs = jnp.concatenate([obs_data, features], axis=0).astype(np.float32)
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        # Flatten the observation immediately after taking a step
        obs_data, obs_images = self.preprocess_obs(obs)
        obs_tensor = torch.tensor(obs_images, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.cnn_model(obs_tensor)
        features = features.cpu().numpy().squeeze()
        obs = jnp.concatenate([obs_data, features], axis=0).astype(np.float32)
        return obs, reward, done, truncated, info
    
    def preprocess_obs(self, obs):
        """
        Preprocessor for TM2020 with images
        """
        grayscale_images = obs[3]
        grayscale_images = grayscale_images.astype(np.float32) / 255.0
        obs_data = jnp.concatenate(
            [
                obs[0].flatten() / 1000.0, 
                obs[1].flatten() / 10.0, 
                obs[2].flatten() / 10000.0, 
                obs[4].flatten(), 
                obs[5].flatten()
            ], 
            axis=0
        )
        return obs_data, grayscale_images



# %%

cnn_model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
cnn_head = nn.Sequential(*list(cnn_model.features), nn.AdaptiveAvgPool2d((1, 1)))

env = get_environment()
wrapped_env = CustomEnvWrapper(env, cnn_head)

# %% Initialize model

model = SAC("MlpPolicy", wrapped_env, verbose=1) 

#%%
model.learn(total_timesteps=30_000, progress_bar=True)

# %%
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

model.save("sac_trackmania_cnn")

# %% Load model

loaded_model = SAC.load("sac_trackmania_cnn")
loaded_model.set_env(wrapped_env)

model = loaded_model
# %%
