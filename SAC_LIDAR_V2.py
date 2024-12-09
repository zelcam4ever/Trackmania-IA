# %%
import time
import jax
import jax.numpy as jnp
import optax
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from tmrl import get_environment
import gymnasium as gym
from gymnasium import spaces
# %%
class CustomEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
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
        return self.flatten_observation(obs), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        # Flatten the observation immediately after taking a step
        return self.flatten_observation(obs), reward, done, truncated, info
    
    def flatten_observation(self, obs):
        speed = obs[0].flatten() / 1000.0       # Flatten speed array
        lidar = obs[1].flatten() / 1000.0          # Flatten LIDAR array
        prev_actions = obs[2].flatten()    # Flatten previous action arrays
        prev_actions_2 = obs[3].flatten()  # Flatten second previous action array

        # Concatenate all flattened arrays into a single 1D array
        processed_obs = jnp.concatenate([speed, lidar, prev_actions, prev_actions_2])
        return processed_obs

#%%

# Initialize environment (Make sure the environment is gym-compatible)
env = get_environment()

env_wrapped = CustomEnvWrapper(env)
# Hyperparameters (tuned closer to SB3 defaults)
GAMMA = 0.99
TAU = 0.005
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
LR_ENTROPY = 1e-3
BATCH_SIZE = 256
MAX_EPISODES = 50000
REPLAY_BUFFER_SIZE = 1000000
POLYAK = 0.995
ENTROPY_ALPHA = 0.2
TARGET_UPDATE_INTERVAL = 2

# Action and observation dimensions
obs_dim = env_wrapped.observation_space.shape[0]
action_dim = env_wrapped.action_space.shape[0]

# Define Policy (Actor) Network
def create_actor_network():
    def model(obs):
        x = jax.nn.relu(jax.numpy.dot(obs, np.random.randn(obs.shape[1], 256)))
        x = jax.nn.relu(jax.numpy.dot(x, np.random.randn(256, 256)))
        mu = jax.numpy.dot(x, np.random.randn(256, action_dim))  # mean
        log_std = jax.numpy.dot(x, np.random.randn(256, action_dim))  # log std
        return mu, log_std
    return model

# Define Q-function Network
def create_critic_network():
    def model(obs, action):
        x = jax.numpy.concatenate([obs, action], axis=-1)
        x = jax.nn.relu(jax.numpy.dot(x, np.random.randn(x.shape[1], 256)))
        x = jax.nn.relu(jax.numpy.dot(x, np.random.randn(256, 256)))
        q_value = jax.numpy.dot(x, np.random.randn(256, 1))
        return q_value
    return model

# Replay Buffer
class ReplayBuffer:
    def __init__(self, size, obs_dim, action_dim):
        self.buffer = deque(maxlen=size)
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def store(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in idxs]
        obs, action, reward, next_obs, done = zip(*batch)
        return np.array(obs), np.array(action), np.array(reward), np.array(next_obs), np.array(done)

    def size(self):
        return len(self.buffer)

# SAC Loss function for Q-value
def sac_loss_fn(actor_params, critic_params, target_critic_params, obs, action, reward, next_obs, done):
    mu, log_std = actor_params
    q1 = critic_params[0](obs, action)
    q2 = critic_params[1](obs, action)
    
    # Compute target Q-values (using target Q network and Bellman backup)
    with jax.no_grad():
        next_mu, next_log_std = actor_params
        target_q1 = target_critic_params[0](next_obs, next_mu)
        target_q2 = target_critic_params[1](next_obs, next_mu)
        target_q = reward + (1.0 - done) * GAMMA * jnp.minimum(target_q1, target_q2)
    
    # Loss for Q-functions (Mean Squared Error)
    q1_loss = jnp.mean((q1 - target_q) ** 2)
    q2_loss = jnp.mean((q2 - target_q) ** 2)
    
    # Total Q loss
    total_q_loss = q1_loss + q2_loss
    return total_q_loss

# Optimizer (Adam) for Critic, Actor, and Entropy parameters
actor_optimizer = optax.adam(LR_ACTOR)
critic_optimizer = optax.adam(LR_CRITIC)
entropy_optimizer = optax.adam(LR_ENTROPY)

# Training Loop
def train_sac():
    # Initialize networks and replay buffer
    actor_network = create_actor_network()
    critic_network = [create_critic_network(), create_critic_network()]
    target_critic_network = [create_critic_network(), create_critic_network()]
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE, obs_dim, action_dim)
    
    # Initialize optimizers
    actor_opt_state = actor_optimizer.init(actor_network)
    critic_opt_state = [critic_optimizer.init(critic_network[0]), critic_optimizer.init(critic_network[1])]
    entropy_opt_state = entropy_optimizer.init(ENTROPY_ALPHA)
    
    total_steps = 0
    episode_rewards = []
    avg_rewards = []
    
    for episode in range(MAX_EPISODES):
        obs = env.reset()
        episode_reward = 0
        
        while True:
            total_steps += 1
            
            # Select action using the current policy (actor)
            mu, log_std = actor_network(obs)
            action = mu + np.random.randn(action_dim) * np.exp(log_std)
            
            # Interact with the environment
            next_obs, reward, done, _ = env.step(action)
            replay_buffer.store(obs, action, reward, next_obs, done)
            
            # Update the networks
            if replay_buffer.size() > BATCH_SIZE:
                # Sample batch from replay buffer
                obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = replay_buffer.sample(BATCH_SIZE)
                
                # Compute losses and gradients (using jax's grad function)
                q_loss = sac_loss_fn(actor_network, critic_network, target_critic_network, obs_batch, action_batch, reward_batch, next_obs_batch, done_batch)
                critic_grads = jax.grad(q_loss)(critic_network)
                
                # Update networks using the optimizers
                actor_opt_state, actor_network = actor_optimizer.update(actor_grads, actor_opt_state)
                critic_opt_state[0], critic_network[0] = critic_optimizer.update(critic_grads[0], critic_opt_state[0])
                critic_opt_state[1], critic_network[1] = critic_optimizer.update(critic_grads[1], critic_opt_state[1])
                
                # Polyak update for target critic networks
                for target_critic, critic in zip(target_critic_network, critic_network):
                    target_critic = POLYAK * target_critic + (1 - POLYAK) * critic

            episode_reward += reward
            
            # Update target network
            if episode % TARGET_UPDATE_INTERVAL == 0:
                for target_critic, critic in zip(target_critic_network, critic_network):
                    target_critic = POLYAK * target_critic + (1 - POLYAK) * critic
            
            # Check if the episode is done
            if done:
                break
        
        episode_rewards.append(episode_reward)
        avg_rewards.append(np.mean(episode_rewards[-100:]))
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward}, Average Reward: {avg_rewards[-1]}")

        # Plot rewards every 100 episodes
        if episode % 100 == 0:
            plt.plot(avg_rewards)
            plt.title("Training Progress")
            plt.xlabel("Episodes")
            plt.ylabel("Average Reward")
            plt.show()

    return actor_network, critic_network, avg_rewards

# Run the training
actor_network, critic_network, avg_rewards = train_sac()

