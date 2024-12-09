# %%
import jax
import jax.numpy as jnp
from jax import random, value_and_grad, nn, jit
import optax
import numpy as np
from collections import deque, namedtuple
from tmrl import get_environment
import matplotlib.pyplot as plt
import pickle
import distrax
import gymnasium as gym
from gymnasium import spaces
from functools import partial
# %% Environment wrapper

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
        reward *= 10
        # Flatten the observation immediately after taking a step
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


# %%
filename_q = "SACBestQ.pkl"
filename_policy = "SACBestPolicy.pkl"
filename_q1_target = "SACBestQ1Target.pkl"
filename_q2_target = "SACBestQ2Target.pkl"
filename_entropy = "SACBestEntropy.pkl"
current_filename_q = "SACCurrentQ.pkl"
current_filename_policy = "SACCurrentPolicy.pkl"
current_filename_q1_target = "SACCurrentQ1Target.pkl"
current_filename_q2_target = "SACCurrentQ2Target.pkl"
current_filename_entropy = "SACCurrentEntropy.pkl"
env = get_environment()
env = CustomEnvWrapper(env)

# %%
# Initialize networks
def initialize_mlp_params(rng, input_size, hidden_sizes, output_size):
    def init_layer_params(m, n, rng):
        w_key, b_key = random.split(rng)
        weight = random.normal(w_key, (m, n)) * jnp.sqrt(2.0 / m)
        bias = jnp.zeros(n)
        return weight, bias

    params = []
    keys = random.split(rng, len(hidden_sizes) + 1)
    
    # Input layer
    params.append(init_layer_params(input_size, hidden_sizes[0], keys[4]))
    
    # Hidden layers
    for i in range(len(hidden_sizes) - 1):
        params.append(init_layer_params(hidden_sizes[i], hidden_sizes[i + 1], keys[i + 5]))
    # Output layer
    params.append(init_layer_params(hidden_sizes[-1], output_size, keys[-1]))
    return params

# Forward functions
def forward(params, obs, action):
    if action.ndim == 1:
        action = action[None,...]
    if obs.ndim == 1:
        obs = obs[None,...]
    activations = jnp.concatenate([obs, action], axis=1) 
    for w, b in params[:-1]:
        activations = nn.relu(jnp.dot(activations, w) + b)  # For bounded actions
    final_w, final_b = params[-1]
    return jnp.dot(activations, final_w) + final_b

def policy_forward(params, obs, key, test=False, compute_logprob=True, act_limit=jnp.array([0.5, 0.5, 1])):
    """Forward pass for the policy network."""
    
    activations = obs
    for w, b in params[:-1]:
        activations = nn.relu(jnp.dot(activations, w) + b)  # Hidden layers

    final_w, final_b = params[-1]
    output = jnp.dot(activations, final_w) + final_b

    if output.ndim == 1:
        output = output[None,...]

    mean, log_std = jnp.split(output, 2, axis=-1)

    log_std = jnp.clip(log_std, -20, 2)
    std = jnp.exp(log_std)
    
    dist = distrax.Normal(mean, std)
    
    if test:
        pi_action = mean
    else:
        pi_action = dist.sample(seed=key)
    
    if compute_logprob:
        logp_pi = dist.log_prob(pi_action).sum(axis=-1)
    else:
        logp_pi = None
    
    # Apply the tanh transformation and scale the action
    pi_action = jnp.tanh(pi_action)  # Apply tanh to squish to [-1, 1]
    
    # Correct for the tanh transformation in the log probability
    if compute_logprob:
        logp_pi -= jnp.log(1 - pi_action ** 2).sum(axis=-1)  # Tanh correction

    pi_action = pi_action.squeeze()  # Remove unnecessary dimensions
    
    return pi_action, logp_pi




# Loss functions 

def critic_loss_fn(Q_params, policy_params ,Q1_target_params, Q2_target_params, batch, key, alpha, gamma = 0.99):
    """Q-function loss."""
    obs, action, reward, next_obs, done = batch
    q1_value = forward(Q_params[0], obs, action)
    q2_value = forward(Q_params[1], obs, action)
    
    next_action, next_logp = policy_forward(policy_params, obs, key)
    
    q1_pi_target = forward(Q1_target_params, next_obs, next_action)
    q2_pi_target = forward(Q2_target_params, next_obs, next_action)
    
    q_pi_target = jnp.minimum(q1_pi_target, q2_pi_target)
    
    q_pi_target = jnp.maximum(q_pi_target, 0.0)
    
    backup = reward + gamma * (1 - done) * (q_pi_target - alpha * next_logp)
    
    loss_q1 = jnp.mean((q1_value - backup)**2)
    loss_q2 = jnp.mean((q2_value - backup)**2)
    
    loss = loss_q1 + loss_q2
    
    return loss

def policy_loss_fn(policy_params, Q1_params, Q2_params, batch, rng, alpha):
    rng, key = random.split(rng)
    obs, _, _, _, _ = batch
    
    pi, logp_pi = policy_forward(policy_params, obs, key)
    
    q1_pi = forward(Q1_params, obs, pi)
    q2_pi = forward(Q2_params, obs, pi)
    q_pi = jnp.minimum(q1_pi, q2_pi)
    
    
    loss_pi = jnp.mean(alpha * logp_pi - q_pi)
    
    return loss_pi

def entropy_loss_fn(log_alpha, policy_params, target_entropy, batch, key):
    obs, _, _, _, _ = batch
    _, logp_pi = policy_forward(policy_params, obs, key)
    alpha = jnp.exp(log_alpha)
    loss_alpha = -alpha * jnp.mean(logp_pi + target_entropy)
    return loss_alpha



# Training step with gradient descent
@jit
def update_actor(params, Q1_params, Q2_params, opt_state, batch, key, alpha):
    policy_loss = partial(policy_loss_fn, Q1_params=Q1_params, Q2_params=Q2_params, 
                          batch=batch, rng=key, alpha=alpha)
    loss, gradients = jax.value_and_grad(policy_loss)(params)
    updates, opt_state = opt_actor.update(gradients, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


@jit
def update_critic(Q_params, policy_params, Q1_target_params, Q2_target_params, opt_state, batch, key, alpha):
    
    # Define the loss function
    q_loss = partial(critic_loss_fn, policy_params=policy_params, Q1_target_params=Q1_target_params, 
                     Q2_target_params=Q2_target_params, batch=batch, key=key, alpha=alpha)
    
    # Compute gradients
    loss, gradients = jax.value_and_grad(q_loss)(Q_params)
    
    # Prepare updates
    updates_q, opt_state = opt_critic.update(gradients, opt_state, Q_params)
    
    
    # Apply updates to critic parameters
    Q_params = optax.apply_updates(Q_params, updates_q)
    
    
    
    # Return updated parameters, optimizer state, and loss
    return Q_params, opt_state, loss

@jit
def update_entropy(log_alpha, policy_params, alpha_opt_state, target_entropy, batch, key):
    entropy_loss = partial(entropy_loss_fn, policy_params=policy_params, 
                           target_entropy=target_entropy, batch=batch, key=key)
    
    loss, gradients = jax.value_and_grad(entropy_loss)(log_alpha)
    updates, opt_state = opt_alpha.update(gradients, alpha_opt_state, log_alpha)
    log_alpha = optax.apply_updates(log_alpha, updates)
    
    return log_alpha, opt_state, loss

# Soft update for target networks
def soft_update(target_params, source_params, polyak=0.995):
    return jax.tree_util.tree_map(lambda t, s: polyak * t + (1 - polyak) * s, target_params, source_params)


def convert_to_game_action(action, action_limit):
    action = action * action_limit
    action = action.at[:2].add(0.5)
    return action

# %% set variables and rngs
seed=0
episodes=301
batch_size=256
gamma=0.99
polyak=0.995
alpha = 0.2
lr_actor=3e-4
lr_critic=3e-4
lr_entropy=1e-5
target_entropy = -3
replay_buffer_size = 100000

rng = random.PRNGKey(seed)
obs_dim = 83
hidden_dim = [256, 256]
action_dim = 3
rngs = random.split(rng, 3)

# %% init params

# Initialize networks
Q1_params = initialize_mlp_params(rngs[0], obs_dim + action_dim, hidden_dim, 1)
Q2_params = initialize_mlp_params(rngs[1], obs_dim + action_dim, hidden_dim, 1)
Q1_target_params = Q1_params.copy()
Q2_target_params = Q2_params.copy()
policy_params = initialize_mlp_params(rngs[2], obs_dim, hidden_dim, action_dim * 2)
#log_alpha = -1.6

# Optimizers
# Define a gradient clipping optimizer
def create_clipped_optimizer(learning_rate, max_norm):
    """Create an optimizer with gradient clipping."""
    return optax.chain(
        optax.clip_by_global_norm(max_norm),  # Gradient clipping
        optax.adam(learning_rate)  # Adam optimizer
    )

# Optimizers
opt_actor = create_clipped_optimizer(lr_actor, max_norm=1.0)
opt_critic = create_clipped_optimizer(lr_critic, max_norm=1.0)
#opt_alpha = create_clipped_optimizer(lr_entropy, max_norm=1.0)
Q_params = (Q1_params, Q2_params)
Q_opt_state = opt_critic.init(Q_params)
policy_opt_state = opt_actor.init(policy_params)
#alpha_opt_state = opt_alpha.init(log_alpha)


replay_buffer = ReplayBuffer(replay_buffer_size, obs_dim, action_dim)
total_steps = 30
episode_rewards = []
avg_rewards = []
highestReward = 0
loss_policy = 0
loss_critic = 0
loss_entropy = 0

# %%

# Training loop
for episode in range(episodes):
    obs, _ = env.reset()
    episode_reward = 0
    done = False
    #action = jnp.zeros(3)
    rng, subkey = random.split(rng)
    
    if episode % 10 == 0 and episode != 0:
        with open(filename_q, 'wb') as f:
            pickle.dump(best_q_params, f)
        with open(filename_policy, 'wb') as f:
            pickle.dump(best_policy_params, f)
        with open(filename_q1_target, 'wb') as f:
            pickle.dump(best_q1_target_params, f)
        with open(filename_q2_target, 'wb') as f:
            pickle.dump(best_q2_target_params, f)
        #with open(filename_entropy, 'wb') as f:
        #    pickle.dump(best_log_alpha, f)
        with open(current_filename_q, 'wb') as f:
            pickle.dump(Q_params, f)
        with open(current_filename_policy, 'wb') as f:
            pickle.dump(policy_params, f)
        with open(current_filename_q1_target, 'wb') as f:
            pickle.dump(Q1_target_params, f)
        with open(current_filename_q2_target, 'wb') as f:
            pickle.dump(Q2_target_params, f)
        #with open(current_filename_entropy, 'wb') as f:
        #    pickle.dump(log_alpha, f)

    while not done:
        total_steps += 1
        rng, subkey = random.split(rng)
        #prev_actio = action
        action, _ = policy_forward(policy_params, obs, subkey, test=False, compute_logprob=False)
        #if total_steps % 4 != 0:
        #    action = action.at[2].set(prev_actio[2])
        game_action = convert_to_game_action(action, jnp.array([0.5, 0.5, 1.0]))
        next_obs, reward, terminated, truncated, info = env.step(game_action)
        done = terminated or truncated

        replay_buffer.store(obs, action, reward, next_obs, done)
        obs = next_obs
        episode_reward += reward
        
        if replay_buffer.size() > batch_size:
            rngs = random.split(rng, 7)
            batch = replay_buffer.sample(batch_size)
            
            Q_params, Q_opt_state, loss_q = update_critic(Q_params, policy_params, Q1_target_params, Q2_target_params, Q_opt_state, batch, rngs[3], alpha)
            
            policy_params, policy_opt_state, loss_policy = update_actor(policy_params, Q1_params, Q2_params, policy_opt_state, batch, rngs[5], alpha)

            #log_alpha, alpha_opt_state, loss_entropy = update_entropy(log_alpha, policy_params, alpha_opt_state, target_entropy, batch, rngs[2])
            
            Q1_target_params = soft_update(Q1_target_params, Q1_params)
            Q2_target_params = soft_update(Q2_target_params, Q2_params)
            
        
        if done:
            if(highestReward < episode_reward):
                best_q_params = Q_params
                best_policy_params = policy_params
                best_q1_target_params = Q1_target_params
                best_q2_target_params = Q2_target_params
                #best_log_alpha = log_alpha
                highestReward = episode_reward
                print("New record:" , highestReward)
                
                
    episode_rewards.append(episode_reward)
    avg_rewards.append(np.mean(episode_rewards[-100:]))
    
    if episode % 10 == 0 and episode != 0:
        print(f"--- Episode {episode} Summary ---")
        print(f"Reward:             {episode_reward:.2f}")
        print(f"Average Reward:     {np.mean(avg_rewards[-100:]):.2f}")
        print(f"Total Timesteps:    {total_steps}")
        print(f"Episode Length (mean): {total_steps / episode:.2f}")
        print(f"Loss (Critic):      {loss_q:.6f}")
        print(f"Loss (Policy):      {loss_policy:.6f}")
        #print(f"Entropy Coefficient: {jnp.exp(log_alpha):.4f}")
        print("-" * 30)
    
    if episode % 50 == 0 and episode != 0:
            plt.plot(avg_rewards)
            plt.title("Training Progress")
            plt.xlabel("Episodes")
            plt.ylabel("Average Reward")
            plt.show()


# %%

with open(filename_q, 'rb') as f:
    loaded_q_params = pickle.load(f)
with open(filename_policy, 'rb') as f:
    loaded_policy_params = pickle.load(f)
with open(filename_q1_target, 'rb') as f:
    loaded_q1_target_params = pickle.load(f)
with open(filename_q2_target, 'rb') as f:
    loaded_q2_target_params = pickle.load(f)
with open(filename_entropy, 'rb') as f:
    loaded_entropy_params = pickle.load(f)
    
# %%
Q_params = loaded_q_params
policy_params = loaded_policy_params
Q1_target_params = loaded_q1_target_params
Q2_target_params = loaded_q2_target_params
Q1_params, Q2_params = Q_params
log_alpha = loaded_entropy_params
critic1_opt_state = opt_critic.init(Q1_params)
critic2_opt_state = opt_critic.init(Q2_params)
policy_opt_state = opt_actor.init(policy_params)
alpha_opt_state = opt_alpha.init(log_alpha)

# %%
with open(current_filename_q, 'rb') as f:
    loaded_q_params = pickle.load(f)
with open(current_filename_policy, 'rb') as f:
    loaded_policy_params = pickle.load(f)
with open(current_filename_q1_target, 'rb') as f:
    loaded_q1_target_params = pickle.load(f)
with open(current_filename_q2_target, 'rb') as f:
    loaded_q2_target_params = pickle.load(f)
with open(current_filename_entropy, 'rb') as f:
    loaded_entropy_params = pickle.load(f)

# %%