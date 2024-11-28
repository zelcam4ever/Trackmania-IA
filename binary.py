# %% -------------------1. Imports, constant variables and functions -------------------
#Imports
import gymnasium as gym  # not jax based
import jax
from jax import random, grad, jit, nn
import jax.numpy as jnp
from tqdm import tqdm
from collections import deque, namedtuple
import optax
import matplotlib.pyplot as plt
import pickle
from tmrl import get_environment

env = get_environment() 
rng = random.PRNGKey(0)
entry = namedtuple("Memory", ["obs", "action", "reward", "next_obs", "done"])
memory = deque(maxlen=1000000)   # <- replay buffer
gamma = 0.99
filename = "params.pkl"

def initialize_mlp_params(rng, input_size, hidden_sizes, output_size):
    # Helper function to initialize network weights and biases
    def init_layer_params(m, n, rng):
        w_key, b_key = random.split(rng)
        weight = random.normal(w_key, (m, n)) * jnp.sqrt(2.0 / m)  # Initialization
        bias = jnp.zeros(n)
        return weight, bias

    params = []
    keys = random.split(rng, len(hidden_sizes) + 1)  # Separate keys for each layer

    # Input layer to first hidden layer
    params.append(init_layer_params(input_size, hidden_sizes[0], keys[0]))

    # Hidden layers
    for i in range(len(hidden_sizes) - 1):
        params.append(init_layer_params(hidden_sizes[i], hidden_sizes[i + 1], keys[i + 1]))

    # Final layer to output (Q-values)
    params.append(init_layer_params(hidden_sizes[-1], output_size, keys[-1]))

    return params

# Forward pass through the network
def forward_mlp(params, x):
    activations = x
    for w, b in params[:-1]:
        activations = nn.tanh(jnp.dot(activations, w) + b)  # Hidden layers

    final_w, final_b = params[-1]
    return jnp.dot(activations, final_w) + final_b




def bellman_loss(params, target_params, batch):
    obs, actions, rewards, next_obs, dones = batch
    
    # Predicted Q-values for the current state-action pair (Q-values for all actions)
    q_values = forward_mlp(params, obs)
    
    # Convert actions to indices (this assumes actions are one-hot or can be interpreted as integers)
    actions_indices = jnp.array([invert_binary_approach(action) for action in actions])
    
    # One-hot encode the actions (create a binary vector for each action)
    one_hot_actions = jax.nn.one_hot(actions_indices, num_classes=q_values.shape[1])

    # Now, select the Q-values corresponding to the one-hot action indices
    q_values_selected = jnp.sum(q_values * one_hot_actions, axis=1)  # Dot product to get the Q-values for the actions

    # Target Q-values using the target network (for next state)
    next_q_values = forward_mlp(target_params, next_obs)
    max_next_q_values = jnp.max(next_q_values, axis=1)  # max_a' Q(s', a')

    # Bellman target
    target = rewards + gamma * max_next_q_values * (1.0 - dones)

    # Loss (mean squared error between Q-values and target)
    loss = jnp.mean((q_values_selected - target) ** 2)
    
    return loss



def random_policy_fn(rng, obs): # action (shape: ())
    n = 12
    return random.randint(rng, (1,), 0, n).item()

def binary_approach(index):
    # Define a dictionary mapping indices to arrays
    switch_dict = {
        0: jnp.array([0, 0, -1]),
        1: jnp.array([0, 0, 0]),
        2: jnp.array([0, 0, 1]),
        3: jnp.array([1, 0, 0]),
        4: jnp.array([1, 0, 1]),
        5: jnp.array([1, 0, -1]),
        6: jnp.array([0, 1, 0]),
        7: jnp.array([0, 1, 1]),
        8: jnp.array([0, 1, -1]),
        9: jnp.array([1, 1, 0]),
        10: jnp.array([1, 1, 1]),
        11: jnp.array([1, 1, -1]),
    }

    # Use .get() to provide a default value if the index is out of range
    return switch_dict.get(index, jnp.array([0, 0, 0]))  # Default to (0, 0, 0)

def invert_binary_approach(action_array):
    # Define the reverse dictionary mapping action arrays to indices
    reverse_dict = {
        (0, 0, -1): 0,
        (0, 0, 0): 1,
        (0, 0, 1): 2,
        (1, 0, 0): 3,
        (1, 0, 1): 4,
        (1, 0, -1): 5,
        (0, 1, 0): 6,
        (0, 1, 1): 7,
        (0, 1, -1): 8,
        (1, 1, 0): 9,
        (1, 1, 1): 10,
        (1, 1, -1): 11
    }
    
    # Convert the action array to a tuple to use as a key in the dictionary
    action_tuple = tuple(action_array)
    
    # Return the index if it exists in the reverse dictionary, else default to None
    return reverse_dict.get(action_tuple, -1)  # Default to -1 if not found




def your_policy_fn(params, rng, obs, epsilon=0.1):
    rng, key = random.split(rng)
    if random.uniform(key) < epsilon:  # Exploration
        return binary_approach(random_policy_fn(rng, obs) ) # Random action
    else:  # Exploitation
        q_values = forward_mlp(params, obs)  # Forward pass through the Q-network
        return binary_approach(jnp.argmax(q_values).item())


# Training update step
@jit
def update(params, target_params, opt_state, batch):
    # Compute gradients of the loss function w.r.t. parameters
    loss, gradients = jax.value_and_grad(bellman_loss)(params, target_params, batch)

    # Update parameters using the optimizer
    updates, opt_state = optimizer.update(gradients, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state, loss


def sample_batch(rng, memory, batch_size):
    memory_size = len(memory)

    # Sample random indices from the memory buffer using jax.random
    indices = random.choice(rng, jnp.arange(memory_size), shape=(batch_size,), replace=False)
    
    # Gather sampled transitions based on the indices
    batch = [memory[i] for i in indices]
    batch = entry(*zip(*batch))  # Unpack namedtuple into individual components

    # Convert batch to JAX arrays
    obs = jnp.array(batch.obs)
    actions = jnp.array(batch.action)
    rewards = jnp.array(batch.reward)
    next_obs = jnp.array(batch.next_obs)
    dones = jnp.array(batch.done)
    
    return obs, actions, rewards, next_obs, dones


def preprocess_obs(obs):
    speed = obs[0].flatten()          # Flatten speed array
    lidar = obs[1].flatten()          # Flatten LIDAR array
    prev_actions = obs[2].flatten()   # Flatten previous action arrays
    prev_actions_2 = obs[3].flatten() # Flatten second previous action array

    # Concatenate all flattened arrays into a single 1D array
    processed_obs = jnp.concatenate([speed, lidar, prev_actions, prev_actions_2])

    return processed_obs





def update_target_network(params, target_params, tau=0.95):
    # Soft update: tau=1.0 makes it a hard update (full copy)
    new_target_params = jax.tree_util.tree_map(
        lambda p, tp: tau * p + (1 - tau) * tp, params, target_params
    )
    return new_target_params


# %% ------------------- 2. Initializer -------------------

rng, init_rng = random.split(rng)
input_size = 83  # Based on Trackmania observations (flattened)
hidden_sizes = [128, 64]  # Hidden layer sizes
output_size = 12

params = initialize_mlp_params(init_rng, input_size, hidden_sizes, output_size)
target_params = params.copy()
# Test a forward pass
obs = jnp.ones(input_size)  # Example observation
q_values = forward_mlp(params, obs)
print(f"Q-values: {q_values}")


learning_rate = 0.01
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params) 

# %% ------------------- 3. Main training loop -------------------------

num_episodes = 100
batch_size = 10
update_frequency = 4  # Update the network every 4 steps
target_update_frequency = 15  # Update the target network every 50 steps
replay_start_size = 20
rewards = []
highestReward = 0
goodAgent = False

with tqdm(total=num_episodes, desc="Training Progress") as pbar:
    for episode in range(num_episodes):
        obs, info = env.reset()
        obs = preprocess_obs(obs)
        total_reward = 0
        t=0
        terminated = False
        truncated = False
        while not (terminated | truncated):
            t += 1 # Limit steps per episode
            rng, key = random.split(rng)
            epsilon = min(0.1, 1.0 - episode / num_episodes)
            action = your_policy_fn(params, key, obs, epsilon)  # Epsilon-greedy action selection
            # Take a step in the environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            print(action)
            done = terminated #| truncated
            next_obs = preprocess_obs(next_obs)
            # Store transition in memory
            memory.append(entry(obs, action, reward, next_obs, done))

            # Update observation
            obs = next_obs if not done else env.reset()[0]
            reward = float(reward)
            total_reward += reward 

            # Start training once replay buffer is sufficiently full
            if len(memory) >= replay_start_size and t % update_frequency == 0:
                # Sample a batch from the replay buffer using jax.random
                rng, key = random.split(rng)
                batch = sample_batch(key, memory, batch_size)

                # Perform a training step
                params, opt_state, loss = update(params, target_params, opt_state, batch)
                # print(f"Step {t}, Loss: {loss}")
            # Update the target network periodically
            if t % target_update_frequency == 0:
                target_params = update_target_network(params, target_params)
                    
            
            if (done):
                if(highestReward < total_reward):
                    best_params = params.copy()
                    highestReward = total_reward
                    print("New record:" , highestReward)
                break
        if(highestReward < total_reward):
            best_params = params.copy()
            highestReward = total_reward
            print("New record:" , highestReward)
            break

        rewards.append(total_reward)
        pbar.update(1)
        pbar.set_postfix({"Episode Reward": total_reward})
        # env.render()
        # print(f"Episode {episode}, Total Reward: {total_reward}")


env.close()

# %% ------------------- 4. Graph -------------------
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Progress')
plt.show()

# %% ------------------- 5. Params Saver -------------------
with open(filename, 'wb') as f:
    pickle.dump(best_params, f)

# %% ------------------- 6. Params Loader -------------------
with open(filename, 'rb') as f:
    loaded_params = pickle.load(f)

# %% ------------------- 7. Agent Playing (loaded params) -------------------
env = gym.make('CartPole-v1',  render_mode="human")

obs, info = env.reset()

for t in range(20000):  # Limit steps per episode
    rng, key = random.split(rng)
    action = your_policy_fn(loaded_params, rng, obs, -1)

    next_obs, reward, terminated, truncated, info = env.step(int(action))
    obs, info = next_obs, info if not (terminated | truncated) else env.close()


