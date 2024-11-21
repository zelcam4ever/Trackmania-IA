# %%
# Load data

import jax.numpy as jnp
import json

def load_recording(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return [(jnp.array(item['obs']), jnp.array(item['act'])) for item in data]

# Load data
recording = load_recording("recording.txt")

# Prepare training data
observations = jnp.stack([obs for obs, act in recording])
actions = jnp.stack([act for obs, act in recording])


# %%
# Define the neural network model

import jax
from jax import random, nn

# Initialize parameters
def initialize_model_params(rng, input_dim, hidden_dim, output_dim):
    w1 = random.normal(rng, (input_dim, hidden_dim)) * 0.01
    b1 = jnp.zeros((hidden_dim,))
    w2 = random.normal(rng, (hidden_dim, hidden_dim)) * 0.01
    b2 = jnp.zeros((hidden_dim,))
    w3 = random.normal(rng, (hidden_dim, output_dim)) * 0.01
    b3 = jnp.zeros((output_dim,))
    return w1, b1, w2, b2, w3, b3

def imitation_model(params, x):
    w1, b1, w2, b2, w3, b3 = params
    z1 = nn.relu(jnp.dot(x, w1) + b1)
    z2 = nn.relu(jnp.dot(z1, w2) + b2)
    z3 = jnp.dot(z2, w3) + b3  # Raw outputs from the model


    gas = jnp.clip(z3[0], 0.0, 1.0)
    brake = jnp.clip(z3[1], 0.0, 1.0)
    steering = jnp.clip(z3[2], -1.0, 1.0)
    z = jnp.stack([gas, brake, steering], axis=-1)
    """x = x.at[idx].set(y)"""

    # Assuming the model output has three components for each action: gas, brake, steering
    # print(f"z3 before at [0]: {z3[..., 0]}")
    # z3 = z3.at[..., 0].set(jnp.clip(z3[..., 0], 0.0, 1.0))  # Clamp gas between 0 and 1
    # print(f"z3 after at [0]: {z3[..., 0]}")
    # z3 = z3.at[..., 1].set(jnp.clip(z3[..., 1], 0.0, 1.0))  # Clamp brake between 0 and 1
    # z3 = z3.at[..., 2].set(jnp.clip(z3[..., 2], -1.0, 1.0)) # Clamp steering between -1 and 1

    return z

# %%
# Define the loss function
def mse_loss(params, model, x, y):
    preds = model(params, x)
    return jnp.mean(jnp.square(preds - y))

# %%
# Define the training step
import optax

# Hyperparameters
learning_rate = 0.001
batch_size = 32
epochs = 100

# Initialize model
rng = random.PRNGKey(0)
input_dim = observations.shape[1]
hidden_dim = 64
output_dim = actions.shape[1]
params = initialize_model_params(rng, input_dim, hidden_dim, output_dim)

# Optimizer
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)

@jax.jit
def train_step(params, opt_state, batch_obs, batch_act):
    loss, grads = jax.value_and_grad(mse_loss)(params, imitation_model, batch_obs, batch_act)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Training
for epoch in range(epochs):
    for i in range(0, len(observations), batch_size):
        batch_obs = observations[i:i+batch_size]
        batch_act = actions[i:i+batch_size]
        params, opt_state, loss = train_step(params, opt_state, batch_obs, batch_act)
    print(f"Epoch {epoch+1}, Loss: {loss}")

#%%
# Test
def test_model(params, observation):
    return imitation_model(params, observation)

# Example usage
test_obs = observations[0]  # Replace with a real test observation
predicted_action = test_model(params, test_obs)
print(f"Predicted Action: {predicted_action}")



# %%
# Test the model
# Setup environment and initialize variables
from tmrl import get_environment

env = get_environment()
env.reset()
rng = random.PRNGKey(43535)

generated_actions = []

for i in range(len(observations)):
    obs = observations[i]
    act = test_model(params, obs)

    act[0] = jnp.clip(act[0], 0.0, 1.0)
    print(f"{i}: {act}")
    generated_actions.append(act)


for i in range(1):
    env.reset()
    terminated, truncated = False, False
    counter = 0
    rng = random.PRNGKey(45 * i)
    while not (terminated | truncated):
        print(f"Next action: {generated_actions[counter % len(generated_actions)]}")
        act = generated_actions[counter % len(generated_actions)]
        
        next_obs, rew, terminated, truncated, info = env.step(act)
        counter += 1