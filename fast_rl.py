import numpy as np
import math
from matplotlib import pyplot as plt
from fast_dynamics import robot_dynamics_func
from functions import statespace
import tensorflow as tf
from collections import deque
import random
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Concatenate



def rk1_step(dynamics_func, state, action, dt):
    next_state = state + dt * dynamics_func(state, action)
    return next_state

statespace = robot_dynamics_func()
next_state_function = lambda x,u,deltat: rk1_step(statespace,x, u, dt=deltat)

def create_actor(state_size, action_size, action_limit):
    state_input = Input(shape=(state_size,))  # Includes current and desired state
    x = Dense(128, activation='relu')(state_input)
    x = Dense(128, activation='relu')(x)
    action_output = Dense(action_size, activation='tanh')(x)  # Output in range [-1, 1]
    scaled_action_output = action_limit * action_output       # Scale to torque range
    return Model(inputs=state_input, outputs=scaled_action_output)

def create_critic(state_size, action_size):
    state_input = Input(shape=(state_size,))
    action_input = Input(shape=(action_size,))
    concat = Concatenate()([state_input, action_input])
    x = Dense(128, activation='relu')(concat)
    x = Dense(128, activation='relu')(x)
    q_value_output = Dense(1, activation=None)(x)  # Single Q-value output
    return Model(inputs=[state_input, action_input], outputs=q_value_output)

def initialize_actor_critic(state_size, action_size, action_limit):
    actor_model = create_actor(state_size, action_size, action_limit)
    actor_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='mse')

    critic_model = create_critic(state_size, action_size)
    critic_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse')

    target_actor = tf.keras.models.clone_model(actor_model)
    target_actor.set_weights(actor_model.get_weights())

    target_critic = tf.keras.models.clone_model(critic_model)
    target_critic.set_weights(critic_model.get_weights())

    return actor_model, critic_model, target_actor, target_critic

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done, desired_state):
        self.buffer.append((state, action, reward, next_state, done, desired_state))

    def sample(self, batch_size):
        minibatch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, desired_states = zip(*minibatch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones), np.array(desired_states))

    def size(self):
        return len(self.buffer)

def soft_update(target_model, source_model, tau):
    target_weights = target_model.get_weights()
    source_weights = source_model.get_weights()

    updated_weights = [tau * source_w + (1 - tau) * target_w
                       for target_w, source_w in zip(target_weights, source_weights)]

    target_model.set_weights(updated_weights)

def train_actor_critic(replay_buffer, gamma, batch_size=64, tau=0.005):
    if replay_buffer.size() < batch_size:
        print("Not enough samples in replay buffer.")
        return

    # Sample from the replay buffer
    states, actions, rewards, next_states, dones, desired_states = replay_buffer.sample(batch_size)

    # Combine current and desired states
    combined_states = np.concatenate([states, desired_states], axis=1)
    combined_next_states = np.concatenate([next_states, desired_states], axis=1)

    # Convert to tensors
    combined_states = tf.convert_to_tensor(combined_states, dtype=tf.float32)
    combined_next_states = tf.convert_to_tensor(combined_next_states, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.float32)
    rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
    dones = tf.convert_to_tensor(dones, dtype=tf.float32)

    # Update the critic
    with tf.GradientTape() as tape:
        next_actions = target_actor(combined_next_states, training=True)
        target_q_values = target_critic([combined_next_states, next_actions], training=True)
        y = rewards + gamma * (1 - dones) * tf.squeeze(target_q_values)
        q_values = tf.squeeze(critic_model([combined_states, actions], training=True))
        critic_loss = tf.reduce_mean((y - q_values) ** 2)


    critic_grads = tape.gradient(critic_loss, critic_model.trainable_variables)
    critic_model.optimizer.apply_gradients(zip(critic_grads, critic_model.trainable_variables))

    # Update the actor
    with tf.GradientTape() as tape:
        predicted_actions = actor_model(combined_states, training=True)
        actor_loss = -tf.reduce_mean(critic_model([combined_states, predicted_actions], training=True))

    actor_grads = tape.gradient(actor_loss, actor_model.trainable_variables)
    actor_model.optimizer.apply_gradients(zip(actor_grads, actor_model.trainable_variables))

    # Soft update target networks
    soft_update(target_actor, actor_model, tau)
    soft_update(target_critic, critic_model, tau)

    print(f"Critic loss: {critic_loss.numpy()}, Actor loss: {actor_loss.numpy()}")

def compute_reward(next_state, action, desired_state, lambda_torque=0.01):
    # Penalise the distance of the next state from the desired state
    state_error = np.linalg.norm(next_state - desired_state)/10

    # Penalise large torques to encourage energy-efficient control
    torque_penalty = np.linalg.norm(action)/10

    # Combine penalties
    reward = -state_error**2 - lambda_torque * torque_penalty**2

    return reward

def collect_experience(actor_model, replay_buffer, desired_state, num_episodes=100, max_steps=200, step_size=0.01, lambda_torque=0.01):
    for _ in range(num_episodes):
        state = np.random.uniform([-10,-10,0,-1,-1,-1], [10,10,2*np.pi,1,1,1])  # Random initial state
        for _ in range(max_steps):
            combined_state = np.concatenate([state, desired_state])  
            action = actor_model.predict(np.expand_dims(combined_state, axis=0), verbose=0)[0]
            next_state = next_state_function(state, action, step_size)  # Tiny step
            reward = compute_reward(next_state, action, desired_state, lambda_torque = lambda_torque)
            if reward < -1E6 or np.linalg.norm(next_state)>1E6:
                break
            done = np.linalg.norm(next_state - desired_state) < 1e-2
            replay_buffer.add(state, action, reward, next_state, done, desired_state)
            state = next_state
            
            if done:
                break

def train_loop(actor_model, critic_model, target_actor, target_critic, replay_buffer, 
               gamma=0.99, tau=0.005, num_iterations=1000, batch_size=64, 
               num_episodes=100, max_steps=200, step_size=0.01, lambda_torque=0.01, verbose=True):
    for iteration in range(num_iterations):
        desired_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Zero desired state
        collect_experience(actor_model, replay_buffer, desired_state, 
                           num_episodes=num_episodes, max_steps=max_steps, step_size=step_size)
        if verbose:
            pass

        if replay_buffer.size() > batch_size:
            train_actor_critic(replay_buffer, gamma=gamma, batch_size=batch_size)
        
        soft_update(target_actor, actor_model, tau)
        soft_update(target_critic, critic_model, tau)

        if verbose and ((iteration + 1) % 10 == 0 or iteration == num_iterations - 1):
            pass
            print(f"Iteration {iteration + 1}/{num_iterations} completed.")

# Parameters
state_size = 12  # Combined state size (current + desired)
action_size = 3  # Torque size (tau1, tau2)
action_limit = 2.0  # Maximum absolute torque
replay_buffer_capacity = 100000

# Initialise networks
actor_model, critic_model, target_actor, target_critic = initialize_actor_critic(
    state_size=state_size, action_size=action_size, action_limit=action_limit)

# Initialise replay buffer
replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)

# Training parameters
DISCOUNT_FACTOR = 0.99
TAU = 0.005
NUM_ITERATIONS = 500
#BATCH_SIZE = 64
BATCH_SIZE = 100
NUM_EPISODES_PER_ITERATION = 10
MAX_STEPS_PER_EPISODE = 50
STEP_SIZE = 0.01
LAMBDA_TORQUE = 0.01

# Train the networks
train_loop(
    actor_model=actor_model,
    critic_model=critic_model,
    target_actor=target_actor,
    target_critic=target_critic,
    replay_buffer=replay_buffer,
    gamma=DISCOUNT_FACTOR,
    tau=TAU,
    num_iterations=NUM_ITERATIONS,
    batch_size=BATCH_SIZE,
    num_episodes=NUM_EPISODES_PER_ITERATION,
    max_steps=MAX_STEPS_PER_EPISODE,
    step_size=STEP_SIZE,
    lambda_torque=LAMBDA_TORQUE,
    verbose=True
)

# Save models after training
actor_model.save("actor_model.keras", save_format='keras')
critic_model.save("critic_model.keras", save_format='keras')
target_actor.save("target_actor.keras", save_format='keras')
target_critic.save("target_critic.keras", save_format='keras')

print("Models saved successfully in `.keras` format!")

# Load models
actor_model = load_model("actor_model.keras")
critic_model = load_model("critic_model.keras")
target_actor = load_model("target_actor.keras")
target_critic = load_model("target_critic.keras")

print("Models loaded successfully from `.keras` files!")



def plot_control_results(history):
    """
    Plots the evolution of states and torques over time.

    Args:
        history (list): List of (state, torque) pairs recorded over time.
    """
    # Extract states and torques
    time_steps = np.arange(len(history))
    states = np.array([h[0] for h in history])  # Extract states
    forces = np.array([h[1] for h in history])  # Extract torques

    # Plot joint angles and velocities
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time_steps, states[:, 0], label=r'$x$ (m)')
    plt.plot(time_steps, states[:, 1], label=r'$y$ (m)')
    plt.plot(time_steps, states[:, 2], label=r'$\theta$ (rad)')
    plt.plot(time_steps, states[:, 3], label=r'$\dot{x}$ (m/s)')
    plt.plot(time_steps, states[:, 4], label=r'$\dot{y}$ (m/s)')
    plt.plot(time_steps, states[:, 5], label=r'$\dot{\theta}$ (rad/s)')
    plt.xlabel("Time Step")
    plt.ylabel("State Values")
    plt.title("Evolution of Joint States")
    plt.legend()
    plt.grid()

    # Plot torques
    plt.subplot(2, 1, 2)
    plt.plot(time_steps, forces[:, 0], label=r'$F_1$ (N)', linestyle="--")
    plt.plot(time_steps, forces[:, 1], label=r'$F_2$ (N)', linestyle="--")
    plt.plot(time_steps, forces[:, 2], label=r'$F_3$ (N)', linestyle="--")
    plt.xlabel("Time Step")
    plt.ylabel("Force (N)")
    plt.title("Control Forces Applied")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(f'rl_result.png', bbox_inches='tight', transparent=True, dpi=1200)


def compute_control_forces(actor_model, initial_state, dt=0.01, max_steps=500):

    state = np.array(initial_state)
    target_state = np.zeros_like(state)  # Zero state goal
    history = []

    for step in range(max_steps):
        # Concatenate current and desired state for actor input
        actor_input = np.concatenate([state, target_state])
        force = actor_model.predict(actor_input.reshape(1, -1), verbose=0)[0]

        # Apply RK4 integration to get next state
        #print(state, torque, params, dt)
        next_state = rk1_step(statespace, state, force, dt)

        # Store history for analysis
        history.append((state.copy(), force.copy()))

        # Update state
        state = next_state

        # Stop if the system is close enough to zero state
        if np.linalg.norm(state) < 1e-3:
            print(f"Converged to zero state in {step+1} steps.")
            break

    return history

initial_state = (0.5, 0.5, math.radians(45), 1.0, 0.5, 0.1)
history = compute_control_forces(actor_model, initial_state, dt=0.01, max_steps = 50)
plot_control_results(history)