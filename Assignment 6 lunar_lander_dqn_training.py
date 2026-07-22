import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

# Attempt to import gymnasium, fallback to gym if needed
try:
    import gymnasium as gym
    ENV_V1 = True
except ImportError:
    import gym
    ENV_V1 = False

# 1. Define the Neural Network Architecture for the Deep Q-Network (DQN)
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        # Lunar Lander state space is 8, action space is 4
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 2. Define the Replay Buffer for Experience Replay
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

def main():
    print("1. Initializing Lunar Lander DQN Training Project...")
    
    # Environment Setup
    # Using 'LunarLander-v3' if available, otherwise 'LunarLander-v2'
    env_name = "LunarLander-v3" if ENV_V1 else "LunarLander-v2"
    
    try:
        env = gym.make(env_name)
    except gym.error.Error:
        env_name = "LunarLander-v2"
        env = gym.make(env_name)
        
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"Environment: {env_name}")
    print(f"State space dimensions: {state_dim} (e.g. coordinates, velocity, angle)")
    print(f"Action space dimensions: {action_dim} (0: nothing, 1: left engine, 2: main engine, 3: right engine)")

    # Hyperparameters
    batch_size = 64
    gamma = 0.99          # Discount factor
    lr = 5e-4             # Learning rate
    epsilon_start = 1.0   # Initial exploration rate
    epsilon_end = 0.01    # Final exploration rate
    epsilon_decay = 0.995 # Exploration decay rate
    target_update = 4     # Update target network every 4 episodes
    max_episodes = 1000   # Maximum number of training episodes
    max_steps = 1000      # Maximum steps per episode

    # Initialize Networks and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(capacity=100000)
    
    epsilon = epsilon_start
    episode_rewards = []

    print(f"\n2. Starting Training on {device}...")
    
    for episode in range(max_episodes):
        if ENV_V1:
            state, _ = env.reset()
        else:
            state = env.reset()
            
        total_reward = 0
        
        for step in range(max_steps):
            # Epsilon-Greedy Action Selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                action = q_values.argmax().item()
            
            # Step in the environment
            if ENV_V1:
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            else:
                next_state, reward, done, _ = env.step(action)
            
            # Store experience in replay buffer
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            # Train the network
            if len(replay_buffer) >= batch_size:
                b_state, b_action, b_reward, b_next_state, b_done = replay_buffer.sample(batch_size)
                
                b_state = torch.FloatTensor(b_state).to(device)
                b_action = torch.LongTensor(b_action).unsqueeze(1).to(device)
                b_reward = torch.FloatTensor(b_reward).unsqueeze(1).to(device)
                b_next_state = torch.FloatTensor(b_next_state).to(device)
                b_done = torch.FloatTensor(b_done).unsqueeze(1).to(device)
                
                # Compute Q values and target Q values
                current_q = policy_net(b_state).gather(1, b_action)
                with torch.no_grad():
                    max_next_q = target_net(b_next_state).max(1)[0].unsqueeze(1)
                    target_q = b_reward + (gamma * max_next_q * (1 - b_done))
                
                # Compute loss and optimize
                loss = F.mse_loss(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            if done:
                break
                
        # Decay Epsilon
        epsilon = max(epsilon_end, epsilon_decay * epsilon)
        episode_rewards.append(total_reward)
        
        # Update Target Network
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode: {episode + 1}/{max_episodes} | Avg Reward: {avg_reward:.2f} | Epsilon: {epsilon:.3f}")
            
            # Lunar Lander is solved if avg reward is 200 over 100 consecutive episodes
            if avg_reward >= 200:
                print(f"\nEnvironment solved in {episode + 1} episodes!")
                break
                
    env.close()
    print("\nTraining Complete.")
    
    # Save the trained model
    torch.save(policy_net.state_dict(), "lunar_lander_dqn.pth")
    print("Model saved to 'lunar_lander_dqn.pth'")

if __name__ == '__main__':
    main()
