import gymnasium as gym
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)


ENV_NAME = "CartPole-v1"

GAMMA = 0.99
LR = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 10000

EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995  # Per episode decay (was 0.9995 per step)

TARGET_UPDATE = 10  # episodes as np


env = gym.make(ENV_NAME)

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

policy_net = DQN(state_size, action_size)
target_net = DQN(state_size, action_size)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = deque(maxlen=MEMORY_SIZE)


def store(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))


def sample_batch():
    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    return (
        torch.FloatTensor(states),
        torch.LongTensor(actions).unsqueeze(1),
        torch.FloatTensor(rewards),
        torch.FloatTensor(next_states),
        torch.FloatTensor(dones)
    )


epsilon = EPS_START

def select_action(state):
    global epsilon
    if random.random() < epsilon:
        return env.action_space.sample()
    with torch.no_grad():
        state = torch.FloatTensor(state).unsqueeze(0)
        return policy_net(state).argmax().item()


def train():
    if len(memory) < BATCH_SIZE:
        return

    states, actions, rewards, next_states, dones = sample_batch()

    q_values = policy_net(states).gather(1, actions).squeeze()

    with torch.no_grad():
        next_q = target_net(next_states).max(1)[0]
        target = rewards + GAMMA * next_q * (1 - dones)

    loss = nn.MSELoss()(q_values, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Removed epsilon decay from here - will decay per episode instead


EPISODES = 300  # Reduced for faster training (was 500)

# Track rewards for plotting
rewards_history = []

def train_agent():
    """Training function"""
    global epsilon  # Add global epsilon declaration
    
    for episode in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        max_steps_per_episode = 500  # CartPole-v1 max is 500

        while steps < max_steps_per_episode:
            action = select_action(state)
            next_state, reward, done, _, _ = env.step(action)

            store(state, action, reward, next_state, done)
            train()

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        rewards_history.append(total_reward)
        
        # Decay epsilon AFTER each episode (not after each step!)
        if epsilon > EPS_END:
            epsilon *= EPS_DECAY
        
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[max(0, episode-9):episode+1])
            print(f"Episode {episode + 1}/{EPISODES} | Reward: {total_reward:.0f} | Avg (last 10): {avg_reward:.2f} | Epsilon: {epsilon:.3f}")

    # Save model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'cartpole_dqn.pt')
    torch.save(policy_net.state_dict(), model_path)
    print(f"\n💾 Model saved to {model_path}")

    # Plot training progress
    plt.figure(figsize=(10, 5))

    # Plot rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards_history, alpha=0.6, label='Episode Reward')
    avg_rewards = [np.mean(rewards_history[max(0, i-99):i+1]) for i in range(len(rewards_history))]
    plt.plot(avg_rewards, label='Average (last 100)', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)

    # Plot moving average
    plt.subplot(1, 2, 2)
    plt.plot(avg_rewards, linewidth=2, color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (last 100)')
    plt.title('Learning Curve')
    plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(script_dir, 'cartpole_training.png')
    plt.savefig(plot_path)
    print(f"📈 Training plot saved to {plot_path}")

    env.close()


# Test function
def test_agent(episodes=5, render=True):
    """Test the trained agent"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'cartpole_dqn.pt')
    
    # Load trained model
    test_net = DQN(state_size, action_size)
    test_net.load_state_dict(torch.load(model_path))
    test_net.eval()
    
    print("\n🎯 Testing trained agent...")
    
    test_env = gym.make(ENV_NAME, render_mode='human' if render else None)
    test_rewards = []
    
    for episode in range(episodes):
        state, _ = test_env.reset()
        total_reward = 0
        
        while True:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action = test_net(state_tensor).argmax().item()
            
            state, reward, done, _, _ = test_env.step(action)
            total_reward += reward
            
            if done:
                break
        
        test_rewards.append(total_reward)
        print(f"Test Episode {episode + 1}/{episodes} | Reward: {total_reward:.0f}")
    
    test_env.close()
    
    print(f"\n📊 Test Results:")
    print(f"   Average Reward: {np.mean(test_rewards):.2f}")
    print(f"   Max Reward: {max(test_rewards):.0f}")
    print(f"   Min Reward: {min(test_rewards):.0f}")



# train_agent()

test_agent(episodes=10)

