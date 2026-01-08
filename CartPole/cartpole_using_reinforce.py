import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)


env = gym.make("CartPole-v1")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy = PolicyNetwork(state_dim, action_dim)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)

gamma = 0.99


def select_action(state):
    state = torch.FloatTensor(state).unsqueeze(0)
    probs = policy(state)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action)


def compute_returns(rewards, gamma):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns)


def test_agent(env, policy, episodes=5, render=True, max_steps=500, deterministic=True):
 
    print(f"\nTesting trained agent ({'Deterministic' if deterministic else 'Stochastic'} mode).")
    
    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0

        while steps < max_steps:
            state_t = torch.FloatTensor(state).unsqueeze(0)

            with torch.no_grad():
                probs = policy(state_t)
                
                if deterministic:
                    action = torch.argmax(probs, dim=1).item()  # Best action
                else:
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample().item()  # Random sampling

            state, reward, done, _, _ = env.step(action)
            total_reward += reward
            steps += 1

            if render:
                env.render()

            if done:
                break

        print(f"Test Episode {ep+1}, Reward: {total_reward}, Steps: {steps}")

    env.close()


EPISODES = 500

def train_agent():
    
    print("Starting REINFORCE training.\n")
    
    for episode in range(EPISODES):

        state, _ = env.reset()
        log_probs = []
        rewards = []
        total_reward = 0

        while True:
            action, log_prob = select_action(state)
            next_state, reward, done, _, _ = env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)

            state = next_state
            total_reward += reward

            if done:
                break

        returns = compute_returns(rewards, gamma)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0
        for log_prob, G in zip(log_probs, returns):
            loss -= log_prob * G

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {total_reward}")
    
    # Save model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'cartpole_reinforce.pt')
    torch.save(policy.state_dict(), model_path)
    print(f"\n💾 Model saved to {model_path}")
    
    env.close()


# train_agent()


test_env = gym.make("CartPole-v1", render_mode='human')
policy.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cartpole_reinforce.pt')))

# Deterministic testing (best performance:100% results)
test_agent(test_env, policy, episodes=5, deterministic=True)

# Stochastic testing 
# test_agent(test_env, policy, episodes=5, deterministic=False)
