import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)


env = gym.make("LunarLander-v3")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

model = ActorCritic(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=3e-4)

gamma = 0.99
eps_clip = 0.2
K_epochs = 4


def select_action(state, model):
    state = torch.FloatTensor(state).unsqueeze(0)
    probs, value = model(state)
    dist = Categorical(probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action), value


def compute_returns(rewards, dones, gamma):
    returns = []
    G = 0
    for r, done in zip(reversed(rewards), reversed(dones)):
        if done:
            G = 0
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)


EPISODES = 1000

def train_agent():
    print("🚀 Starting PPO training on LunarLander...\n")
    
    episode_rewards = []
    
    for episode in range(EPISODES):
        state, _ = env.reset()
        
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []
        
        ep_reward = 0
        steps = 0
        max_steps = 1000

        while steps < max_steps:
            action, log_prob, value = select_action(state, model)
            next_state, reward, done, truncated, _ = env.step(action)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done or truncated)
            values.append(value)

            state = next_state
            ep_reward += reward
            steps += 1

            if done or truncated:
                break

        # PPO Update after each episode
        if len(rewards) > 0:
            returns = compute_returns(rewards, dones, gamma)
            values_t = torch.cat(values).squeeze().float()

            advantages = returns - values_t.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            states_t = torch.tensor(np.array(states), dtype=torch.float32)
            actions_t = torch.tensor(actions)
            old_log_probs = torch.stack(log_probs).detach()

            for _ in range(K_epochs):
                probs, new_values = model(states_t)
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(actions_t)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs)

                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(new_values.squeeze(), returns)

                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

        episode_rewards.append(ep_reward)

        if episode % 20 == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            print(f"Episode {episode}/{EPISODES} | Reward: {ep_reward:.1f} | Avg(100): {avg_reward:.1f}")
    
    # Save model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    torch.save(model.state_dict(), os.path.join(script_dir, "lunarlander_ppo.pt"))
    print("\n✅ Model saved!")
    env.close()


def test_agent(episodes=5, render=True):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model.load_state_dict(torch.load(os.path.join(script_dir, "lunarlander_ppo.pt")))
    model.eval()
    
    test_env = gym.make("LunarLander-v3", render_mode='human' if render else None)
    
    print("\n🎯 Testing trained agent...")
    test_rewards = []
    
    for ep in range(episodes):
        state, _ = test_env.reset()
        total_reward = 0
        steps = 0
        
        while steps < 1000:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                probs, _ = model(state_t)
                action = torch.argmax(probs, dim=1).item()
            
            state, reward, done, truncated, _ = test_env.step(action)
            total_reward += reward
            steps += 1
            
            if done or truncated:
                break
        
        test_rewards.append(total_reward)
        print(f"Test Episode {ep+1}, Reward: {total_reward:.1f}, Steps: {steps}")
    
    print(f"\n📊 Test Results:")
    print(f"   Average Reward: {np.mean(test_rewards):.2f}")
    print(f"   Max Reward: {max(test_rewards):.1f}")
    print(f"   Min Reward: {min(test_rewards):.1f}")
    
    test_env.close()


# ============================================
# RUN MODE
# ============================================

# To TRAIN: Uncomment
# train_agent()

# To TEST: Uncomment
test_agent(episodes=5)


