import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import os

# Create environment
env = gym.make("BipedalWalker-v3")
env = Monitor(env)

# Create PPO model
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,  # Added entropy for exploration
    verbose=1
)

TIMESTEPS = 1_000_000

# ============================================
# TRAINING
# ============================================
def train_bipedal():
    print("🚀 Starting training... This will take 25-30 minutes!\n")
    
    # Train
    model.learn(total_timesteps=TIMESTEPS)
    
    # Save model in same     folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "bipedal_ppo")
    model.save(model_path)
    
    env.close()
    print(f"✅ Training finished and model saved to: {model_path}.zip")


# ============================================
# TESTING
# ============================================
def test_agent(episodes=5, render=True):
    """Test the trained agent"""
    test_env = gym.make("BipedalWalker-v3", render_mode='human' if render else None)
    
    # Load model from same folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "bipedal_ppo")
    loaded_model = PPO.load(model_path)

    print("\n🎯 Testing trained agent...")
    test_rewards = []

    for ep in range(episodes):
        obs, _ = test_env.reset()
        done = False
        truncated = False
        total_reward = 0
        steps = 0

        while not (done or truncated):
            action, _ = loaded_model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = test_env.step(action)
            total_reward += reward
            steps += 1

        test_rewards.append(total_reward)
        print(f"Test Episode {ep + 1}: Reward: {total_reward:.2f}, Steps: {steps}")

    print(f"\n📊 Test Results:")
    print(f"   Average Reward: {sum(test_rewards)/len(test_rewards):.2f}")
    print(f"   Max Reward: {max(test_rewards):.2f}")
    print(f"   Min Reward: {min(test_rewards):.2f}")

    test_env.close()


# ============================================
# RUN MODE
# ============================================

# To TRAIN: Uncomment the line below
# train_bipedal()

# To TEST: Uncomment the line below (after training is complete!)
test_agent(episodes=5)
