import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import os

# Create environment
env = gym.make("BipedalWalker-v3")
env = Monitor(env)

# Create PPO model with optimized hyperparameters for perfect walk
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=1e-4,  # Lower for fine-tuning
    n_steps=4096,  # More steps per update
    batch_size=128,  # Larger batches for stable learning
    n_epochs=20,  # More epochs per batch
    gamma=0.995,  # Higher gamma for long-term rewards
    gae_lambda=0.98,  # Better advantage estimation
    clip_range=0.15,  # Tighter clipping for stability
    ent_coef=0.005,  # Less entropy for more deterministic policy
    vf_coef=0.5,  # Value function coefficient
    max_grad_norm=0.5,  # Gradient clipping
    verbose=1,
    policy_kwargs=dict(
        net_arch=[dict(pi=[256, 256], vf=[256, 256])]  # Deeper network
    )
)

TIMESTEPS = 2_000_000  # Double the training time for perfection!

# ============================================
# TRAINING
# ============================================
def train_bipedal():
    print("🚀 Starting training for PERFECT WALK...\n")
    print("⏱️  This will take 45-60 minutes for 2M timesteps!\n")
    print("🎯 Target: +300 reward (perfect walk)\n")
    
    # Train
    model.learn(total_timesteps=TIMESTEPS)
    
    # Save model in same folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "bipedal_ppo_perfect")
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
    model_path = os.path.join(script_dir, "bipedal_ppo_perfect")
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

# To TRAIN: Uncomment the line below (45-60 min for perfect walk!)
# train_bipedal()

# To TEST: Uncomment the line below (after training is complete!)
test_agent(episodes=5)
