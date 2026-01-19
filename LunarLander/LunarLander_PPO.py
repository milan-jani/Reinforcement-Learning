import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import os

env_id = "LunarLander-v3"

print(f"Environment: {env_id}")
print("Using PPO algorithm (Stable-Baselines3)\n")


# ============================================
# Training Function with PPO
# ============================================
def train_lunarlander():
    print("Training LunarLander with PPO\n")
    
    # Create vectorized environment (4 parallel environments)
    env = make_vec_env(env_id, n_envs=4)
    
    # Create PPO model with optimized hyperparameters
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
        ent_coef=0.01,  # Entropy coefficient for exploration
        vf_coef=0.5,     # Value function coefficient
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=None
    )
    
    print("Starting training for 1,000,000 timesteps...")
    print("Expected time: ~45-60 minutes")
    print("Goal: Achieve 200+ average reward\n")
    
    # Train the model
    model.learn(total_timesteps=1_000_000, progress_bar=True)
    
    # Save model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "lunarlander_ppo_sb3.zip")
    model.save(model_path)
    
    print(f"\nTraining complete!")
    print(f"Model saved to: {model_path}")
    
    env.close()


# ============================================
# Testing Function
# ============================================
def test_lunarlander(episodes=10, render=True):
    """Test the trained PPO agent"""
    print("\nTesting LunarLander Agent...\n")
    
    # Load model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "lunarlander_ppo_sb3.zip")
    model = PPO.load(model_path)
    
    # Create test environment
    test_env = gym.make(env_id, render_mode='human' if render else None)
    
    test_scores = []
    successful_landings = 0
    
    for episode in range(episodes):
        state, _ = test_env.reset()
        total_reward = 0
        steps = 0
        
        while steps < 1000:
            action, _ = model.predict(state, deterministic=True)
            state, reward, done, truncated, _ = test_env.step(action)
            total_reward += reward
            steps += 1
            
            if done or truncated:
                break
        
        test_scores.append(total_reward)
        if total_reward >= 200:
            successful_landings += 1
        
        status = "SUCCESS" if total_reward >= 200 else "CRASH"
        print(f"Episode {episode + 1}/{episodes} | Score: {total_reward:.2f} | Steps: {steps} | {status}")
    
    avg_score = sum(test_scores) / len(test_scores)
    success_rate = (successful_landings / episodes) * 100
    
    print(f"\nTest Results:")
    print(f"   Average Score: {avg_score:.2f}")
    print(f"   Best Score: {max(test_scores):.2f}")
    print(f"   Worst Score: {min(test_scores):.2f}")
    print(f"   Success Rate: {success_rate:.1f}% ({successful_landings}/{episodes})")
    print(f"\n   Goal (200+): {'ACHIEVED!' if avg_score >= 200 else 'Keep training'}")
    
    test_env.close()

# ============================================
# train_lunarlander()


test_lunarlander(episodes=10)
