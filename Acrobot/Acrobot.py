import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os

env_id = "Acrobot-v1"

print(f"Environment: {env_id}")
print("Using PPO algorithm for training\n")

# ============================================
# Training Function with PPO
# ============================================
def train_acrobot():
    print("🎮 Training Acrobot with PPO\n")
    
    # Create vectorized environment
    env = make_vec_env(env_id, n_envs=4)
    
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
        verbose=1,
        tensorboard_log=None
    )
    
    print("Starting training for 300,000 timesteps...")
    print("This should take ~15-20 minutes for better results\n")
    
    # Train the model
    model.learn(total_timesteps=300000)
    
    # Save model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "acrobot_ppo.zip")
    model.save(model_path)
    
    print(f"\n✅ Training complete!")
    print(f"💾 Model saved to: {model_path}")
    
    env.close()


# ============================================
# Testing Function
# ============================================
def test_acrobot(episodes=5, render=True):
    """Test the trained PPO agent"""
    print("\n🎯 Testing Acrobot Agent...\n")
    
    # Load model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "acrobot_ppo.zip")
    model = PPO.load(model_path)
    
    # Create test environment
    test_env = gym.make(env_id, render_mode='human' if render else None)
    
    test_scores = []
    test_steps = []
    
    for episode in range(episodes):
        state, _ = test_env.reset()
        total_reward = 0
        steps = 0
        
        while steps < 500:
            action, _ = model.predict(state, deterministic=True)
            state, reward, done, truncated, _ = test_env.step(action)
            total_reward += reward
            steps += 1
            
            if done or truncated:
                break
        
        test_scores.append(total_reward)
        test_steps.append(steps)
        print(f"Episode {episode + 1}/{episodes} | Score: {total_reward:.2f} | Steps: {steps}")
    
    print(f"\n📊 Test Results:")
    print(f"   Average Score: {sum(test_scores)/len(test_scores):.2f}")
    print(f"   Average Steps: {sum(test_steps)/len(test_steps):.2f}")
    print(f"   Best Score: {max(test_scores):.2f}")
    
    test_env.close()



# train_acrobot()

test_acrobot(episodes=5)