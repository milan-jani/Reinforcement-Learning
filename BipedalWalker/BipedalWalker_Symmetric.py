import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import numpy as np
import os


# ============================================
# Custom Wrapper for Symmetric Walking
# ============================================
class SymmetricWalkWrapper(gym.Wrapper):
    """
    Wrapper that penalizes asymmetric leg movements
    to encourage both legs working equally
    """
    def __init__(self, env):
        super().__init__(env)
        self.symmetry_penalty_weight = 0.1  # Adjust this for stronger/weaker symmetry
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Calculate symmetry penalty
        # Actions: [hip_1, knee_1, hip_2, knee_2]
        left_leg_action = np.array([action[0], action[1]])   # Hip 1, Knee 1
        right_leg_action = np.array([action[2], action[3]])  # Hip 2, Knee 2
        
        # Penalize difference between left and right leg actions
        symmetry_diff = np.abs(left_leg_action - right_leg_action).sum()
        symmetry_penalty = -self.symmetry_penalty_weight * symmetry_diff
        
        # Add symmetry penalty to reward
        modified_reward = reward + symmetry_penalty
        
        return obs, modified_reward, done, truncated, info


# ============================================
# Create Environment with Symmetry Wrapper
# ============================================
env = gym.make("BipedalWalker-v3")
env = SymmetricWalkWrapper(env)  # Add symmetry wrapper
env = Monitor(env)

# Create PPO model - Balanced config
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
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    policy_kwargs=dict(
        net_arch=[dict(pi=[256, 256], vf=[256, 256])]
    )
)

TIMESTEPS = 2_000_000

# ============================================
# TRAINING
# ============================================
def train_symmetric():
    print("🚀 Starting training for SYMMETRIC WALK...\n")
    print("⏱️  This will take 45-60 minutes for 2M timesteps!\n")
    print("🎯 Target: Both legs working equally!\n")
    
    # Train
    model.learn(total_timesteps=TIMESTEPS)
    
    # Save model in same folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "bipedal_symmetric")
    model.save(model_path)
    
    env.close()
    print(f"✅ Training finished and model saved to: {model_path}.zip")


# ============================================
# TESTING
# ============================================
def test_agent(episodes=5, render=True):
    """Test the trained symmetric agent"""
    test_env = gym.make("BipedalWalker-v3", render_mode='human' if render else None)
    test_env = SymmetricWalkWrapper(test_env)  # Use same wrapper for consistency
    
    # Load model from same folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "bipedal_symmetric")
    loaded_model = PPO.load(model_path)

    print("\n🎯 Testing symmetric trained agent...")
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

# To TRAIN: Uncomment (45-60 min for symmetric walk!)
train_symmetric()

# To TEST: Uncomment (after training complete!)
# test_agent(episodes=5)
