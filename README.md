# Reinforcement Learning Implementations

This repository contains my journey of learning and implementing various Reinforcement Learning algorithms across different Gymnasium environments.

## Overview

This project demonstrates practical implementations of RL algorithms, progressing from simple tabular methods to advanced deep reinforcement learning techniques. Each environment folder contains complete training and testing code with optimized hyperparameters.

## Algorithms Implemented

- **Q-Learning** - Tabular method for discrete environments
- **Deep Q-Network (DQN)** - Deep RL with experience replay and target networks
- **REINFORCE** - Monte Carlo policy gradient method
- **Actor-Critic** - Policy gradient with value function baseline
- **Proximal Policy Optimization (PPO)** - State-of-the-art policy optimization

## Environments

The implementations cover various control tasks:
- Classic control (discrete and continuous action spaces)
- Robotic control tasks
- Balance and swing-up problems
- Navigation and landing tasks

### Implemented Environments

1. **FrozenLake-v1** - Q-Learning (Tabular) & Deep Q-Network (DQN)
2. **Taxi-v3** - Q-Learning (Tabular)
3. **CartPole-v1** - Deep Q-Network (DQN) & REINFORCE
4. **MountainCar-v0** - Deep Q-Network (DQN)
5. **LunarLander-v3** - Actor-Critic & Proximal Policy Optimization (PPO)
6. **BipedalWalker-v3** - PPO (Standard & Symmetric Walking)
7. **Acrobot-v1** - Proximal Policy Optimization (PPO)

---

*Add new environments here as you implement them*

## Features

- Train/Test mode switching for easy experimentation
- Model persistence (save/load trained agents)
- Performance metrics and visualization
- Optimized hyperparameters for stable training
- Clean, readable code structure

## Usage

Each environment folder contains a standalone Python script with:
- Training function with configurable timesteps
- Testing function with rendering support
- Comments explaining key concepts

Simply run the desired script and toggle between training and testing modes.

## Requirements

- Python 3.8+
- Gymnasium
- PyTorch
- NumPy
- Stable-Baselines3 (for PPO implementations)

## Learning Approach

This repository follows a step-by-step learning methodology:
1. Start with simple tabular methods
2. Progress to deep learning approaches
3. Understand algorithm trade-offs
4. Optimize for specific behaviors
5. Compare performance across different algorithms

## Results

All implementations achieve solid performance on their respective environments, with detailed metrics available in test outputs including success rates, average scores, and convergence behavior.

---

*This is an ongoing learning project focused on understanding RL fundamentals through practical implementation.*
