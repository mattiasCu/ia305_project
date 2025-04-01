# Part1 Open The Chests - Parallel Event RL Environment

This project simulates a simplified decision-making environment where an agent must respond to randomly occurring events such as **locked chests**, **normal chests**, **traps**, and **keys**.

## 🎮 Environment Overview

The environment simulates a sequential event stream with the following event types:

| Event Type       | Code | Description                    | Action Effect                            |
|------------------|------|--------------------------------|-------------------------------------------|
| Locked Chest     | 0    | Can only be opened with key   | +2 reward (if key), -1 (if not)           |
| Normal Chest     | 1    | Can be opened freely          | +1 reward                                 |
| Trap             | 2    | Should be avoided             | -0.5 penalty if opened                    |
| Key              | 3    | Grants a key if collected     | +0.5 reward, adds key to inventory        |

---





## Part2 KUKA Grasping Environment (PyBullet + Reinforcement Learning)

After the previous part, we've already know the goals of chests to reach

This part implements a PyBullet-based robotic grasping simulation using the KUKA IIWA arm. The goal is to train a reinforcement learning agent (PPO) to pick up target cubes located at predefined positions.

---

### 🚀 Features

- ✅ Multiple target chests at 4 fixed positions
- ✅ Action space includes `[x, y, z, gripper_cmd]`
- ✅ Observation includes joint states, gripper fingertip position, target position, one-hot encoded goal, and gripper opening
- ✅ Reward shaped for distance, alignment, successful grasp, and penalties for ground contact
- ✅ Uses `stable-baselines3` PPO algorithm

---
