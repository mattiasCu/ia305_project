





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
