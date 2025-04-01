import os
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


class KukaGraspingEnv(gym.Env):
    def __init__(self, render=False):
        super(KukaGraspingEnv, self).__init__()

        self.grasp_constraint_id = None

        # 连接物理引擎
        self.render_mode = "human" if render else None
        if self.render_mode == "human":
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        # 定义动作空间和观察空间
        # 动作：末端执行器的dx, dy, dz, 抓取
        # 控制末端目标位置范围更小，防止初期乱跳
        self.action_space = spaces.Box(
            low=np.array([-0.4, -0.4, 0.1]),
            high=np.array([0.4, 0.4, 0.6]),
            dtype=np.float32
        )

        # 观察：机械臂状态(关节位置和速度) + 目标箱子位置
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)


        # 箱子的位置
        self.chest_positions = [
            [0.5, -0.5, 0.0],  # 箱子 0
            [0.5, 0.5, 0.0],  # 箱子 1
            [-0.5, 0.5, 0.0],  # 箱子 2
            [-0.5, -0.5, 0.0]  # 箱子 3
        ]

        # 目标箱子索引
        self.target_chest_idx = 0

        # 初始化环境
        self.reset()

    def reset(self, *, seed=None, options=None, target_chest_idx=None):
        super().reset(seed=seed)

        # ✅ 每次训练或测试，根据传参决定目标箱子编号
        if target_chest_idx is not None:
            self.target_chest_idx = target_chest_idx
        else:
            self.target_chest_idx = np.random.randint(0, 4)  # ✅ 随机选一个目标

        self.grasp_constraint_id = None
        self.just_grasped = False

        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.plane_id = p.loadURDF("plane.urdf")
        self.kuka_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0])
        self.kuka_end_effector_index = 6
        self.reset_arm()

        for _ in range(10):
            p.stepSimulation()

        self.chest_ids = []
        for i, pos in enumerate(self.chest_positions):
            new_pos = pos.copy()
            new_pos[2] = 0.05  # 提高一点，避免贴地
            chest_id = p.loadURDF("cube_small.urdf", new_pos, p.getQuaternionFromEuler([0, 0, 0]))
            p.changeVisualShape(chest_id, -1,
                                rgbaColor=[0.8, 0.2, 0.2, 1] if i == self.target_chest_idx else [0.2, 0.2, 0.8, 1])
            self.chest_ids.append(chest_id)

        for _ in range(10):
            p.stepSimulation()

        observation = self._get_observation()

        return observation, {"target_idx": self.target_chest_idx}

    def reset_arm(self):
        # 重置机械臂到初始位置
        for i in range(p.getNumJoints(self.kuka_id)):
            p.resetJointState(self.kuka_id, i, 0)

    def _get_observation(self):
        # 获取机械臂关节状态
        joint_states = []
        for i in range(p.getNumJoints(self.kuka_id)):
            state = p.getJointState(self.kuka_id, i)
            joint_states.extend([state[0], state[1]])  # 位置和速度

        # 获取目标箱子位置
        target_chest_pos = self.chest_positions[self.target_chest_idx]

        # 获取末端执行器位置
        end_effector_state = p.getLinkState(self.kuka_id, self.kuka_end_effector_index)
        end_effector_pos = end_effector_state[0]

        # 组合观察
        one_hot = np.zeros(4)
        one_hot[self.target_chest_idx] = 1.0
        observation = np.array(joint_states + list(end_effector_pos) + target_chest_pos + list(one_hot),
                               dtype=np.float32)


        return observation

    def step(self, action):
        # 将动作作为末端目标位置
        target_pos = np.array(action)
        target_pos = np.clip(target_pos, [-0.4, -0.4, 0.1], [0.4, 0.4, 0.6])

        # IK 解算
        joint_poses = p.calculateInverseKinematics(
            self.kuka_id,
            self.kuka_end_effector_index,
            target_pos,
            maxNumIterations=100,
            residualThreshold=0.001
        )

        # 控制执行
        for i in range(len(joint_poses)):
            if i < p.getNumJoints(self.kuka_id):
                p.setJointMotorControl2(
                    bodyIndex=self.kuka_id,
                    jointIndex=i,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=joint_poses[i],
                    force=500,
                    positionGain=0.03
                )

        for _ in range(30):  # 增强控制反馈
            p.stepSimulation()
            if self.render_mode == "human":
                time.sleep(0.01)

        end_effector_pos = np.array(p.getLinkState(self.kuka_id, self.kuka_end_effector_index)[0])
        chest_pos = np.array(p.getBasePositionAndOrientation(self.chest_ids[self.target_chest_idx])[0])
        chest_center = chest_pos + np.array([0, 0, 0.05])

        distance = np.linalg.norm(end_effector_pos - chest_center)
        z_diff = abs(end_effector_pos[2] - chest_center[2])

        # ✅ 抓取逻辑 + 奖励标志
        if distance < 0.15 and z_diff < 0.05 and self.grasp_constraint_id is None:
            self.grasp_constraint_id = p.createConstraint(
                parentBodyUniqueId=self.kuka_id,
                parentLinkIndex=self.kuka_end_effector_index,
                childBodyUniqueId=self.chest_ids[self.target_chest_idx],
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0.1],
                childFramePosition=[0, 0, 0]
            )
            self.just_grasped = True  # ✅ 记录抓取动作

        # 可选释放逻辑
        if self.grasp_constraint_id is not None and chest_pos[2] > 0.4:
            try:
                p.removeConstraint(self.grasp_constraint_id)
            except Exception:
                pass
            self.grasp_constraint_id = None

        observation = self._get_observation()
        reward, done, info = self._compute_reward(end_effector_pos, chest_pos)

        return observation, reward, done, False, info

    def _compute_reward(self, end_effector_pos, chest_pos):
        chest_center = chest_pos + np.array([0, 0, 0.05])

        distance = np.linalg.norm(end_effector_pos - chest_center)
        distance_xy = np.linalg.norm(end_effector_pos[:2] - chest_center[:2])
        z_diff = abs(end_effector_pos[2] - chest_center[2])

        reward = 0
        done = False
        info = {"success": False}

        # 每步距离 shaping 奖励（持续）
        reward += (1.0 - np.tanh(distance * 5)) * 0.5

        # 对准高度小奖励
        if distance_xy < 0.2 and z_diff < 0.1:
            reward += 0.3

        # 抓取动作奖励（立即响应）
        if hasattr(self, "just_grasped") and self.just_grasped:
            reward += 0.5
            self.just_grasped = False  # 只奖励一次

        # 成功抓取提起
        if chest_pos[2] > 0.4:
            reward += 1.0
            done = True
            info["success"] = True

        # 防撞惩罚
        if end_effector_pos[2] < 0.08:
            reward -= 0.1

        assert np.isfinite(reward), "Reward not finite"

        return reward, done, info

    def close(self):
        p.disconnect()