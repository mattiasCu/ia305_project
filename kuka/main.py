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
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)

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

        # 初期训练时固定目标编号为 0
        self.training_phase = True  # 你可以设置一个开关
        if self.training_phase:
            self.target_chest_idx = 0
        else:
            self.target_chest_idx = np.random.randint(0, 4)

        self.grasp_constraint_id = None

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

        self.just_grasped = False
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
        observation = np.array(joint_states + list(end_effector_pos) + target_chest_pos, dtype=np.float32)

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

        # ✅ 每步距离 shaping 奖励（持续）
        reward += (1.0 - np.tanh(distance * 5)) * 0.5

        # ✅ 对准高度小奖励
        if distance_xy < 0.2 and z_diff < 0.1:
            reward += 0.3

        # ✅ 抓取动作奖励（立即响应）
        if hasattr(self, "just_grasped") and self.just_grasped:
            reward += 0.5
            self.just_grasped = False  # 只奖励一次

        # ✅ 成功抓取提起
        if chest_pos[2] > 0.2:
            reward += 1.0
            done = True
            info["success"] = True

        # ❌ 防撞惩罚
        if end_effector_pos[2] < 0.08:
            reward -= 0.1

        assert np.isfinite(reward), "Reward not finite"

        return reward, done, info

    def close(self):
        p.disconnect()


def train_kuka_agent(env_fn, total_timesteps=100000, log_dir="./logs/"):
    """
    训练KUKA机械臂抓取代理
    """
    # 创建向量化环境
    vec_env = DummyVecEnv([env_fn])

    # 创建PPO模型
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=None,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
    )

    # 训练模型
    model.learn(total_timesteps=total_timesteps)

    # 保存模型
    model.save("kuka_grasping_model")

    return model


def evaluate_agent(model, env_fn, n_eval_episodes=10, render=True):
    """
    评估训练好的代理
    """
    # 创建评估环境
    eval_env = env_fn(render=render)

    # 评估模型
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=n_eval_episodes)

    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # 关闭环境
    eval_env.close()


def demo_agent(model, target_chest_idx=0, render=True):
    """
    演示训练好的代理
    """
    # 创建演示环境
    env = KukaGraspingEnv(render=render)

    # 重置环境，指定目标箱子
    obs, _ = env.reset(target_chest_idx=target_chest_idx)

    done = False
    total_reward = 0

    while not done:
        # 获取模型动作
        action, _ = model.predict(obs, deterministic=True)

        # 执行动作
        obs, reward, done, _, info = env.step(action)

        total_reward += reward

        if render:
            time.sleep(0.01)

    print(f"Demo completed. Total reward: {total_reward:.2f}")
    print(f"Success: {info['success']}")

    # 关闭环境
    env.close()


def interactive_demo():
    """
    交互式演示，用户输入目标箱子编号
    """
    # 加载训练好的模型
    model = PPO.load("kuka_grasping_model")

    while True:
        try:
            chest_idx = int(input("输入目标箱子编号(0-3)，或输入-1退出: "))
            if chest_idx == -1:
                break

            if 0 <= chest_idx <= 3:
                demo_agent(model, target_chest_idx=chest_idx, render=True)
            else:
                print("无效的箱子编号，请输入0-3之间的整数。")
        except ValueError:
            print("请输入有效的整数。")
        except KeyboardInterrupt:
            break

    print("演示结束。")


def main():
    # 创建日志目录
    os.makedirs("./logs/", exist_ok=True)

    # 定义环境创建函数
    def make_env(render=False):
        return KukaGraspingEnv(render=render)

    # 训练代理
    print("开始训练KUKA抓取代理...")
    model = train_kuka_agent(lambda: make_env(render=False), total_timesteps=100000)

    # 评估代理
    print("评估训练好的代理...")
    evaluate_agent(model, make_env, n_eval_episodes=5, render=True)

    # 交互式演示
    print("开始交互式演示...")
    interactive_demo()


if __name__ == "__main__":
    #main()
    interactive_demo()