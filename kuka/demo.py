import time
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from custom_kuka_env import KukaGraspingEnv

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
            chest_idx = int(input("Enter the target box number (0-3), or enter -1 to exit:"))
            if chest_idx == -1:
                break

            if 0 <= chest_idx <= 3:
                demo_agent(model, target_chest_idx=chest_idx, render=True)
            else:
                print("Invalid box number, please enter an integer between 0 and 3")
        except ValueError:
            print("Please enter a valid integer.")
        except KeyboardInterrupt:
            break

    print("The demonstration is over.")

if __name__ == "__main__":
    interactive_demo()