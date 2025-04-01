from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from custom_kuka_env import KukaGraspingEnv

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

if __name__ == "__main__":
    # 定义环境创建函数
    def make_env(render=False):
        return KukaGraspingEnv(render=render)

    # 训练代理
    print("Start training the KUKA crawling agent...")
    model = train_kuka_agent(lambda: make_env(render=False), total_timesteps=100000)