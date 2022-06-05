import gym
import os
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper

# ログフォルダの生成
log_dir = './logs/'
os.makedirs(log_dir, exist_ok=True)

# 環境の生成
unity_env = UnityEnvironment(
    file_name='3DBall', worker_id=0, no_graphics=False)
env = UnityToGymWrapper(unity_env=unity_env, uint8_visual=True)
env = Monitor(env, log_dir, allow_early_resets=True)
env = DummyVecEnv([lambda: env])

# モデルの生成
model = PPO('MlpPolicy', env, verbose=1)

# モデルの読み込み
# model = PPO.load('mlagents_model')

# モデルの学習
model.learn(total_timesteps=12800)

# モデルの保存
model.save('mlagents_model')

# モデルのテスト
state = env.reset()
total_reward = 0
while True:
    # 環境の描画
    env.render()

    # 行動の取得
    action, _ = model.predict(state)

    # 1ステップ実行
    state, reward, done, info = env.step(action)
    total_reward += reward[0]

    # エピソード完了
    if done:
        print('reward:', total_reward)
        state = env.reset()
        total_reward = 0
