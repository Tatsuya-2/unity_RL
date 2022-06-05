import gym
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper

# 環境の生成
unity_env = UnityEnvironment('3DBall')
env = UnityToGymWrapper(unity_env=unity_env, uint8_visual=True)

# ランダム行動による動作確認
state = env.reset()
while True:
    # 環境の描画
    env.render()

    # 行動の取得
    action = env.action_space.sample()

    # 1ステップ実行
    state, reward, done, info = env.step(action)
    print('reward:', reward)

    # エピソード完了
    if done:
        print('done')
        state = env.reset()
