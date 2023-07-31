import sys
import gymnasium as gym
sys.modules['gym'] = gym

import math
import random
import numpy as np
import bouncing_ball

from trees.models import Tree
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

if __name__ == '__main__':
    t_step = 0.3
    N = 120
    sb3 = True

    if sb3:
        env = make_vec_env(
            'bouncing_ball/BouncingBall-v0',
            env_kwargs={ 'ts_size': t_step, 'render_mode': 'human' }
        )

        try:
            model = PPO.load('./gym_envs/ppo_ball.zip')
        except FileNotFoundError:
            model = PPO('MlpPolicy', env, verbose=1)
            model.learn(250000)
            model.save('./gym_envs/ppo_ball')
    else:
        env = gym.make('bouncing_ball/BouncingBall-v0', ts_size=t_step)

        model = Tree.load_from_file(
            './automated/bouncing_ball/generated/constructed_0/trees/dt_max_parts.json'
        )

    epoch_rewards = []
    for epoch in range(1000):
        obs = env.reset() if sb3 else env.reset()[0]

        rewards = []
        for t in range(int(N / t_step)):

            if sb3:
                action, _states = model.predict(obs, deterministic=True)
            else:
                action = int(model.get(obs[::-1]))

            res = env.step(action)
            obs, reward, done, info = res if sb3 else res[:3] + res[4:]

            rewards.append(reward)
            if done:
                print(f'epoch {epoch + 1}: DEAD', t * t_step)
                break

        # env.render()
        if (epoch + 1) % 100 == 0:
            print(f'reward for epoch {epoch + 1}: {sum(rewards)}')

        epoch_rewards.append(sum(rewards))

    epoch_rewards = np.array(epoch_rewards)
    print('\nResult: {} (+/- {})'.format(
        epoch_rewards.mean(), epoch_rewards.std()
    ))

    if not done:
        env.render()
