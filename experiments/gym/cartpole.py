import sys
import gymnasium as gym
sys.modules['gym'] = gym

import math
import random
import numpy as np

from trees.models import DecisionTree
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


class SimpleStrat:
    def predict(self, obs):
        return 0 if obs[3] < 0.0 else 1


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    # tree = DecisionTree.load_from_file('./automated/cartpole/generated/constructed_0/trees/dt_original.json')
    # tree = DecisionTree.load_from_file('./automated/cartpole/generated/constructed_0/trees/dt_old_max_parts.json')
    tree = DecisionTree.load_from_file('./automated/cartpole/generated/constructed_0/trees/dt_new_max_parts_trans.json')
    # tree = DecisionTree.load_from_file('../cartpole/strat_maxparts.json')
    # tree = SimpleStrat()

    # env = make_vec_env('CartPole-v1')
    # try:
    #     model = PPO.load('../cartpole_strat_ppo.zip')
    # except FileNotFoundError:
    #     model = PPO('MlpPolicy', env, verbose=1)
    #     model.learn(250000)
    #     model.save('../cartpole_strat_ppo')

    rewards = []
    for epoch in range(1000):

        obs, info = env.reset()
        # obs = env.reset()
        epoch_reward = 0

        done, trunc = False, False
        while not (done or trunc):

            action = tree.predict(obs)
            # action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = env.step(action)
            # obs, reward, done, info = env.step(action)

            epoch_reward += reward

        rewards.append(epoch_reward)

        # if (epoch + 1) % 10 == 0:
        #     print(f'mean after {epoch + 1} epochs: {np.mean(rewards):02f}')

    print('mean: {} (+/- {})'.format(
        np.round(np.mean(rewards), 2),
        np.round(np.std(rewards), 2)
    ))
