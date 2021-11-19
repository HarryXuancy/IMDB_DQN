"""
Use Reinforcement learning(DQN) to classify IMDB
"""

import random
import numpy as np
from DQN_agent import DQN
from IMDB_env import environment

#主函数
RL = DQN()
env = environment()
total_step = 0

for i_episode in range(1000):
    #随机初始态
    observation = env.reset()
    RL.reward = 0
    print('回合开始')
    while True:
        action = RL.choose_action(observation)
        observation_, reward, done = env.step(action)
        RL.reward += reward
        RL.store_transition(observation, action, reward,observation_,done)
        total_step += 1

        if total_step >= 1000:
            RL.learn()
            print("learned")

        if done:
            print('回合总收益为：', RL.reward)
            break
        
        observation = observation_

