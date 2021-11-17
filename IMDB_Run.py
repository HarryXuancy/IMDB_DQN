"""
Use Reinforcement learning(DQN) to classify IMDB
"""

import random
import numpy as np
from DQN_agent import DQN
from IMDB_env import environment

#主函数
RL = DQN()
print(1)
env = environment()
print(2)
total_step = 0

for i_episode in range(1000):
    #随机初始态
    observation = env.reset()
    print(3)
    while True:
        action = RL.choose_action(observation)
        observation_, reward, done = env.step(action)
        RL.store_transition(observation, action, reward,observation_,done)
        total_step += 1

        if total_step >= 1000:
            RL.learn()
            print("learned")

        if done:
            break
        
        observation = observation_
