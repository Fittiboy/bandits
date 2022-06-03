from collections import defaultdict
from pprint import pprint
from math import log

import pandas as pd
import numpy as np
import gym_bandits
import gym

env = gym.make("BanditTenArmedGaussian-v0")
action_values = np.zeros(env.action_space.n)
action_counts = np.zeros(env.action_space.n)
TIMESTEP_MULTIPLIER = 10000
REWARDS = 0
C = 0.1

for t in range(TIMESTEP_MULTIPLIER * env.action_space.n):
    action = None
    for a in range(env.action_space.n):
        if action_counts[a] == 0:
            action = a
            break
    if action is None:
        confidence = [action_values[a] + C * (log(t+1)/action_counts[a])**0.5
                      for a in range(env.action_space.n)]
        action = max(list(range(env.action_space.n)), key=lambda x:
                     confidence[x])
    state, reward, _, _ = env.step(action)
    action_counts[action] += 1
    action_values[action] += (1 / action_counts[action]) * \
        (reward - action_values[action])
    REWARDS += reward

MAX_REWARD = TIMESTEP_MULTIPLIER * env.action_space.n * \
    max(env.r_dist, key=lambda x: x[0])[0]
reward_percentage = (REWARDS / MAX_REWARD) * 100
print(f"\nTotal reward: {reward_percentage:.2f}%\n")
data_1 = {r[0]: p for r, p in zip(env.r_dist, action_values)}
data_2 = {r[0]: c for r, c in zip(env.r_dist, confidence)}
data_1 = pd.DataFrame(data_1.items(), columns=["Real", "Predicted"])
data_2 = pd.DataFrame(data_2.items(), columns=["Real", "Predicted UCB"])
data = pd.merge(data_1, data_2, on="Real")
pprint(data)
