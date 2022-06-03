from collections import defaultdict
from random import uniform, choices
from pprint import pprint

import pandas as pd
import numpy as np
import gym_bandits
import gym

env = gym.make("BanditTenArmedGaussian-v0")
action_values = np.zeros(env.action_space.n)
action_counts = np.zeros(env.action_space.n)
TIMESTEP_MULTIPLIER = 100000
REWARDS = 0
TEMP = 30

for t in range(TIMESTEP_MULTIPLIER * env.action_space.n):
    softmax = np.exp(action_values/TEMP) / np.sum(np.exp(action_values/TEMP))
    action = choices(list(range(env.action_space.n)), softmax)[0]
    state, reward, _, _ = env.step(action)
    action_counts[action] += 1
    action_values[action] += (1 / action_counts[action]) * \
        (reward - action_values[action])
    REWARDS += reward
    if TEMP > 0.1:
        TEMP *= 0.999

MAX_REWARD = TIMESTEP_MULTIPLIER * env.action_space.n * \
    max(env.r_dist, key=lambda x: x[0])[0]
reward_percentage = (REWARDS / MAX_REWARD) * 100
print(f"\nTotal reward: {reward_percentage:.2f}%\n")
data = {p: r[0] for p, r in zip(action_values, env.r_dist)}
pprint(pd.DataFrame(data.items(), columns=["Predicted", "Real"]))
print()
softmax = [f"{prob:.10f}%" for prob in softmax]
softmax = pd.DataFrame(softmax, columns=["Softmax"])
softmax.index.name = "Arm"
pprint(softmax)
