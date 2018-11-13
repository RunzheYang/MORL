# reward type (X_POSITION, ENERMY, TIME, DEATH, COIN)

from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import torch
import random

env = gym_super_mario_bros.make('SuperMarioBros2-v1')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)

random.seed(1)

for eps in range(50):
	done = True
	for step in range(2000):
	    if done:
	        state = env.reset()
	    act = env.action_space.sample()
	    state, reward, done, info = env.step(act)
	    print(info)	
	    env.render()
env.close()