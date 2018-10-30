from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

env = gym_super_mario_bros.make('SuperMarioBros-v1')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)

# reward type (X_POSITION, ENERMY, TIME, DEATH, COIN)
import torch

a = torch.Tensor(env.observation_space.high)

done = True
for step in range(2000):
    if done:
        state = env.reset()
    print("state", state)
    act = env.action_space.sample()
    state, reward, done, info = env.step(act)
    print(info['rewards'])
    env.render()

env.close()