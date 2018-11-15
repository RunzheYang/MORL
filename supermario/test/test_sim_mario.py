# reward type (X_POSITION, ENERMY, TIME, DEATH, COIN)

import torch
import numpy as np
import random
from multiprocessing import Process, Pipe

def run_one_episode(conn):
    from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    import gym_super_mario_bros
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
    state = env.reset()
    for step in range(2000):
        # act = env.action_space.sample()
        act = int(np.random.choice(env.action_space.n, 1)[0])
        if step == 1: print(act)
        state, reward, done, info = env.step(act)
        print(info)
        env.render()
        print 
        if done:
            break
    print("end of the epsiode {}".format(eps))
    conn.send(info)
    env.close()
    del gym_super_mario_bros
    del env
    conn.close()

for eps in range(50):
    random.seed()
    print(np.random.random())
    parent_conn, child_conn = Pipe()
    p = Process(target=run_one_episode, args=(child_conn,))
    p.start()
    print("receive:", parent_conn.recv())
    p.join()