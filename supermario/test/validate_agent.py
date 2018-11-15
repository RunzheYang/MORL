from __future__ import absolute_import, division, print_function
import argparse
import numpy as np
import torch

from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from policy.morlpolicy import MetaAgent
from policy.model import get_new_model
from utils.rescale import rescale

from tensorboardX import SummaryWriter
from datetime import datetime
import socket

import random
import torch.multiprocessing as mp

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

def run_one_episode(args, probe, exp):
    from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    import gym_super_mario_bros
    env = gym_super_mario_bros.make(args.env_name)
    env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)

    # get state / action / reward sizes
    state_size = torch.Tensor(env.observation_space.high).size() 
    state_size = torch.Size(
                    [60 * args.nframe, 
                     64, 
                     1])
    action_size = env.action_space.n
    reward_size = 5

    # generate an agent for validating
    model = torch.load("{}{}.pkl".format(args.save,
                                             "m.{}_{}_n.{}_tmp".format(args.method, args.model, args.name)))
    optimizer = torch.load("{}{}_opt.pkl".format(args.save,
                                         "m.{}_{}_n.{}_tmp".format(args.method, args.model, args.name)))

    args_new = args
    args_new.eps = 0.05
    
    agent = MetaAgent(model, args_new, is_train=True) # is_train is true for adding epsilon noise
    
    terminal = False
    loss = 0
    cnt = 0
    utility = 0
    score = 0
    acc_pred_q = 0
    acc_reward = np.zeros(5)

    state = rescale(env.reset())

    history_f = [state] * args.nframe
    state = np.array(history_f).reshape(-1, state.shape[1], state.shape[2])

    pred_q = agent.predict(state, probe)

    while not terminal:
        if args.single:
            action = agent.act(state, preference=probe)
        else:
            action = agent.act(state)

        next_state, score, terminal, info = env.step(action)
        next_state = rescale(next_state)

        history_f[0] = 0
        for i in range(args.nframe-1):
            history_f[i] = history_f[i+1]
        history_f[args.nframe-1] = next_state
        next_state = np.array(history_f).reshape(-1, next_state.shape[1], next_state.shape[2])

        _reward =info['rewards']
        div = [10.0, 0.1, 160.0, 10.0, 0.1]
        reward = np.array([_reward[i] / div[i] for i in range(5)])
        score = info['score']
        if info['flag_get'] or cnt > 2000:
            terminal = True

        state = next_state

        utility = utility + (probe.cpu().numpy().dot(reward)) * np.power(args.gamma, cnt)
        acc_reward = acc_reward + reward
        cnt = cnt + 1
    
    exp.send((acc_reward, score, utility, float(pred_q[0])))
    env.close()
    del gym_super_mario_bros
    del env
    exp.close()


def validate(args, writer, probe, num_eps):

    REPEAT = 8

    print("start validating...")        
    
    acc_acc_reward = np.zeros(5)
    acc_utility = 0
    acc_score = 0
    acc_pred_q = 0

    for num_eps_sub in range(REPEAT):
        random.seed()
        exp_recv, exp_send = mp.Pipe()
        p = mp.Process(target=run_one_episode, args=(args, probe, exp_send,))
        p.start()
        acc_reward, score, utility, pred_q = exp_recv.recv()
        p.join()

        acc_acc_reward = acc_acc_reward + acc_reward
        acc_score = acc_score + score
        acc_utility = acc_utility + utility
        acc_pred_q = acc_pred_q + pred_q

    acc_acc_reward = acc_acc_reward * 1.0 / REPEAT
    acc_score = acc_score * 1.0 / REPEAT
    acc_utility = acc_utility * 1.0 / REPEAT
    acc_pred_q = acc_pred_q * 1.0 / REPEAT

    writer.add_scalars('train/rewards', {
        "x_pos": acc_acc_reward[0],
        "enemy": acc_acc_reward[1],
        "time": acc_acc_reward[2],
        "death": acc_acc_reward[3],
        "coin": acc_acc_reward[4],
        }, num_eps)

    writer.add_scalars('train/utility', {
        "real utility": acc_utility,
        "predicted utility": acc_pred_q,
        }, num_eps)

    writer.add_scalar('train/score', acc_score, num_eps)
