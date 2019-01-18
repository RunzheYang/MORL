## multi-obejcetive super mario bros
## modified by Runzhe Yang on Dec. 8, 2018

import gym
import os
import random
import argparse
from itertools import chain

import numpy as np
import codecs, json

import torch.nn.functional as F
import torch.nn as nn
import torch
import time
from datetime import datetime

from model import *

import torch.optim as optim
from torch.multiprocessing import Pipe, Process

from collections import deque

import gym_super_mario_bros
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from env import MoMarioEnv
from agent import NaiveMoActorAgent

parser = argparse.ArgumentParser(description='MORL')

# set envrioment id and directory
parser.add_argument('--env-id', default='SuperMarioBros-v2', metavar='ENVID',
                    help='Super Mario Bros Game 1-2 (Skip Frame) v0-v3')
parser.add_argument('--name', default='n3c', metavar='name',
                    help='specify the model name')
parser.add_argument('--logdir', default='logs/', metavar='LOG',
                    help='path for recording training informtion')
parser.add_argument('--prev-model', default='SuperMarioBros-v2_n3c_Dec15_03-40-48.model', metavar='PREV',
                    help='specify the full name of the previous model')

# running configuration
parser.add_argument('--use-cuda', action='store_true',
                    help='use cuda (default FALSE)')
parser.add_argument('--use-gae', action='store_true',
                    help='use gae (defualt FALSE)')
parser.add_argument('--life-done', action='store_true',
                    help='terminate when die')
parser.add_argument('--single-stage', action='store_true',
                    help='only train on one stage ')
parser.add_argument('--load-model', action='store_true',
                    help='load previous model (default FALSE)')
parser.add_argument('--training', action='store_true',
                    help='run for training (default FALSE)')
parser.add_argument('--render', action='store_true',
                    help='render the game (default FALSE)')
parser.add_argument('--standardization', action='store_true',
                    help='load previous model (default FALSE)')
parser.add_argument('--num-worker', type=int, default=1, metavar='NWORKER',
                    help='number of wokers (defualt 1)')

# hyperparameters
parser.add_argument('--lam', type=float, default=0.95, metavar='LAM',
                    help='lambda for gae (default 0.95)')
parser.add_argument('--num-step', type=int, default=5, metavar='NSTEP',
                    help='number of gae steps (default 5)')
parser.add_argument('--max-step', type=int, default=1.15e8, metavar='MSTEP',
                    help='max number of steps for learning rate scheduling (default 1.15e8)')
parser.add_argument('--learning-rate', type=float, default=2.5e-4, metavar='LR',
                    help='initial learning rate (default 2.5e-4)')
parser.add_argument('--lr-schedule', action='store_true',
                    help='enable learning rate scheduling')
parser.add_argument('--entropy-coef', type=float, default=0.02, metavar='ENTROPY',
                    help='entropy coefficient for regurization (default 0.02)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='GAMMA',
                    help='gamma for discounted rewards (default 0.99)')
parser.add_argument('--clip-grad-norm', type=float, default=0.5, metavar='CLIP',
                    help='gradient norm clipping (default 0.5)')
parser.add_argument('--reward-scale', type=float, default=1.0, metavar='RS',
                    help='reward scaling (default 1.0)')
parser.add_argument('--sample-size', type=int, default=1, metavar='SS',
                    help='number of preference samples for updating')

def make_train_data(args, reward, done, value, next_value):
    discounted_return = np.empty([args.num_step])

    # Discounted Return
    if args.use_gae:
        gae = 0
        for t in range(args.num_step - 1, -1, -1):
            delta = reward[t] + args.gamma * \
                next_value[t] * (1 - done[t]) - value[t]
            gae = delta + args.gamma * args.lam * (1 - done[t]) * gae

            discounted_return[t] = gae + value[t]

        # For Actor
        adv = discounted_return - value

    else:
        running_add = next_value[-1]
        for t in range(args.num_step - 1, -1, -1):
            running_add = reward[t] + args.gamma * running_add * (1 - done[t])
            discounted_return[t] = running_add

        # For Actor
        adv = discounted_return - value

    return discounted_return, adv


def generate_w(num_prefence, reward_size, fixed_w=None):
    if fixed_w is not None:
        w = np.random.randn(num_prefence-1, reward_size)
        # normalize as a simplex
        w = np.abs(w) / np.linalg.norm(w, ord=1, axis=1).reshape(num_prefence-1, 1)
        return np.concatenate(([fixed_w], w))
    else:
        w = np.random.randn(num_prefence, reward_size)
        w = np.abs(w) / np.linalg.norm(w, ord=1, axis=1).reshape(num_prefence, 1)
        return w

def renew_w(preferences, dim):
    w = np.random.randn(reward_size)
    w = np.abs(w) / np.linalg.norm(w, ord=1, axis=1).reshape(num_prefence-1, 1)
    preferences[dim] = w
    return preferences


if __name__ == '__main__':

    args = parser.parse_args()

    # get enviroment information
    env = BinarySpaceToDiscreteSpaceEnv(
        gym_super_mario_bros.make(args.env_id), SIMPLE_MOVEMENT)
    input_size = env.observation_space.shape
    output_size = env.action_space.n
    reward_size = 5

    env.close()

    # setup 
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    tag = ["test", "train"][int(args.training)]
    log_dir = os.path.join(args.logdir, '{}_{}_{}_{}'.format(
        args.env_id, args.name, current_time, tag))

    model_path = 'saved/{}_{}_{}.model'.format(args.env_id, 
                    args.name, current_time)
    load_model_path = 'saved/{}'.format(args.prev_model)

    agent = NaiveMoActorAgent(
        args,
        input_size,
        output_size,
        reward_size)

    if args.load_model:
        if args.use_cuda:
            agent.model.load_state_dict(torch.load(load_model_path))
        else:
            agent.model.load_state_dict(
                torch.load(
                    load_model_path,
                    map_location='cpu'))

    if not args.training:
        agent.model.eval()

    works = []
    parent_conns = []
    child_conns = []
    for idx in range(args.num_worker):
        parent_conn, child_conn = Pipe()
        work = MoMarioEnv(args, idx, child_conn)
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    states = np.zeros([args.num_worker, 4, 84, 84])

    sample_episode = 0
    sample_rall = 0
    sample_morall = 0
    sample_step = 0
    sample_env_idx = 0
    global_step = 0
    recent_prob = deque(maxlen=10)

    f = open('result_n3c', 'w')
    record = {}
    record['w'] = []
    record['utility'] = []
    record['score'] = []
    record['reward'] = []

    for repeat_out in range(2000):
        fixed_w = generate_w(args.num_worker, reward_size)
        # fixed_w = np.array([0.00, 0.00, 0.00, 1.00, 0.00])
        # fixed_w = np.array([0.00, 0.00, 0.00, 0.00, 1.00])
        # fixed_w = np.array([1.00, 0.00, 0.00, 0.00, 0.00])
        # fixed_w = np.array([0.00, 1.00, 0.00, 0.00, 0.00])
        # fixed_w = np.array([0.00, 0.00, 1.00, 0.00, 0.00])
        explore_w = fixed_w
        record['w'].append(fixed_w.tolist())

        while True:
            total_state, total_reward, total_done, total_next_state, total_action, total_moreward = [], [], [], [], [], []
            global_step += (args.num_worker * args.num_step)

            if sample_episode == 10:
                sample_episode = 0
                break

            for _ in range(args.num_step):
                actions = agent.get_action(states, explore_w)

                for parent_conn, action in zip(parent_conns, actions):
                    parent_conn.send(action)

                next_states, rewards, dones, real_dones, morewards, scores = [], [], [], [], [], []
                cnt = 0
                for parent_conn in parent_conns:
                    s, r, d, rd, mor, sc = parent_conn.recv()
                    next_states.append(s)
                    rewards.append(fixed_w.dot(mor))
                    dones.append(d)
                    real_dones.append(rd)
                    morewards.append(mor)
                    scores.append(sc)
                    # resample if done
                    if cnt > 0 and d:
                        explore_w = renew_w(explore_w, cnt)
                        print("renew the preference for exploration", explore_w)

                next_states = np.stack(next_states)
                rewards = np.hstack(rewards) * args.reward_scale
                dones = np.hstack(dones)
                real_dones = np.hstack(real_dones)
                morewards = np.stack(morewards) * args.reward_scale

                total_state.append(states)
                total_next_state.append(next_states)
                total_reward.append(rewards)
                total_done.append(dones)
                total_action.append(actions)
                total_moreward.append(morewards)

                states = next_states[:, :, :, :]

                sample_rall += rewards[sample_env_idx]
                sample_morall = sample_morall + morewards[sample_env_idx]
                sample_step += 1
                if real_dones[sample_env_idx]:
                    sample_episode += 1
                    record['utility'].append(sample_rall)
                    record['score'].append(scores[sample_env_idx])
                    record['reward'].append(sample_morall.tolist())
                    sample_rall = 0
                    sample_step = 0
                    sample_morall = 0
    
    json.dump(record, f)
    f.closed
