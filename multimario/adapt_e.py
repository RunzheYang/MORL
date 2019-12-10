## multi-obejcetive super mario bros
## created by Runzhe Yang on Jan. 21, 2019

import gym
import os
import random
import argparse
from itertools import chain

import numpy as np

import torch.nn.functional as F
import torch.nn as nn
import torch
import time
from datetime import datetime

from model import *

import torch.optim as optim
from torch.multiprocessing import Pipe, Process

from collections import deque

from tensorboardX import SummaryWriter
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from env import MoMarioEnv
from agent import EnveMoActorAgent

parser = argparse.ArgumentParser(description='MORL')

# set envrioment id and directory
parser.add_argument('--env-id', default='SuperMarioBros-v2', metavar='ENVID',
                    help='Super Mario Bros Game 1-2 (Skip Frame) v0-v3')
parser.add_argument('--name', default='e3c', metavar='name',
                    help='specify the model name')
parser.add_argument('--logdir', default='logs/', metavar='LOG',
                    help='path for recording training informtion')
parser.add_argument('--prev-model', default='SuperMarioBros-v3_2018-11-24.model', metavar='PREV',
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
                    help='number of wokers (defualt 1 for adaptation)')
parser.add_argument('--episode-limit', type=int, default=100, metavar='EL',
                    help='upper bound for the number of episodes to adapte the preference')

# hyperparameters
parser.add_argument('--lam', type=float, default=0.95, metavar='LAM',
                    help='lambda for gae (default 0.95)')
parser.add_argument('--beta', type=float, default=0.95, metavar='LAM',
                    help='beta for balancing l1 and l2 loss')
parser.add_argument('--num-step', type=int, default=50, metavar='NSTEP',
                    help='number of gae steps (default 5)')
parser.add_argument('--max-step', type=int, default=1.15e8, metavar='MSTEP',
                    help='max number of steps for learning rate scheduling (default 1.15e8)')
parser.add_argument('--T', type=float, default=1.0, metavar='TEMP',
                    help='softmax with tempreture to encorage exploration')
parser.add_argument('--learning-rate', type=float, default=2.5e-4, metavar='LR',
                    help='initial learning rate (default 2.5e-4)')
parser.add_argument('--enve-start', type=int, default=1e5, metavar='ESTART',
                    help='minimum number of naive traning before envelope')
parser.add_argument('--lr-schedule', action='store_true',
                    help='enable learning rate scheduling')
parser.add_argument('--entropy-coef', type=float, default=0.02, metavar='ENTROPY',
                    help='entropy coefficient for regurization (default 0.02)')
parser.add_argument('--gamma', type=float, default=1.00, metavar='GAMMA',
                    help='gamma for discounted rewards (default 1.00)')
parser.add_argument('--clip-grad-norm', type=float, default=0.5, metavar='CLIP',
                    help='gradient norm clipping (default 0.5)')
parser.add_argument('--reward-scale', type=float, default=1.0, metavar='RS',
                    help='reward scaling (default 1.0)')
parser.add_argument('--sample-size', type=int, default=1, metavar='SS',
                    help='number of preference samples for updating (default 1 for adaptation)')

# x_pos, time, death, coin, enermy
UNKNOWN_PREFERENCE = np.array([1.00, 0.00, 0.00, 0.00, 1.00])

def make_train_data(num_step, reward):
    discounted_return = np.empty([num_step])
    # Discounted Return
    running_add = 0
    for t in range(num_step - 1, -1, -1):
        running_add += reward[t]
        discounted_return[t] = running_add
    return discounted_return[0]


def generate_w(num_prefence, pref_param, fixed_w=None):
    if fixed_w is not None and num_prefence>1:
        sigmas = torch.Tensor([0.01]*len(pref_param))
        w = torch.distributions.normal.Normal(torch.Tensor(pref_param), sigmas)
        w = w.sample(torch.Size((num_prefence-1,))).numpy()
        w = np.abs(w) / np.linalg.norm(w, ord=1, axis=1).reshape(num_prefence-1, 1)
        return np.concatenate(([fixed_w], w))
    elif fixed_w is not None and num_prefence==1:
        return np.array([fixed_w])
    else:
        sigmas = torch.Tensor([0.01]*len(pref_param))
        w = torch.distributions.normal.Normal(torch.Tensor(pref_param), sigmas)
        w = w.sample(torch.Size((num_prefence,))).numpy()
        w = np.abs(w) / np.linalg.norm(w, ord=1, axis=1).reshape(num_prefence, 1)
        return w
    return w

def renew_w(preferences, dim, pref_param):
    sigmas = torch.Tensor([0.01]*len(pref_param))
    w = torch.distributions.normal.Normal(torch.Tensor(pref_param), sigmas)
    w = w.sample(torch.Size((1,))).numpy()
    w = np.abs(w) / np.linalg.norm(w, ord=1, axis=1)
    preferences[dim] = w
    return preferences

if __name__ == '__main__':

    args = parser.parse_args()

    # get enviroment information
    env = JoypadSpace(
        gym_super_mario_bros.make(args.env_id), SIMPLE_MOVEMENT)
    input_size = env.observation_space.shape
    output_size = env.action_space.n
    reward_size = 5

    env.close()

    # setup 
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    tag = "adapt"
    log_dir = os.path.join(args.logdir, '{}_{}_{}_{}'.format(
        args.env_id, args.name, current_time, tag))
    writer = SummaryWriter(log_dir)

    model_path = 'saved/{}_{}_{}.model'.format(args.env_id, 
                    args.name, current_time)
    load_model_path = 'saved/{}'.format(args.prev_model)

    agent = EnveMoActorAgent(
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

    # set agent model as eval since we won't update its weights.
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
    sample_target_rall = 0
    sample_morall = 0
    sample_step = 0
    sample_env_idx = 0
    recent_prob = deque(maxlen=10)
    
    pref_param = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    REPEAT = 10

    explore_w = generate_w(args.num_worker, pref_param)

    max_target = 0

    while True:
        
        acc_target = 0
        # best_param = 0
        rep_target = []
        rep_explore_w = []

        for _ in range(REPEAT):

            total_state, total_reward, total_target_reward, total_done, total_action, total_moreward\
                = [], [], [], [], [], []
            
            while True:
                actions = agent.get_action(states, explore_w)

                for parent_conn, action in zip(parent_conns, actions):
                    parent_conn.send(action)

                next_states, rewards, target_rewards, dones, real_dones, morewards, scores\
                    = [], [], [], [], [], [], []
                
                for parent_conn in parent_conns:
                    s, r, d, rd, mor, sc = parent_conn.recv()
                    next_states.append(s)
                    rewards.append(explore_w.dot(mor))
                    target_rewards.append(UNKNOWN_PREFERENCE.dot(mor))
                    dones.append(d)
                    real_dones.append(rd)
                    morewards.append(mor)
                    scores.append(sc)
                    # resample if done
                    # if d:
                    #     explore_w = renew_w(explore_w, cnt, pref_param)

                next_states = np.stack(next_states)
                rewards = np.hstack(rewards) * args.reward_scale
                target_rewards = np.hstack(target_rewards) * args.reward_scale
                dones = np.hstack(dones)
                real_dones = np.hstack(real_dones)
                morewards = np.stack(morewards) * args.reward_scale

                total_state.append(states)
                total_reward.append(rewards)
                total_target_reward.append(target_rewards)
                total_done.append(dones)
                total_action.append(actions)
                total_moreward.append(morewards)

                states = next_states[:, :, :, :]

                sample_rall += rewards[sample_env_idx]
                sample_target_rall += target_rewards[sample_env_idx]
                sample_morall = sample_morall + morewards[sample_env_idx]
                sample_step += 1
                if real_dones[sample_env_idx]:
                    sample_episode += 1
                    writer.add_scalar('data/reward', sample_rall, sample_episode)
                    writer.add_scalar('data/target_rewards', sample_target_rall, sample_episode)
                    writer.add_scalar('data/step', sample_step, sample_episode)
                    writer.add_scalar('data/score', scores[sample_env_idx], sample_episode)
                    writer.add_scalar('data/x_pos_reward', sample_morall[0], sample_episode)
                    writer.add_scalar('data/time_penalty', sample_morall[1], sample_episode)
                    writer.add_scalar('data/death_penalty', sample_morall[2], sample_episode)
                    writer.add_scalar('data/coin_reward', sample_morall[3], sample_episode)
                    writer.add_scalar('data/enemy_reward', sample_morall[4], sample_episode)
                    sample_rall = 0
                    sample_target_rall = 0
                    sample_step = 0
                    sample_morall = 0
                    break

            # [w1, w1, w1, w2, w2, w2, w3, w3, w3...]
            # [s1, s2, s3, s1, s2, s3, s1, s2, s3...]
            # expand w batch
            real_w = np.array([UNKNOWN_PREFERENCE]*args.sample_size)
            real_w = real_w.repeat(len(total_state)*args.num_worker, axis=0)
            
            # calculate utility from reward vectors
            total_moreward = np.array(total_moreward).transpose([1, 0, 2]).reshape([-1, reward_size])
            total_moreward = np.tile(total_moreward, (args.sample_size, 1))
            # total_utility = np.sum(total_moreward * update_w, axis=-1).reshape([-1])
            total_target_utility = np.sum(total_moreward * real_w, axis=-1).reshape([-1])

            total_target = []

            num_step = len(total_done)

            for idw in range(args.sample_size):
                ofs = args.num_worker * num_step
                for idx in range(args.num_worker):
                    target = make_train_data(num_step,
                                  total_target_utility[idx*num_step+idw*ofs : (idx+1)*num_step+idw*ofs])
                    target = scores[0]
                    total_target.append(target)

            acc_target += target
            rep_target.append(target)
            rep_explore_w.append(explore_w)

        acc_target = acc_target/REPEAT

        print("avg performance", acc_target)

        writer.add_scalar('data/avg_target_rewards', acc_target, sample_episode)

        if acc_target > max_target:
            max_target = acc_target
            best_param = pref_param

        pref_param = agent.find_preference(
            np.stack(rep_explore_w),
            np.hstack(rep_target),
            pref_param)

        explore_w = renew_w(explore_w, 0, pref_param)

        if sample_episode >= args.episode_limit:
            print("the best param:", best_param)
            break


