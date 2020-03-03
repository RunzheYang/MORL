## multi-obejcetive super mario bros
## modified by Runzhe Yang on Dec. 18, 2018

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
from agent import EnveDoubleMoActorAgent

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
                    help='adds standardization of weight preferences (default FALSE)')
parser.add_argument('--num-worker', type=int, default=16, metavar='NWORKER',
                    help='number of wokers (defualt 16)')

# hyperparameters
parser.add_argument('--lam', type=float, default=0.95, metavar='LAM',
                    help='lambda for gae (default 0.95)')
parser.add_argument('--beta', type=float, default=0.95, metavar='LAM',
                    help='beta for balancing l1 and l2 loss')
parser.add_argument('--T', type=float, default=10, metavar='TEMP',
                    help='softmax with tempreture to encorage exploration')
parser.add_argument('--num-step', type=int, default=5, metavar='NSTEP',
                    help='number of gae steps (default 5)')
parser.add_argument('--max-step', type=int, default=1.15e8, metavar='MSTEP',
                    help='max number of steps for learning rate scheduling (default 1.15e8)')
parser.add_argument('--learning-rate', type=float, default=2.5e-4, metavar='LR',
                    help='initial learning rate (default 2.5e-4)')
parser.add_argument('--lr-schedule', action='store_true',
                    help='enable learning rate scheduling')
parser.add_argument('--enve-start', type=int, default=1e5, metavar='ESTART',
                    help='minimum number of naive traning before envelope')
parser.add_argument('--update-target-critic', type=int, default=1e4, metavar='UTC',
                    help='the number of steps to update target critic')
parser.add_argument('--entropy-coef', type=float, default=0.02, metavar='ENTROPY',
                    help='entropy coefficient for regurization (default 0.2)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='GAMMA',
                    help='gamma for discounted rewards (default 0.99)')
parser.add_argument('--clip-grad-norm', type=float, default=0.5, metavar='CLIP',
                    help='gradient norm clipping (default 0.5)')
parser.add_argument('--reward-scale', type=float, default=1.0, metavar='RS',
                    help='reward scaling (default 1.0)')
parser.add_argument('--sample-size', type=int, default=8, metavar='SS',
                    help='number of preference samples for updating')

def make_train_data(args, reward, done, value, next_value, reward_size):
    discounted_return = np.empty([args.num_step, reward_size])
    
    # Discounted Return
    if args.use_gae:
        gae = np.zeros(reward_size)
        for t in range(args.num_step - 1, -1, -1):
            delta = reward[t] + args.gamma * \
                next_value[t] * (1 - done[t]) - value[t]
            gae = delta + args.gamma * args.lam * (1 - done[t]) * gae

            discounted_return[t] = gae + value[t]

    else:
        running_add = next_value[-1]
        for t in range(args.num_step - 1, -1, -1):
            running_add = reward[t] + args.gamma * running_add * (1 - done[t])
            discounted_return[t] = running_add

    return discounted_return


def envelope_operator(args, preference, target, value, reward_size, g_step):
    
    # [w1, w1, w1, w1, w1, w1, w2, w2, w2, w2, w2, w2...]
    # [s1, s2, s3, u1, u2, u3, s1, s2, s3, u1, u2, u3...]

    # weak envelope calculation
    ofs = args.num_worker * args.num_step
    target = np.concatenate(target).reshape(-1, reward_size)
    if g_step > args.enve_start:
        prod = np.inner(target, preference)
        envemask = prod.transpose().reshape(args.sample_size, -1, ofs).argmax(axis=1)
        envemask = envemask.reshape(-1) * ofs + np.array(list(range(ofs))*args.sample_size)
        target = target[envemask]
    # For Actor
    adv = target - value

    return target, adv


def generate_w(num_prefence, reward_size, fixed_w=None):
    if fixed_w is not None:
        w = np.random.randn(num_prefence-1, reward_size)
        # normalize as a simplex
        w = np.abs(w) / np.linalg.norm(w, ord=1, axis=1).reshape(num_prefence-1, 1)
        return np.concatenate(([fixed_w], w))
    else:
        w = np.random.randn(num_prefence-1, reward_size)
        w = np.abs(w) / np.linalg.norm(w, ord=1, axis=1).reshape(num_prefence, 1)
        return w

def renew_w(preferences, dim):
    w = np.random.randn(reward_size)
    w = np.abs(w) / np.linalg.norm(w, ord=1, axis=0)
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
    tag = ["test", "train"][int(args.training)]
    log_dir = os.path.join(args.logdir, '{}_{}_{}_{}'.format(
        args.env_id, args.name, current_time, tag))
    writer = SummaryWriter(log_dir)

    model_path = 'saved/{}_{}_{}.model'.format(args.env_id, 
                    args.name, current_time)
    load_model_path = 'saved/{}'.format(args.prev_model)

    agent = EnveDoubleMoActorAgent(
        args,
        input_size,
        output_size,
        reward_size)

    if args.load_model:
        if args.use_cuda:
            agent.model.load_state_dict(torch.load(load_model_path))
            agent.model_ = copy.deepcopy(agent.model)
        else:
            agent.model.load_state_dict(
                torch.load(
                    load_model_path,
                    map_location='cpu'))
            agent.model_ = copy.deepcopy(agent.model)

    if not args.training:
        agent.model.eval()
        agent.model_.eval()

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

    fixed_w = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
    # fixed_w = np.array([0.00, 0.00, 0.00, 1.00, 0.00])
    # fixed_w = np.array([0.00, 0.00, 0.00, 0.00, 1.00])
    # fixed_w = np.array([1.00, 0.00, 0.00, 0.00, 0.00])
    # fixed_w = np.array([0.00, 1.00, 0.00, 0.00, 0.00])
    # fixed_w = np.array([0.00, 0.00, 1.00, 0.00, 0.00])
    explore_w = generate_w(args.num_worker, reward_size, fixed_w)

    while True:
        total_state, total_reward, total_done, total_next_state, total_action, total_moreward = [], [], [], [], [], []
        global_step += (args.num_worker * args.num_step)

        for _ in range(args.num_step):
            if not args.training:
                time.sleep(0.05)
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
                cnt += 1

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
                agent.anneal()
                writer.add_scalar('data/reward', sample_rall, sample_episode)
                writer.add_scalar('data/step', sample_step, sample_episode)
                writer.add_scalar('data/score', scores[sample_env_idx], sample_episode)
                writer.add_scalar('data/x_pos_reward', sample_morall[0], sample_episode)
                writer.add_scalar('data/time_penalty', sample_morall[1], sample_episode)
                writer.add_scalar('data/death_penalty', sample_morall[2], sample_episode)
                writer.add_scalar('data/coin_reward', sample_morall[3], sample_episode)
                writer.add_scalar('data/enemy_reward', sample_morall[4], sample_episode)
                writer.add_scalar('data/tempreture', agent.T, sample_episode)
                sample_rall = 0
                sample_step = 0
                sample_morall = 0

        if args.training:
            # [w1, w1, w1, w1, w1, w1, w2, w2, w2, w2, w2, w2...]
            # [s1, s2, s3, u1, u2, u3, s1, s2, s3, u1, u2, u3...]
            # expand w batch
            update_w = generate_w(args.sample_size, reward_size, fixed_w)
            total_update_w = update_w.repeat(args.num_step*args.num_worker, axis=0)
            # expand state batch
            # WRONG!!! total_state = total_state * args.sample_size
            total_state = np.stack(total_state).transpose(
                [1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
            total_state = np.tile(total_state, (args.sample_size, 1, 1, 1))
            # expand next_state batch
            # WRONG!!! total_next_state = total_next_state * args.sample_size
            total_next_state = np.stack(total_next_state).transpose(
                [1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
            total_next_state = np.tile(total_next_state, (args.sample_size, 1, 1, 1))
            # calculate utility from reward vectors
            total_moreward = np.array(total_moreward).transpose([1, 0, 2]).reshape([-1, reward_size])
            total_moreward = np.tile(total_moreward, (args.sample_size, 1))
            # total_utility here is defined for debugging purporses. See
            # https://github.com/RunzheYang/MORL/issues/12
            # total_utility = np.sum(total_moreward * total_update_w, axis=-1).reshape([-1])
            # expand action batch
            total_action = np.stack(total_action).transpose().reshape([-1])
            total_action = np.tile(total_action, args.sample_size)
            # expand done batch
            total_done = np.stack(total_done).transpose().reshape([-1])
            total_done = np.tile(total_done, args.sample_size)

            value, next_value, policy = agent.forward_transition(
                total_state, total_next_state, total_update_w)

            # logging utput to see how convergent it is.
            policy = policy.detach()
            m = F.softmax(policy, dim=-1)
            recent_prob.append(m.max(1)[0].mean().cpu().numpy())
            writer.add_scalar(
                'data/max_prob',
                np.mean(recent_prob),
                sample_episode)

            total_target = []
            total_adv = []
            for idw in range(args.sample_size):
                ofs = args.num_worker * args.num_step
                for idx in range(args.num_worker):
                    target = make_train_data(args,
                                  total_moreward[idx*args.num_step+idw*ofs : (idx+1)*args.num_step+idw*ofs],
                                  total_done[idx*args.num_step+idw*ofs: (idx+1)*args.num_step+idw*ofs],
                                  value[idx*args.num_step+idw*ofs : (idx+1)*args.num_step+idw*ofs],
                                  next_value[idx*args.num_step+idw*ofs : (idx+1)*args.num_step+idw*ofs],
                                  reward_size)
                    total_target.append(target)

            total_target, total_adv = envelope_operator(args, update_w, total_target, value, reward_size, global_step)

            agent.train_model(
                total_state,
                total_next_state,
                total_update_w,
                total_target,
                total_action,
                total_adv)

            # adjust learning rate
            if args.lr_schedule:
                new_learing_rate = args.learning_rate - \
                    (global_step / args.max_step) * args.learning_rate
                for param_group in agent.optimizer.param_groups:
                    param_group['lr'] = new_learing_rate
                    writer.add_scalar(
                        'data/lr', new_learing_rate, sample_episode)

            if global_step % (args.num_worker * args.num_step * 100) == 0:
                torch.save(agent.model.state_dict(), model_path)

            if global_step % args.update_target_critic == 0:
                agent.sync()
