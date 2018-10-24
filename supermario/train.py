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

from tensorboardX import SummaryWriter
from datetime import datetime
import socket

parser = argparse.ArgumentParser(description='MORL')
# CONFIG
parser.add_argument('--env-name', default='SuperMarioBros-v2', metavar='ENVNAME',
                    help='Super Mario Bros Game 1-2 (Skip Frame) v0-v3 ')
parser.add_argument('--method', default='naive', metavar='METHODS',
                    help='methods: naive | envelope')
parser.add_argument('--model', default='test', metavar='MODELS',
                    help='linear | cnn | lstmcnn')
parser.add_argument('--gamma', type=float, default=0.99, metavar='GAMMA',
                    help='gamma for infinite horizonal MDPs')
parser.add_argument('--nframe', type=int, default=4, metavar='NFRAME',
                    help='number of frames in one state')

# TRAINING
parser.add_argument('--mem-size', type=int, default=4000, metavar='M',
                    help='max size of the replay memory')
parser.add_argument('--batch-size', type=int, default=20, metavar='B',
                    help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate')
parser.add_argument('--epsilon', type=float, default=0.5, metavar='EPS',
                    help='epsilon greedy exploration')
parser.add_argument('--epsilon-decay', default=False, action='store_true',
                    help='linear epsilon decay to zero')
parser.add_argument('--weight-num', type=int, default=4, metavar='WN',
                    help='number of sampled weights per iteration')
parser.add_argument('--episode-num', type=int, default=100, metavar='EN',
                    help='number of episodes for training')
parser.add_argument('--optimizer', default='Adam', metavar='OPT',
                    help='optimizer: Adam | RMSprop')
parser.add_argument('--priority', default=False, action='store_true',
                    help='using prioritized experience replay')
parser.add_argument('--update-freq', type=int, default=100, metavar='OPT',
                    help='update frequency for double Q learning')
parser.add_argument('--homotopy', default=False, action='store_true',
                    help='use homotopy optimization method')
parser.add_argument('--beta', type=float, default=0.01, metavar='BETA',
                    help='(initial) beta for evelope algorithm using homotopy, default = 0.01')

# Special
parser.add_argument('--single', default=False, action='store_true',
                    help='single objective reinforcement learning, remember to set weight-num as 1')

# LOG & SAVING
parser.add_argument('--serialize', default=False, action='store_true',
                    help='serialize a model')
parser.add_argument('--save', default='saved/', metavar='SAVE',
                    help='path for saving trained models')
parser.add_argument('--name', default='test', metavar='name',
                    help='specify a name for saving the model')
parser.add_argument('--log', default='logs/', metavar='LOG',
                    help='path for recording training informtion')


use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


def train(env, agent, args):
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(
                args.log, current_time + '_' + args.name)
    writer = SummaryWriter(log_dir)
    print("start training...")        
    env.reset()
    for num_eps in range(args.episode_num):
        terminal = False
        loss = 0
        cnt = 0
        utility = 0
        score = 0
        acc_reward = np.zeros(5)

        probe = FloatTensor([0.6, 0.1, 0.1, 0.1, 0.1])
        state = env.reset()
    
        history_f = [state] * args.nframe
        state = np.array(history_f).reshape(-1, state.shape[1], state.shape[2])

        while not terminal:
            print("frame", num_eps, cnt)

            if args.single:
                action = agent.act(state, preference=probe)
            else:
                action = agent.act(state)

            next_state, score, terminal, info = env.step(action)

            history_f[0] = 0
            for i in range(args.nframe-1):
                history_f[i] = history_f[i+1]
            history_f[args.nframe-1] = next_state
            next_state = np.array(history_f).reshape(-1, next_state.shape[1], next_state.shape[2])

            _reward =info['rewards']
            div = [10.0, 0.1, 10.0, 10.0, 0.1]
            reward = np.array([_reward[i] / div[i] for i in range(5)])
            score = info['score']
            if info['flag_get']: 
                terminal = True
            print("action", action)
            print("reward", reward, "\n")

            agent.memorize(state, action, next_state, reward, terminal)

            state = next_state
            
            if args.single:
                # single objective learning
                loss += agent.learn(probe) 
            else:
                # multi-objective learning
                loss += agent.learn()

            if cnt > 2000:
                terminal = True
                agent.reset()

            utility = utility + (probe.cpu().numpy().dot(reward)) * np.power(args.gamma, cnt)
            acc_reward = acc_reward + reward
            cnt = cnt + 1
        
        writer.add_scalar('train/loss', loss / (cnt / 10), num_eps)
        writer.add_scalars('train/rewards', {
            "x_pos": acc_reward[0],
            "enermy": acc_reward[1],
            "time": acc_reward[2],
            "death": acc_reward[3],
            "coin": acc_reward[4],
            }, num_eps)
        writer.add_scalar('train/score', score, num_eps)

        print("end of eps %d with utility %0.2f loss: %0.4f" % (
            num_eps,
            utility,
            loss / (cnt / 10)))

        if (num_eps + 1) % 10 == 0:
            agent.save(args.save, "m.{}_{}_n.{}_tmp".format(
                args.method, args.model, args.name))
    
    env.close()
    writer.close()
    agent.save(args.save, "m.{}_{}_n.{}".format(args.method, args.model, args.name))


if __name__ == '__main__':
    args = parser.parse_args()

    env = gym_super_mario_bros.make(args.env_name)
    env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
    # reward type (X_POSITION, ENERMY, TIME, DEATH, COIN)

    # get state / action / reward sizes
    state_size = torch.Tensor(env.observation_space.high).size() 
    state_size = torch.Size(
                    [state_size[0] * args.nframe, 
                     state_size[1], 
                     state_size[2]])
    action_size = env.action_space.n
    reward_size = 5


    # generate an agent for training
    if args.serialize:
        model = torch.load("{}{}.pkl".format(args.save,
                                             "m.{}_{}_n.{}".format(args.method, args.model, args.name)))
    else:
        model = get_new_model(args.method, args.model, state_size, action_size, reward_size)
    
    agent = MetaAgent(model, args, is_train=True)

    train(env, agent, args)
