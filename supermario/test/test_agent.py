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
parser.add_argument('--env-name', default='SuperMarioBros-v0', metavar='ENVNAME',
                    help='Super Mario Bros Game 1-2 (Skip Frame) v0-v3 ')
parser.add_argument('--method', default='naive', metavar='METHODS',
                    help='methods: naive | envelope')
parser.add_argument('--model', default='test', metavar='MODELS',
                    help='linear | cnn | cnn + lstm')
parser.add_argument('--gamma', type=float, default=0.99, metavar='GAMMA',
                    help='gamma for infinite horizonal MDPs')
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
parser.add_argument('--episode-num', type=int, default=5, metavar='EN',
                    help='number of episodes for training')
parser.add_argument('--optimizer', default='Adam', metavar='OPT',
                    help='optimizer: Adam | RMSprop')
parser.add_argument('--update-freq', type=int, default=10, metavar='OPT',
                    help='optimizer: Adam | RMSprop')
parser.add_argument('--beta', type=float, default=0.01, metavar='BETA',
                    help='(initial) beta for evelope algorithm, default = 0.01')
parser.add_argument('--homotopy', default=False, action='store_true',
                    help='use homotopy optimization method')
# LOG & SAVING
parser.add_argument('--serialize', default=True, action='store_true',
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


def test(env, agent, args):
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(
                args.log, current_time + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir)
    print("start testing...")        
    env.reset()
    for num_eps in range(args.episode_num):
        terminal = False
        loss = 0
        cnt = 0
        utility = 0
        score = 0
        acc_reward = np.zeros(5)

        probe = FloatTensor([1.0, 0.0, 0.0, 0.0, 0.0])
        state = env.reset()
        state = np.array(state)

        while not terminal:
            print("frame", cnt)
            action = agent.act(state, preference=probe)
            next_state, score, terminal, info = env.step(action)

            env.render()

            next_state = np.array(next_state)
            reward = np.array(info['rewards'])
            score = info['score']
            print("action", action)
            print("reward", reward, "\n")

            agent.memorize(state, action, next_state, reward, terminal)
            
            # if cnt % 10 == 0: loss += agent.learn()
            
            if cnt > 1000:
                terminal = True
                agent.reset()
            utility = utility + (probe.cpu().numpy().dot(reward)) * np.power(args.gamma, cnt)
            acc_reward = acc_reward + reward
            cnt = cnt + 1
        

        writer.add_scalar('test/loss', loss / (cnt / 10), num_eps)
        writer.add_scalars('test/rewards', {
            "x_pos": acc_reward[0],
            "enermy": acc_reward[1],
            "time": acc_reward[2],
            "death": acc_reward[3],
            "coin": acc_reward[4],
            }, num_eps)
        writer.add_scalar('test/score', score, num_eps)

        print("end of eps %d with utility %0.2f loss: %0.4f" % (
            num_eps,
            utility,
            loss / (cnt / 10)))
    
    env.close()
    writer.close()


if __name__ == '__main__':
    args = parser.parse_args()

    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
    # reward type (X_POSITION, ENERMY, TIME, DEATH, COIN)

    # get state / action / reward sizes
    state_size = torch.Tensor(env.observation_space.high).size()
    action_size = env.action_space.n
    reward_size = 5


    # generate an agent for testing
    if args.serialize:
        model = torch.load("{}{}.pkl".format(args.save,
                                             "m.{}_{}_n.{}".format(args.method, args.model, args.name)))
    else:
        model = get_new_model(args.method, args.model, state_size, action_size, reward_size)
    
    agent = MetaAgent(model, args, is_train=False)

    test(env, agent, args)
