from __future__ import absolute_import, division, print_function
import argparse
import numpy as np
import torch

from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import random
import torch.multiprocessing as mp

import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from policy.morlpolicy import MetaAgent
from policy.model import get_new_model
from test.validate_agent import validate
from utils.rescale import rescale

from tensorboardX import SummaryWriter
from datetime import datetime
import socket

parser = argparse.ArgumentParser(description='MORL')
# CONFIG
parser.add_argument('--env-name', default='SuperMarioBros2-v1', metavar='ENVNAME',
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

def gain_exp(args, probe, exp, num_eps_start, delta_n=10):
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

    # generate an agent for training
    if num_eps_start > 0:
        model = torch.load("{}{}.pkl".format(args.save,
                                             "m.{}_{}_n.{}_tmp".format(args.method, args.model, args.name)))
        optimizer = torch.load("{}{}_opt.pkl".format(args.save,
                                             "m.{}_{}_n.{}_tmp".format(args.method, args.model, args.name)))
    else:
        model = get_new_model(args.method, args.model, state_size, action_size, reward_size)
        optimizer = None
    
    agent = MetaAgent(model, args, optimizer=optimizer, is_train=True)
    
    trajectory = []
    utility = 0

    for eps in range(delta_n):
        terminal = False
        cnt = 0
        score = 0
        state = rescale(env.reset())

        history_f = [state] * args.nframe
        state = np.array(history_f).reshape(-1, state.shape[1], state.shape[2])

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
            div = [10.0, 0.1, 10.0, 10.0, 0.1]
            reward = np.array([_reward[i] / div[i] for i in range(5)])
            # reward clipping
            for i in range(len(reward)):
                if reward[i] > 50.0:
                    reward[i] = 50.0
            
            score = info['score']
            if info['flag_get'] or cnt > 2000:
                terminal = True

            trajectory.append((state, action, next_state, reward, terminal))

            state = next_state

            utility = utility + (probe.cpu().numpy().dot(reward)) * np.power(args.gamma, cnt)
            cnt = cnt + 1
        
        agent.reset()
        
        print("end of the epsiode {}".format(num_eps_start+eps))
    
    exp.send(dict(trajectory=trajectory, utility=utility/delta_n))
    env.close()
    del gym_super_mario_bros
    del env
    exp.close()
    

def train(agent, args):
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(
                args.log, current_time + '_' + args.name + '_train')
    log_name = current_time + '_' + args.name
    writer = SummaryWriter(log_dir)
    print("start training...")        
    
    probe = FloatTensor([0.4, 0.2, 0.1, 0.1, 0.2])
    
    mp.set_start_method('spawn')
    
    for num_eps in range(0, int(args.episode_num), 10):
        loss = 0.0
        
        random.seed()
        exp_recv, exp_send = mp.Pipe()
        
        args.epsilon = agent.epsilon

        p = mp.Process(target=gain_exp, args=(args, probe, exp_send, num_eps))
        p.start()
        experience = exp_recv.recv()
        p.join()

        for tr in experience["trajectory"]:
            s, a, s_, r, t = tr
            agent.memorize(s, a, s_, r, t)

        hardworking = 1000
        for hw in range(hardworking):
            if args.single:
                # single objective learning
                loss += agent.learn(probe) 
            else:
                # multi-objective learning
                loss += agent.learn()

        writer.add_scalar('train/loss', loss/hardworking, num_eps)
        
        print("end of eps %d with utility %0.2f loss: %0.4f" % (
            num_eps,
            experience["utility"],
            loss/hardworking))

        agent.save(args.save, "m.{}_{}_n.{}_tmp".format(
                args.method, args.model, args.name))

        if num_eps % 50 == 0:
            t = mp.Process(target=validate, args=(args, log_name, probe, num_eps))
            t.start()
    
    t.joint()    
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
                    [60 * args.nframe, 
                     64, 
                     1])
    action_size = env.action_space.n
    reward_size = 5
    env.close()

    # generate an agent for training
    if args.serialize:
        model = torch.load("{}{}.pkl".format(args.save,
                                             "m.{}_{}_n.{}_tmp".format(args.method, args.model, args.name)))
        optimizer = torch.load("{}{}_opt.pkl".format(args.save,
                                             "m.{}_{}_n.{}_tmp".format(args.method, args.model, args.name)))
    else:
        model = get_new_model(args.method, args.model, state_size, action_size, reward_size)
        optimizer = None
    
    agent = MetaAgent(model, args, optimizer=optimizer, is_train=True)

    train(agent, args)
