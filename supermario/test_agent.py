from __future__ import absolute_import, division, print_function
import argparse
import numpy as np
import torch

from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

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
parser.add_argument('--episode-num', type=int, default=2000, metavar='EN',
                    help='number of episodes for training')
parser.add_argument('--optimizer', default='Adam', metavar='OPT',
                    help='optimizer: Adam | RMSprop')
parser.add_argument('--update-freq', type=int, default=100, metavar='OPT',
                    help='optimizer: Adam | RMSprop')
parser.add_argument('--beta', type=float, default=0.01, metavar='BETA',
                    help='(initial) beta for evelope algorithm, default = 0.01')
parser.add_argument('--homotopy', default=False, action='store_true',
                    help='use homotopy optimization method')
# LOG & SAVING
parser.add_argument('--serialize', default=False, action='store_true',
                    help='serialize a model')
parser.add_argument('--save', default='crl/naive/saved/', metavar='SAVE',
                    help='path for saving trained models')
parser.add_argument('--name', default='', metavar='name',
                    help='specify a name for saving the model')
parser.add_argument('--log', default='crl/naive/logs/', metavar='LOG',
                    help='path for recording training informtion')


use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


def train(env, agent, args):
    print("start training...")        
    env.reset()
    for num_eps in range(args.episode_num):
        terminal = False
        loss = 0
        cnt = 0
        utility = 0

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
            print("action", action)
            print("reward", reward, "\n")

            agent.memorize(state, action, next_state, reward, terminal)
            
            if cnt % 200 == 0: loss += agent.learn()
            
            if cnt >500:
                terminal = True
                agent.reset()
            utility = utility + (probe.cpu().numpy().dot(reward)) * np.power(args.gamma, cnt)
            cnt = cnt + 1
    
        print("end of eps %d with utility %0.2f loss: %0.4f" % (
            num_eps,
            utility,
            loss / cnt))
    
    env.close()

    # if num_eps+1 % 100 == 0:
    # 	agent.save(args.save, args.model+args.name+"_tmp_{}".format(number))
    agent.save(args.save, "m.{}_e.{}_n.{}".format(args.model, args.env_name, args.name))


if __name__ == '__main__':
    args = parser.parse_args()

    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
    # reward type (X_POSITION, ENERMY, TIME, DEATH, COIN)

    # get state / action / reward sizes
    state_size = torch.Tensor(env.observation_space.high).size()
    action_size = env.action_space.n
    reward_size = 5

    print("import 1")

    # generate an agent for initial training
    from policy.morlpolicy import MetaAgent

    print("import 2")
    from policy.model import get_new_model

    print("if serialize")        

    if args.serialize:
        model = torch.load("{}{}.pkl".format(args.save,
                                             "m.{}_e.{}_n.{}".format(args.model, args.env_name, args.name)))
    else:
        model = get_new_model(args.method, args.model, state_size, action_size, reward_size)
    
    print("prepare for agent")        
    agent = MetaAgent(model, args, is_train=True)

    train(env, agent, args)
