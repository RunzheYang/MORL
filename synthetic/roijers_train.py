from __future__ import absolute_import, division, print_function
import argparse
import numpy as np
import torch
from utils.monitor import Monitor
from envs.mo_env import MultiObjectiveEnv

import Queue
from sets import Set
from collections import namedtuple

parser = argparse.ArgumentParser(description='MORL')
# CONFIG
parser.add_argument('--env-name', default='dst', metavar='ENVNAME',
                    help='environment to train on: dst | ft | ft5 | ft7')
parser.add_argument('--method', default='crl-naive', metavar='METHODS',
                    help='methods: crl-naive | crl-envelope | crl-energy')
parser.add_argument('--model', default='linear', metavar='MODELS',
                    help='linear | cnn | cnn + lstm')
parser.add_argument('--gamma', type=float, default=0.99, metavar='GAMMA',
                    help='gamma for infinite horizonal MDPs')
# TRAINING
parser.add_argument('--ws', type=int, default=20, metavar='W',
                    help='checked corner weights')
parser.add_argument('--mem-size', type=int, default=4000, metavar='M',
                    help='max size of the replay memory')
parser.add_argument('--batch-size', type=int, default=256, metavar='B',
                    help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate')
parser.add_argument('--epsilon', type=float, default=0.5, metavar='EPS',
                    help='epsilon greedy exploration')
parser.add_argument('--epsilon-decay', default=False, action='store_true',
                    help='linear epsilon decay to zero')
parser.add_argument('--weight-num', type=int, default=1, metavar='WN',
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


optvalue = namedtuple("value", ["v", "l", "u"])

def is_corner(corner_w, S):
    for s in S:
        if s.l < corner_w[0] and\
           s.u > corner_w[0]:
            return False
    return True

def update_ccs(S, corWs, new_value):
    if len(S) == 0:
        S.add(optvalue(new_value, 0.0, 1.0))
    else:
        discard = False
        nv = optvalue(new_value, 0.0, 1.0) 
        for s in S:
            if new_value[0] > s.v[0] and new_value[1] > s.v[1]:
                nv.l = s.l if s.l > nv.l
                nv.u = s.u if s.u < nv.u
                S.remove(s)
            elif new_value[0] < s.v[0] and new_value[1] < s.v[1]:
                # do nothing for this point
                discard = True
                break
            else:
                # None if the intersection is out of range
                w = intersect(nv, s)
                if w and nv.v[1] > s.v[1]:
                    nv.u = w if w < nv.u
                    s.l = w if w > s.l
                    corWs.put(FloatTensor([w, 1.0-w]))
                elif w and nv.v[0] > s.v[0]:
                    nv.l = w if w > nv.l
                    s.u = w if w < s.u
                    corWs.put(FloatTensor([w, 1.0-w]))
        if not discard:
            S.add(nv)
    return S, corWs


def train(env, agent, args):
    monitor = Monitor(train=True, spec="-{}".format(args.method))
    monitor.init_log(args.log, "roi_m.{}_e.{}_n.{}".format(args.model, args.env_name, args.name))
    env.reset()

    S = Set()

    corWs = Queue.Queue()

    # add two extreme points
    corWs.put(FloatTensor([1.0, 0.0]))
    corWs.put(FloatTensor([0.0, 1.0]))

    # outer_loop!
    for _ in range(args.ws):

        corner_w = corWs.get()
        while not is_corner(corner_w, S) and not corWs.empty():
            corner_w = corWs.get()
        if not is_corner(corner_w, S):
            print("no more corner w...")
            break
        print("solve for w:", corner_w)

        for num_eps in range(args.episode_num / args.ws):
            terminal = False
            env.reset()
            loss = 0
            cnt = 0
            tot_reward = 0

            tot_reward_mo = 0

            probe = None
            if args.env_name == "dst":
                probe = FloatTensor([0.8, 0.2])
            elif args.env_name in ['ft', 'ft5', 'ft7']:
                probe = FloatTensor([0.8, 0.2, 0.0, 0.0, 0.0, 0.0])

            while not terminal:
                state = env.observe()
                action = agent.act(state, corner_w)
                next_state, reward, terminal = env.step(action)
                if args.log:
                    monitor.add_log(state, action, reward, terminal, agent.w_kept)
                agent.memorize(state, action, next_state, reward, terminal)
                loss += agent.learn(corner_w)
                if cnt > 100:
                    terminal = True
                    agent.reset()
                tot_reward = tot_reward + (probe.cpu().numpy().dot(reward)) * np.power(args.gamma, cnt)

                tot_reward_mo = tot_reward_mo + reward * np.power(args.gamma, cnt)

                cnt = cnt + 1

            _, q = agent.predict(probe)

            if args.env_name == "dst":
                act_1 = q[0, 3]
                act_2 = q[0, 1]
            elif args.env_name in ['ft', 'ft5', 'ft7']:
                act_1 = q[0, 1]
                act_2 = q[0, 0]

            if args.method == "crl-naive":
                act_1 = act_1.data.cpu()
                act_2 = act_2.data.cpu()
            elif args.method == "crl-envelope":
                act_1 = probe.dot(act_1.data)
                act_2 = probe.dot(act_2.data)
            elif args.method == "crl-energy":
                act_1 = probe.dot(act_1.data)
                act_2 = probe.dot(act_2.data)
            print("end of eps %d with total reward (1) %0.2f, the Q is %0.2f | %0.2f; loss: %0.4f" % (
                num_eps,
                tot_reward,
                act_1,
                act_2,
                # q__max,
                loss / cnt))
            monitor.update(num_eps,
                           tot_reward,
                           act_1,
                           act_2,
                           #    q__max,
                           loss / cnt)

        S, corWs = update_ccs(S, corWs, tot_reward_mo)

    # if num_eps+1 % 100 == 0:
    # 	agent.save(args.save, args.model+args.name+"_tmp_{}".format(number))
    agent.save(args.save, "roi_m.{}_e.{}_n.{}".format(args.model, args.env_name, args.name))


if __name__ == '__main__':
    args = parser.parse_args()

    # setup the environment
    env = MultiObjectiveEnv(args.env_name)

    # get state / action / reward sizes
    state_size = len(env.state_spec)
    action_size = env.action_spec[2][1] - env.action_spec[2][0]
    reward_size = len(env.reward_spec)

    # generate an agent for initial training
    agent = None
    if args.method == 'crl-naive':
        from crl.naive.meta import MetaAgent
        from crl.naive.models import get_new_model
    elif args.method == 'crl-envelope':
        from crl.envelope.meta import MetaAgent
        from crl.envelope.models import get_new_model
    elif args.method == 'crl-energy':
        from crl.energy.meta import MetaAgent
        from crl.energy.models import get_new_model

    if args.serialize:
        model = torch.load("{}{}.pkl".format(args.save,
                                             "roi_m.{}_e.{}_n.{}".format(args.model, args.env_name, args.name)))
    else:
        model = get_new_model(args.model, state_size, action_size, reward_size)
    agent = MetaAgent(model, args, is_train=True)

    train(env, agent, args)
