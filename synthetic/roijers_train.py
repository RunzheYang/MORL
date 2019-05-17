from __future__ import absolute_import, division, print_function
import argparse
import numpy as np
import torch
from utils.monitor import Monitor
from envs.mo_env import MultiObjectiveEnv

import queue
from collections import namedtuple
from termcolor import colored

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


optvalue = namedtuple("value", ["v0", "v1", "l", "u"])

def is_corner(corner_w, S):
    eps = 1e-6
    for s in S:
        if s.l < corner_w[0]-eps and\
           s.u > corner_w[0]+eps:
            print(colored("skip {} ...".format(corner_w), "green"))
            return False
    return True

def intersect(v, s):
    d = (v.v1 - v.v0) - (s.v1 - s.v0)
    if d == 0: return None
    w = (v.v1 - s.v1) / d
    if w < v.l or w > v.u or w < s.l or w > s.u:
        return None
    return w

def update_ccs(S, corWs, new_value):
    if len(S) == 0:
        nv = optvalue(new_value[0], new_value[1], 0.0, 1.0)
        S.add(nv)
        print(colored("add {} to set.".format(nv), "green"))
    else:
        discard = True
        useless = []
        updates = []
        nv = optvalue(new_value[0], new_value[1], 0.0, 1.0) 
        for s in S:
            dnv = nv.v0 - nv.v1
            ds = s.v0 - s.v1
            if (nv.v1+dnv*s.l > s.v1+ds*s.l or nv.v1+dnv*s.l == s.v1+ds*s.l) and\
               (nv.v1+dnv*s.u > s.v1+ds*s.u or nv.v1+dnv*s.u == s.v1+ds*s.u):
                if nv.v1+dnv*s.l == s.v1+ds*s.l: nv = nv._replace(l = s.l)
                if nv.v1+dnv*s.u == s.v1+ds*s.u: nv = nv._replace(u = s.u)
                if nv.v1+dnv*s.l == s.v1+ds*s.l and\
                   nv.v1+dnv*s.u == s.v1+ds*s.u:
                    print("repeat! compare to ", s)
                    discard = True
                else:
                    useless.append(s)
                    discard = False
            # elif (nv.v1+dnv*s.l < s.v1+ds*s.l or nv.v1+dnv*s.l == s.v1+ds*s.l) and\
            #      (nv.v1+dnv*s.u < s.v1+ds*s.u or nv.v1+dnv*s.u == s.v1+ds*s.u):
            
            else:
                # None if the intersection is out of range
                w = intersect(nv, s)
                if w and nv.v1 > s.v1:
                    if w < nv.u: nv = nv._replace(u = w)
                    if w > s.l: 
                        useless.append(s)
                        s = s._replace(l = w)
                        updates.append(s)
                    corWs.put_nowait(FloatTensor([w, 1.0-w]))
                    print(colored("add perference {} to set.".format(w), "green"))
                    discard = False
                elif w and nv.v0 > s.v0:
                    if w > nv.l: nv = nv._replace(l = w)
                    if w < s.u: 
                        useless.append(s)
                        s = s._replace(u = w)
                        updates.append(s)
                    corWs.put_nowait(FloatTensor([w, 1.0-w]))
                    print(colored("add perference {} to set.".format(w), "green"))
                    discard = False
        
        for s in useless:
            print(colored("remove {} from set.".format(s), "green"))
            S.remove(s)

        for s in updates:
            print(colored("update {} in set.".format(s), "green"))
            S.add(s)

        if not discard:
            S.add(nv)
            print(colored("add {} to set.".format(nv), "green"))
        else:
            print(colored("give up to add {} to set.".format(nv), "green"))

    return S, corWs


def train(env, agent, args):
    monitor = Monitor(train=True, spec="-{}".format(args.method))
    monitor.init_log(args.log, "roi_m.{}_e.{}_n.{}".format(args.model, args.env_name, args.name))
    env.reset()

    S = set()

    corWs = queue.Queue()

    # add two extreme points
    corWs.put(FloatTensor([1.0, 0.0]))
    corWs.put(FloatTensor([0.0, 1.0]))

    # outer_loop!
    for _ in range(args.ws):

        print(colored("size of corWs: {}".format(corWs.qsize()), "green"))

        if corWs.qsize() == 0:
            corWs.put(FloatTensor([1.0, 0.0]))
            corWs.put(FloatTensor([0.0, 1.0]))

        corner_w = corWs.get_nowait()
        while not is_corner(corner_w, S) and corWs.qsize()>0:
            corner_w = corWs.get_nowait()
            print(colored("{} left....".format(corWs.qsize()), "green"))
        if not is_corner(corner_w, S):
            print(colored("no more corner w...", "green"))
            print(colored("Final S contains", "green"))
            for s in S:
                print(colored(s, "green"))
            break
        print(colored("solve for w: {}".format(corner_w), "green"))

        for num_eps in range(int(args.episode_num / args.ws)):
            terminal = False
            env.reset()
            loss = 0
            cnt = 0
            tot_reward = 0

            tot_reward_mo = 0

            probe = None
            if args.env_name == "dst":
                probe = corner_w
            elif args.env_name in ['ft', 'ft5', 'ft7']:
                probe = FloatTensor([0.8, 0.2, 0.0, 0.0, 0.0, 0.0])

            while not terminal:
                state = env.observe()
                action = agent.act(state, corner_w)
                agent.w_kept = corner_w
                next_state, reward, terminal = env.step(action)
                if args.log:
                    monitor.add_log(state, action, reward, terminal, agent.w_kept)
                agent.memorize(state, action, next_state, reward, terminal, roi=True)
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
            print("end of eps %d with total reward (1) %0.2f (%0.2f, %0.2f), the Q is %0.2f | %0.2f; loss: %0.4f" % (
                num_eps,
                tot_reward,
                tot_reward_mo[0],
                tot_reward_mo[1],
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


        # agent.is_train=False
        terminal = False
        env.reset()
        cnt = 0
        tot_reward_mo = 0
        while not terminal:
            state = env.observe()
            action = agent.act(state, corner_w)
            agent.w_kept = corner_w
            next_state, reward, terminal = env.step(action)
            if cnt > 100:
                terminal = True
                agent.reset()
            tot_reward_mo = tot_reward_mo + reward * np.power(args.gamma, cnt)
            cnt = cnt + 1
        agent.is_train=True

        S, corWs = update_ccs(S, corWs, tot_reward_mo)

        print(colored("----------------\n", "red"))
        print(colored("Current S contains", "red"))
        for s in S:
            print(colored(s, "red"))
        print(colored("----------------\n", "red"))

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
