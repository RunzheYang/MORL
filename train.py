import argparse
import numpy as np
import torch
from torch.autograd import Variable
from ENV.mo_env import MultiObjectiveEnv


parser = argparse.ArgumentParser(description='MORL')
# CONFIG
parser.add_argument('--env-name', default='dst', metavar='ENVNAME',
				help='environment to train on (default: dst)')
parser.add_argument('--method', default='crl-naive', metavar='METHODS',
				help='methods: crl-naive | crl-envelope | ols')
parser.add_argument('--model', default='linear', metavar='MODELS',
				help='linear | cnn | cnn + lstm')
parser.add_argument('--gamma', type=float, default=0.99, metavar='GAMMA',
				help='gamma for infinite horizonal MDPs')
# TRAINING
parser.add_argument('--mem-size', type=int, default=20000, metavar='M',
				help='max size of the replay memory')
parser.add_argument('--batch-size', type=int, default=256, metavar='B',
				help='batch size')
parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
				help='learning rate')
parser.add_argument('--epsilon', type=float, default=0.3, metavar='EPS',
				help='epsilon greedy exploration')
parser.add_argument('--weight-num', type=int, default=1, metavar='WN',
				help='number of sampled weights per iteration')
parser.add_argument('--episode-num', type=int, default=1000, metavar='EN',
				help='number of episodes for training')
parser.add_argument('--optimizer', default='Adam', metavar='OPT',
				help='optimizer: Adam | RMSprop')
# LOG & SAVING
parser.add_argument('--save', default='CRL/NAIVE/saved/', metavar='SAVE',
				help='address for saving trained models')
parser.add_argument('--log', default='CRL/NAIVE/logs/', metavar='LOG',
				help='address for recording training informtions')


def train(env, agent, args, shared_mem=None):
	env.reset()
	for num_eps in xrange(args.episode_num):
		terminal = False
		env.reset()
		loss = 0
		cnt = 0
		while not terminal:
			state  = env.observe()
			action = agent.act(state)
			next_state, reward, terminal = env.step(action)
			if cnt > 50:
				terminal = True
			agent.memorize(state, action, next_state, reward, terminal)
			tmp = agent.learn()
			if tmp is not None:
				loss = loss + tmp
			cnt = cnt + 1
		_, q = agent.model(Variable(torch.FloatTensor([0,0]).unsqueeze(0)), 
						Variable(torch.FloatTensor([1,0]).unsqueeze(0)))
		print "end of eps %d with final treasure %0.2f, the Q is %0.2f | %0.2f; loss: %0.4f"%(
						num_eps, 0.9*reward[0]+0.1*reward[1], 
						q[0, 3].data.numpy(), q[0, 1].data.numpy(), loss / cnt)
			# print "s:", state, "\ta:", action, "\ts':", next_state, "\tr:", reward
		# print "EPISODE %d ENDS"%(num_eps)


if __name__ == '__main__':	
	args = parser.parse_args()	

	# setup the environment
	env = MultiObjectiveEnv(args.env_name)
	
	# get state / action / reward sizes
	state_size  = len(env.state_spec)
	action_size = env.action_spec[2][1] - env.action_spec[2][0]
	reward_size = len(env.reward_spec)

	# generate an agent for initial training
	agent = None
	if args.method == 'crl-naive':
		from CRL.NAIVE.meta   import MetaAgent
		from CRL.NAIVE.models import get_new_model
		agent = get_new_model(args.model, state_size, action_size, reward_size)
		agent = MetaAgent(agent, args, is_train=True)

	train(env, agent, args)


