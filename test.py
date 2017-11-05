import argparse
import numpy as np
import torch
from torch.autograd import Variable
from utils.monitor import Monitor
from ENV.mo_env import MultiObjectiveEnv


parser = argparse.ArgumentParser(description='MORL-TEST')
# CONFIG
parser.add_argument('--env-name', default='dst', metavar='ENVNAME',
				help='environment to train on (default: dst)')
parser.add_argument('--method', default='crl-naive', metavar='METHODS',
				help='methods: crl-naive | crl-envelope | ols')
parser.add_argument('--model', default='linear', metavar='MODELS',
				help='linear | cnn | cnn + lstm')
parser.add_argument('--gamma', type=float, default=0.99, metavar='GAMMA',
				help='gamma for infinite horizonal MDPs')
# TEST
parser.add_argument('--mem-size', type=int, default=10000, metavar='M',
				help='max size of the replay memory')
parser.add_argument('--batch-size', type=int, default=256, metavar='B',
				help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
				help='learning rate')
parser.add_argument('--epsilon', type=float, default=0.5, metavar='EPS',
				help='epsilon greedy exploration')
parser.add_argument('--weight-num', type=int, default=32, metavar='WN',
				help='number of sampled weights per iteration')
parser.add_argument('--episode-num', type=int, default=100, metavar='EN',
				help='number of episodes for training')
parser.add_argument('--optimizer', default='Adam', metavar='OPT',
				help='optimizer: Adam | RMSprop')
parser.add_argument('--update-freq', type=int, default=32, metavar='OPT',
				help='optimizer: Adam | RMSprop')
# LOG & SAVING
parser.add_argument('--save', default='CRL/NAIVE/saved/', metavar='SAVE',
				help='address for saving trained models')
parser.add_argument('--log', default='CRL/NAIVE/logs/', metavar='LOG',
				help='address for recording training informtions')


def test(env, agent, args, shared_mem=None):
	monitor = Monitor(spec="-rand rand [0.01, 0.99]")
	env.reset()
	for num_eps in xrange(args.episode_num):
		terminal = False
		env.reset()
		loss = 0
		cnt  = 0
		tot_reward = 0
		reward = [0, 0]
		pref = [0.01, 0.99]
		while not terminal:
			state  = env.observe()
			action = agent.act(state, preference=torch.FloatTensor(pref))
			# _, q = agent.model(Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True), 
			# 			Variable(torch.FloatTensor(pref).unsqueeze(0), volatile=True))
			# if num_eps == 0: monitor.text("Qmax: %f"%q[0, action].data[0])
			# action = agent.act(state)
			next_state, reward, terminal = env.step(action)

			# HQ, _ = agent.model(Variable(torch.FloatTensor(next_state).unsqueeze(0), volatile=True), 
			# 			Variable(torch.FloatTensor(pref).unsqueeze(0), volatile=True))
			# if num_eps == 0: monitor.text("wr+yHQ: %f"%(args.gamma * HQ.data[0] + pref[0]*reward[0]+pref[1]*reward[1]))

			if cnt > 50:
				terminal = True
			cnt = cnt + 1
			tot_reward = args.gamma * tot_reward + pref[0]*reward[0]+pref[1]*reward[1]
		
		_, q = agent.model(Variable(torch.FloatTensor([0,0]).unsqueeze(0), volatile=True), 
						Variable(torch.FloatTensor(pref).unsqueeze(0), volatile=True))
		print "end of eps %d with total reward (1) %0.2f, the Q is %0.2f | %0.2f; loss: %0.4f"%(
						num_eps, 
						tot_reward, 
						q[0, 3].data[0], 
						q[0, 1].data[0], 
						loss / cnt)
		monitor.update(num_eps, 
					   tot_reward, 
					   q[0, 3].data[0], 
					   q[0, 1].data[0], 
					   loss / cnt)


if __name__ == '__main__':	
	args = parser.parse_args()	

	# setup the environment
	env = MultiObjectiveEnv(args.env_name)
	
	# get state / action / reward sizes
	state_size  = len(env.state_spec)
	action_size = env.action_spec[2][1] - env.action_spec[2][0]
	reward_size = len(env.reward_spec)

	# generate an agent for testing
	agent = None
	if args.method == 'crl-naive':
		from CRL.NAIVE.meta   import MetaAgent
		model = torch.load("{}{}.pkl".format(args.save, args.model))
		agent = MetaAgent(model, args, is_train=False)

	test(env, agent, args)


