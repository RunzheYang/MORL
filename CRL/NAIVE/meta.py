import random
import torch
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from collections import namedtuple
from collections import deque

class MetaAgent(object):
	'''
	(1) act: how to sample an action to examine the learning 
		outcomes or explore the environment;
	(2) memorize: how to store observed observations in order to
		help learing or establishing the empirical model of the 
		enviroment;
	(3) learn: how the agent learns from the observations via 
		explicitor implicit inference, how to optimize the policy 
		model.
	'''
	def __init__(self, model, args, is_train=False):
		self.model = model
		self.is_train = is_train
		self.gamma =   args.gamma
		self.epsilon = args.epsilon

		self.mem_size = args.mem_size
		self.batch_size = args.batch_size
		self.weight_num = args.weight_num
		self.trans_mem = deque()
		self.trans = namedtuple('trans', ['s', 'a', 's_', 'r', 'd'])

		if args.optimizer == 'Adam':
			self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
		elif args.optimizer == 'RMSprop':
			self.optimizer = optim.RMSprop(self.model.parameters(), lr=args.lr)

		if self.is_train: self.model.train()
		
		
	def act(self, state, preference=None):
		# random pick a preference if it is not specified
		if preference is None:
			preference = torch.randn(self.model.reward_size)
			preference = torch.abs(preference) / torch.norm(preference, p=1)
		state = torch.from_numpy(state).float()

		_, Q = self.model(
				Variable(state.unsqueeze(0)), 
				Variable(preference.unsqueeze(0)))
		
		action = Q.max(1)[1].data.numpy()
		action = action[0, 0]

		if len(self.trans_mem) < self.batch_size or \
				self.is_train and torch.rand(1)[0] < self.epsilon:
			action = np.random.choice(self.model.action_size, 1)[0]
			
		return action


	def memorize(self, state, action, next_state, reward, terminal):
		self.trans_mem.append(self.trans(
							torch.from_numpy(state).float(),		# state
							action, 								# action
							torch.from_numpy(next_state).float(), 	# next state
							torch.from_numpy(reward).float(), 		# reward
							terminal))								# terminal
		if len(self.trans_mem) > self.mem_size:
			self.trans_mem.popleft()


	def get_one_hot(self, num_dim, index):
		tensor = torch.ByteTensor(num_dim).zero_()
		tensor[index] = 1
		return tensor


	def learn(self):
		if len(self.trans_mem) > self.batch_size:
			minibatch = random.sample(self.trans_mem, self.batch_size)
			batchify = lambda x: list(x) * self.weight_num
			state_batch      = batchify(map(lambda x: x.s.unsqueeze(0), minibatch))
			action_batch     = batchify(map(lambda x: self.get_one_hot(self.model.action_size, x.a), minibatch))
			reward_batch     = batchify(map(lambda x: x.r.unsqueeze(0), minibatch))
			next_state_batch = batchify(map(lambda x: x.s_.unsqueeze(0), minibatch))
			terminal_batch   = batchify(map(lambda x: x.d, minibatch))

			# preference_batch = np.random.randn(self.weight_num, self.model.reward_size)
			preference_batch = np.array([[0.99,0.01]]).repeat(self.weight_num, axis=0)
			preference_batch = np.abs(preference_batch) / \
								np.linalg.norm(preference_batch, ord=1, axis=1, keepdims=True)
			preference_batch = torch.from_numpy(preference_batch.repeat(self.batch_size, axis=0)).float()

			Target_Q = []
			__, Q    = self.model(Variable(torch.cat(state_batch, dim=0)),
								  Variable(preference_batch))
			HQ, _    = self.model(Variable(torch.cat(next_state_batch, dim=0)),
								  Variable(preference_batch))

			w_reward_batch = torch.bmm(preference_batch.unsqueeze(1),
									   	torch.cat(reward_batch, dim=0).unsqueeze(2)
									  ).squeeze()
			for i in range(0, self.batch_size * self.weight_num):
				r = Variable(torch.FloatTensor([w_reward_batch[i]]))
				if terminal_batch[i]:
					Target_Q.append(r)
				else:
					Target_Q.append(r + self.gamma * HQ[i])
			
			Target_Q = torch.cat(Target_Q, dim=0)

			self.optimizer.zero_grad()
			action_mask = Variable(torch.cat(action_batch, dim=0))
			loss = torch.sum((Q.masked_select(action_mask) - Target_Q).pow(2))
			report_loss = loss.data[0] / (self.batch_size * self.weight_num)
			loss.backward()

			self.optimizer.step()
			return	report_loss
		return 1.0

