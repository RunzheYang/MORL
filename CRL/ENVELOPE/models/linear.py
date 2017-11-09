import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class EnvelopeLinearCQN(torch.nn.Module):

	'''
		Linear Controllable Q-Network, Envelope Version
	'''
	
	def __init__(self, state_size, action_size, reward_size):
		super(EnvelopeLinearCQN, self).__init__()
		
		self.state_size  = state_size
		self.action_size = action_size
		self.reward_size = reward_size

		# S x A -> (W -> R^n). =>. S x W -> (A -> R^n)
		self.affine1 = nn.Linear(state_size + reward_size, 
								 (state_size + reward_size) * 40)
		self.affine2 = nn.Linear((state_size + reward_size) * 40, 
								 action_size * reward_size)

	def H(self, Q, w, s_num, w_num):
		# mask for reordering the batch
		mask = torch.cat(
				[torch.arange(i, s_num * w_num + i, s_num) 
					for i in xrange(s_num)]).long()
		reQ = Q.view(-1, self.action_size * self.reward_size
							)[mask].view(-1, self.reward_size)
		
		# extend Q batch and preference batch
		reQ_ext = reQ.repeat(w_num, 1)
		w_ext = w.unsqueeze(2).repeat(1, self.action_size * w_num, 1).view(-1,2)

		# produce the inner products
		prod = torch.bmm(reQ_ext.unsqueeze(1), w_ext.unsqueeze(2)).squeeze()

		# mask for take max over actions and weights
		prod = prod.view(-1, self.action_size * w_num)
		inds = prod.max(1)[1]
		mask = torch.ByteTensor(prod.size()).zero_()
		mask.scatter_(1, inds.data.unsqueeze(1), 1)
		mask = mask.view(-1, 1).repeat(1, self.reward_size)

		# get the HQ
		HQ = reQ_ext.masked_select(Variable(mask)).view(-1, self.reward_size)
		
		return HQ


	def forward(self, state, preference, w_num=1):
		s_num = preference.size(0) / w_num
		x = torch.cat((state, preference), dim=1)
		x = x.view(x.size(0), -1)
		x = F.relu(self.affine1(x))
		q = self.affine2(x)
		q = q.view(q.size(0), self.action_size, self.reward_size)
		
		hq = self.H(q.view(-1, self.reward_size), preference, s_num, w_num)

		return hq, q
