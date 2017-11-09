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

	def forward(self, state, preference, w_num=1):
		s_num = preference.size(0) / w_num
		x = torch.cat((state, preference), dim=1)
		x = x.view(x.size(0), -1)
		x = F.relu(self.affine1(x))
		q = self.affine2(x)
		# batch x action x reward_n
		q = q.view(q.size(0), self.action_size, self.reward_size)
		# (batch x action) x reward_n
		hq = q.view(-1, self.reward_size)
		# <(batch x action) x reward_n, reward_n x pref>
		mask = torch.arange(start=0, end=preference.size(0), 
							step=s_num).long()
		hq = torch.mm(hq, preference[
					Variable(mask)
					].t())
		hq = hq.view(-1, self.action_size, w_num)
		print hq
		hq = hq.max(2)[0].max(1)[0]
		print hq
		print m
		return hq, q
