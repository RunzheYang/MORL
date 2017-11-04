import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class LinearDQN(torch.nn.Module):
	
	def __init__(self, state_size, action_size, reward_size):
		super(LinearDQN, self).__init__()
		self.affine1 = nn.Linear(state_size, state_size * 20)
		self.affine2 = nn.Linear(state_size * 20, action_size * reward_size)

	def forward(self, x):
		x = x.view(x.size(0), state_size)
		x = F.relu(self.affine1(x))
		act_rew = self.affine2(x)
		return act_rew
		