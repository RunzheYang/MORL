from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class NaiveLinearCQN2(torch.nn.Module):
    '''
        Linear Controllable Q-Network, Naive Version
    '''

    def __init__(self, state_size, action_size, reward_size):
        super(NaiveLinearCQN2, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size

        # S x A -> (W -> R). =>. S x W -> (A -> R)
        self.affine1 = nn.Linear(state_size + reward_size,
                                 (state_size + reward_size) * 16)
        self.affine2 = nn.Linear((state_size + reward_size) * 16,
                                 (state_size + reward_size) * 32)
        self.affine3 = nn.Linear((state_size + reward_size) * 32,
                                 (state_size + reward_size) * 64)
        self.affine4 = nn.Linear((state_size + reward_size) * 64,
                                 (state_size + reward_size) * 32)
        self.affine5 = nn.Linear((state_size + reward_size) * 32,
                                 action_size*reward_size)
        self.affine6 = nn.Linear(action_size*reward_size,
                                 action_size)

    def forward(self, state, preference):
        x = torch.cat((state, preference), dim=1)
        x = x.view(x.size(0), -1)
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        x = F.relu(self.affine3(x))
        x = F.relu(self.affine4(x))
        x = F.relu(self.affine5(x))
        q = self.affine6(x)
        hq = q.detach().max(dim=1)[0]
        return hq, q
