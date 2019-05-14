from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class NaiveOnelayerCQN(torch.nn.Module):
    '''
        One-Layer Controllable Q-Network, Naive Version
    '''

    def __init__(self, state_size, action_size, reward_size):
        super(NaiveLinearCQN, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size

        self.affine = nn.Linear(state_size + reward_size,
                                 action_size)

    def forward(self, state, preference):
        x = torch.cat((state, preference), dim=1)
        x = x.view(x.size(0), -1)
        q = self.affine(x)
        hq = q.detach().max(dim=1)[0]
        return hq, q
