import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class NaiveLinearCQN(torch.nn.Module):
    '''
        Linear Controllable Q-Network, Naive Version
    '''

    def __init__(self, state_size, action_size, reward_size):
        super(NaiveLinearCQN, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size

        # S x A -> (W -> R). =>. S x W -> (A -> R)
        self.affine1 = nn.Linear(state_size + reward_size,
                                 (state_size + reward_size) * 40)
        self.affine2 = nn.Linear((state_size + reward_size) * 40,
                                 (state_size + reward_size) * 60)
        self.affine3 = nn.Linear((state_size + reward_size) * 60,
                                 action_size)

    def forward(self, state, preference):
        x = torch.cat((state, preference), dim=1)
        x = x.view(x.size(0), -1)
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        q = self.affine3(x)
        hq = q.max(dim=1)[0]
        return hq, q
