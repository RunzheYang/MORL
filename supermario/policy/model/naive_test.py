"""
Implementation of Naive MODQN - Multi-Objective Deep Q Network

The algorithm is developed with Pytorch

Author: Runzhe Yang
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class NaiveTestCQN(torch.nn.Module):
    '''
        Convolutional Controllable Q-Network, Naive Version
    '''

    def __init__(self, state_size, action_size, reward_size):
        super(NaiveTestCQN, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size

        in_channel = self.state_size[2]
        wi = int(((self.state_size[0] - 3) / 4 / 2 - 3) / 2 / 4)
        hi = int(((self.state_size[1] - 3) / 4 / 2 - 2) / 2 / 4)
        feature_size = int(wi * hi * 2)

        # S x A -> (W -> R). =>. S x W -> (A -> R)
        self.conv1 = nn.Conv2d(in_channel, 2, kernel_size=5, stride=2)
        # self.bn1 = nn.BatchNorm2d(4)
        self.pool1 = torch.nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(2, 2, kernel_size=5, stride=2)
        self.pool2 = torch.nn.MaxPool2d(4, 4)
        # self.bn2 = nn.BatchNorm2d(4)
        
        self.affine1 = nn.Linear(feature_size + reward_size,
                                 (feature_size + reward_size) * 1)
        self.affine2 = nn.Linear((feature_size + reward_size) * 1,
                                 action_size)

    def forward(self, state, preference, execmask=None):
        state = state.transpose(1, -1).transpose(-2,-1) / 255.0
        feat = self.pool1(self.conv1(state))
        feat = self.pool2(self.conv2(feat))
        feat = feat.view(feat.size(0), -1)
        x = torch.cat((feat, preference), dim=1)
        x = x.view(x.size(0), -1)
        x = F.relu(self.affine1(x))
        q = self.affine2(x)
        if execmask is not None:
            q = torch.add(q, execmask)
        hq = q.detach().max(dim=1)[0]
        return hq, q
