"""
Implementation of Naive MODQN - Multi-Objective Deep Q Network

The algorithm is developed with Pytorch

Author: Runzhe Yang
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class NaiveCnnCQN(torch.nn.Module):
    '''
        Convolutional Controllable Q-Network, Naive Version
    '''

    def __init__(self, state_size, action_size, reward_size):
        super(NaiveCnnCQN, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size

        in_channel = self.state_size[2]
        wi = ((self.state_size[0] - 3) / 2 - 3) / 2
        hi = ((self.state_size[1] - 3) / 2 - 2) / 2
        feature_size = int(wi * hi * 32)

        # S x A -> (W -> R). =>. S x W -> (A -> R)
        self.conv1 = nn.Conv2d(in_channel, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        self.affine1 = nn.Linear(feature_size + reward_size,
                                 (feature_size + reward_size) * 16)
        self.affine2 = nn.Linear((feature_size + reward_size) * 16,
                                 (feature_size + reward_size) * 32)
        self.affine3 = nn.Linear((feature_size + reward_size) * 32,
                                 (feature_size + reward_size) * 32)
        self.affine4 = nn.Linear((feature_size + reward_size) * 32,
                                 action_size)

    def forward(self, state, preference, execmask=None):
        feat = self.bn1(self.conv1(state))
        feat = self.bn2(self.conv2(feat))
        feat = self.bn3(self.conv3(feat))
        feat = feat.view(feat.size(0), -1)
        x = torch.cat((feat, preference), dim=1)
        x = x.view(x.size(0), -1)
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        x = F.relu(self.affine3(x))
        q = self.affine4(x)
        if execmask is not None:
            q = torch.add(q, execmask)
        hq = q.detach().max(dim=1)[0]
        return hq, q
