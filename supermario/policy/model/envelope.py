"""
Implementation of Envelope MODQN - Multi-Objective Deep Q Network

The algorithm is developed with Pytorch

Author: Runzhe Yang
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class EnvelopeCnnCQN(torch.nn.Module):
    '''
        Convolutional Controllable Q-Network, Envelope Version
    '''

    def __init__(self, state_size, action_size, reward_size):
        super(EnvelopeCnnCQN, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size

        in_channel = self.state_size[2]
        wi = int((((self.state_size[0] - 3) / 4 - 3) / 4 - 3) / 4)
        hi = int((((self.state_size[1] - 3) / 4 - 3) / 4 - 3) / 4)
        feature_size = int(wi * hi * 2)

        # S x A -> (W -> R^n). =>. S x W -> (A -> R^n)
        self.conv1 = nn.Conv2d(in_channel, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = torch.nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = torch.nn.MaxPool2d(2, 2)

        self.affine1 = nn.Linear(feature_size + reward_size,
                                 (feature_size + reward_size) * 2)
        self.affine2 = nn.Linear((feature_size + reward_size) * 2,
                                 (feature_size + reward_size) * 4)
        self.affine3 = nn.Linear((feature_size + reward_size) * 4,
                                 action_size * reward_size)

    def H(self, Q, w, s_num, w_num):
        # mask for reordering the batch
        mask = torch.cat(
            [torch.arange(i, s_num * w_num + i, s_num)
             for i in range(s_num)]).type(LongTensor)
        reQ = Q.view(-1, self.action_size * self.reward_size
                     )[mask].view(-1, self.reward_size)

        # extend Q batch and preference batch
        reQ_ext = reQ.repeat(w_num, 1)
        w_ext = w.unsqueeze(2).repeat(1, self.action_size * w_num, 1)
        w_ext = w_ext.view(-1, self.reward_size)

        # produce the inner products
        prod = torch.bmm(reQ_ext.unsqueeze(1), w_ext.unsqueeze(2)).squeeze()

        # mask for take max over actions and weights
        prod = prod.view(-1, self.action_size * w_num)
        inds = prod.max(1)[1]
        mask = ByteTensor(prod.size()).zero_()
        mask.scatter_(1, inds.data.unsqueeze(1), 1)
        mask = mask.view(-1, 1).repeat(1, self.reward_size)

        # get the HQ
        HQ = reQ_ext.masked_select(Variable(mask)).view(-1, self.reward_size)

        return HQ

    def H_(self, Q, w, s_num, w_num):
        reQ = Q.view(-1, self.reward_size)

        # extend preference batch
        w_ext = w.unsqueeze(2).repeat(1, self.action_size, 1).view(-1, 2)

        # produce hte inner products
        prod = torch.bmm(reQ.unsqueeze(1), w_ext.unsqueeze(2)).squeeze()

        # mask for take max over actions
        prod = prod.view(-1, self.action_size)
        inds = prod.max(1)[1]
        mask = ByteTensor(prod.size()).zero_()
        mask.scatter_(1, inds.data.unsqueeze(1), 1)
        mask = mask.view(-1, 1).repeat(1, self.reward_size)

        # get the HQ
        HQ = reQ.masked_select(Variable(mask)).view(-1, self.reward_size)

        return HQ

    def forward(self, state, preference, w_num=1, execmask=None):
        s_num = int(preference.size(0) / w_num)

        state = state.transpose(1, -1).transpose(-2,-1)
        feat = self.pool1(self.bn1(self.conv1(state)))
        feat = self.pool2(self.bn2(self.conv2(feat)))
        feat = self.pool3(self.bn3(self.conv3(feat)))
        feat = feat.view(feat.size(0), -1)

        x = torch.cat((feat, preference), dim=1)
        x = x.view(x.size(0), -1)
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        q = self.affine3(x)
        q = q.view(q.size(0), self.action_size, self.reward_size)
        if execmask is not None:
            execmask = execmask.view(execmask.size(0), self.action_size, -1)
            q = torch.add(q, execmask)

        hq = self.H(q.detach().view(-1, self.reward_size), preference, s_num, w_num)

        return hq, q
