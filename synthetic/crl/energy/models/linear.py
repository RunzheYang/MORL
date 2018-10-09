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


class EnergyLinearCQN(torch.nn.Module):
    '''
        Linear Controllable Q-Network, Envelope Version
    '''

    def __init__(self, state_size, action_size, reward_size):
        super(EnergyLinearCQN, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size

        # S x A -> (W -> R^n). =>. S x W -> (A -> R^n)
        self.affine1 = nn.Linear(state_size + reward_size,
                                 (state_size + reward_size) * 16)
        self.affine2 = nn.Linear((state_size + reward_size) * 16,
                                 (state_size + reward_size) * 32)
        self.affine3 = nn.Linear((state_size + reward_size) * 32,
                                 action_size * reward_size)

    def H(self, Q, w, s_num, w_num, alpha):
        # mask for reordering the batch
        mask = torch.cat(
            [torch.arange(i, s_num * w_num + i, s_num)
             for i in xrange(s_num)]).type(LongTensor)
        reQ = Q.view(-1, self.action_size * self.reward_size
                     )[mask].view(-1, self.reward_size)

        # extend Q batch and preference batch
        reQ_ext = reQ.repeat(w_num, 1)
        w_ext = w.unsqueeze(2).repeat(1, self.action_size * w_num, 1).view(-1, 2)

        # produce the inner products
        prod = torch.bmm(reQ_ext.unsqueeze(1), w_ext.unsqueeze(2)).squeeze()

        # mask for take max over actions and weights
        prod = prod.view(-1, self.action_size * w_num)
        reQ_ext = reQ_ext.view(-1, self.action_size * w_num, self.reward_size)
        # inds = prod.max(1)[1]
        # mask = ByteTensor(prod.size()).zero_()
        # mask.scatter_(1, inds.data.unsqueeze(1), 1)
        # mask = mask.view(-1, 1).repeat(1, self.reward_size)
        softind = F.softmax((1.0 / alpha) * prod)

        # print softind.unsqueeze(1)
        # print reQ_ext
        # get the HQ
        # HQ = reQ_ext.masked_select(Variable(mask)).view(-1, self.reward_size)
        HQ = torch.bmm(softind.unsqueeze(1), reQ_ext).view(-1, self.reward_size)

        return HQ

    def forward(self, state, preference, w_num=1, alpha=1e-5):
        s_num = preference.size(0) / w_num
        x = torch.cat((state, preference), dim=1)
        x = x.view(x.size(0), -1)
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        q = self.affine3(x)
        q = q.view(q.size(0), self.action_size, self.reward_size)

        hq = self.H(q.view(-1, self.reward_size),
                    preference, s_num, w_num, alpha)

        return hq, q
