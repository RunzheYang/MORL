## multi-obejcetive super mario bros
## modified by Runzhe Yang on Dec. 8, 2018

import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import math
from torch.nn import init

from torch.distributions.categorical import Categorical


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
        

class BaseActorCriticNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(BaseActorCriticNetwork, self).__init__()
        linear = nn.Linear
        self.feature = nn.Sequential(
            linear(input_size, 128),
            nn.ReLU(),
            linear(128, 128),
            nn.ReLU()
        )
        self.actor = linear(128, output_size)
        self.critic = linear(128, 1)

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.kaiming_uniform_(p.weight, a=1.0)
                p.bias.data.zero_()

    def forward(self, state):
        x = self.feature(state)
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value


class DeepCnnActorCriticNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DeepCnnActorCriticNetwork, self).__init__()
        
        linear = nn.Linear

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=4,
                stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4),
            nn.ReLU(),
            Flatten(),
            linear(50176, 512),
            nn.ReLU()
        )
        self.actor = linear(512, output_size)
        self.critic = linear(512, 1)

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.kaiming_uniform_(p.weight, a=1.0)
                p.bias.data.zero_()

    def forward(self, state):
        x = self.feature(state)
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value


class CnnActorCriticNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(CnnActorCriticNetwork, self).__init__()
        
        linear = nn.Linear

        self.feature = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
            linear(
                7 * 7 * 64,
                512),
            nn.LeakyReLU(),
        )
        self.actor = linear(512, output_size)
        self.critic = linear(512, 1)

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.kaiming_uniform_(p.weight, a=1.0)
                p.bias.data.zero_()

    def forward(self, state):
        x = self.feature(state)
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value


class NaiveMoCnnActorCriticNetwork(nn.Module):
    def __init__(self, input_size, output_size, reward_size):
        super(NaiveMoCnnActorCriticNetwork, self).__init__()
        linear = nn.Linear
        self.feature = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
            linear(
                7 * 7 * 64,
                512),
            nn.LeakyReLU(),
        )
        self.actor = nn.Sequential(
            linear(512+reward_size, 128),
            nn.LeakyReLU(),
            linear(128, output_size),
        )
        self.critic = nn.Sequential(
            linear(512+reward_size, 128),
            nn.LeakyReLU(),
            linear(128, 1),
        )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.kaiming_uniform_(p.weight, a=1.0)
                p.bias.data.zero_()

    def forward(self, state, preference):
        x = self.feature(state)
        x = torch.cat((x, preference), dim=1)
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value


class EnveMoCnnActorCriticNetwork(nn.Module):
    def __init__(self, input_size, output_size, reward_size):
        super(EnveMoCnnActorCriticNetwork, self).__init__()
        
        linear = nn.Linear

        self.feature = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
            linear(
                7 * 7 * 64,
                512),
            nn.LeakyReLU(),
        )
        self.actor = nn.Sequential(
            linear(512+reward_size, 128),
            nn.LeakyReLU(),
            linear(128, output_size),
        )
        self.critic = nn.Sequential(
            linear(512+reward_size, 128),
            nn.LeakyReLU(),
            linear(128, reward_size),
        )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.kaiming_uniform_(p.weight, a=1.0)
                p.bias.data.zero_()

    def forward(self, state, preference):
        x = self.feature(state)
        x = torch.cat((x, preference), dim=1)
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value
