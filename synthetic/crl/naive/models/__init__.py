from __future__ import absolute_import, division, print_function
from .linear import NaiveLinearCQN
from .linear_ext import NaiveLinearCQN2
from .onelayer import NaiveOnelayerCQN


def get_new_model(name, state_size, action_size, reward_size):
    if name == 'linear':
        return NaiveLinearCQN(state_size, action_size, reward_size)
    elif name == 'linear_ext':
        return NaiveLinearCQN2(state_size, action_size, reward_size)
    elif name == 'onelayer':
        return NaiveOnelayerCQN(state_size, action_size, reward_size)
    else:
        print("model %s doesn't exist." % (name))
        return None
