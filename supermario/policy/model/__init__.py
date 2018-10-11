from __future__ import absolute_import, division, print_function
from .naive import NaiveCnnCQN
from .envelope import EnvelopeCnnCQN
from .naive_test import NaiveTestCQN


def get_new_model(method, name, state_size, action_size, reward_size):
    if method == 'naive':
        if name == 'cnn':
            return NaiveCnnCQN(state_size, action_size, reward_size)
        if name == 'test':
            return NaiveTestCQN(state_size, action_size, reward_size)
    if method == 'envelope':
       if name == 'cnn':
            return EnvelopeCnnCQN(state_size, action_size, reward_size)
    else:
        print("model %s doesn't exist." % (name))
        return None
