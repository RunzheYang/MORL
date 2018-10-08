from __future__ import absolute_import, division, print_function
import numpy as np


class DeepSeaTreasure(object):

    def __init__(self):
        # the map of the deep sea treasure (convex version)
        self.sea_map = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0.7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [-10, 8.2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [-10, -10, 11.5, 0, 0, 0, 0, 0, 0, 0, 0],
             [-10, -10, -10, 14.0, 15.1, 16.1, 0, 0, 0, 0, 0],
             [-10, -10, -10, -10, -10, -10, 0, 0, 0, 0, 0],
             [-10, -10, -10, -10, -10, -10, 0, 0, 0, 0, 0],
             [-10, -10, -10, -10, -10, -10, 19.6, 20.3, 0, 0, 0],
             [-10, -10, -10, -10, -10, -10, -10, -10, 0, 0, 0],
             [-10, -10, -10, -10, -10, -10, -10, -10, 22.4, 0, 0],
             [-10, -10, -10, -10, -10, -10, -10, -10, -10, 23.7, 0]]
        )

        # DON'T normalize
        self.max_reward = 1.0

        # state space specification: 2-dimensional discrete box
        self.state_spec = [['discrete', 1, [0, 10]], ['discrete', 1, [0, 10]]]

        # action space specification: 1 dimension, 0 up, 1 down, 2 left, 3 right
        self.action_spec = ['discrete', 1, [0, 4]]

        # reward specification: 2-dimensional reward
        # 1st: treasure value || 2nd: time penalty
        self.reward_spec = [[0, 14], [-1, 0]]

        self.current_state = np.array([0, 0])
        self.terminal = False

    def get_map_value(self, pos):
        return self.sea_map[pos[0]][pos[1]]

    def reset(self):
        '''
            reset the location of the submarine
        '''
        self.current_state = np.array([0, 0])
        self.terminal = False

    def step(self, action):
        '''
            step one move and feed back reward
        '''
        dir = {
            0: np.array([-1, 0]),  # up
            1: np.array([1, 0]),  # down
            2: np.array([0, -1]),  # left
            3: np.array([0, 1])  # right
        }[action]
        next_state = self.current_state + dir

        valid = lambda x, ind: (x[ind] >= self.state_spec[ind][2][0]) and (x[ind] <= self.state_spec[ind][2][1])

        if valid(next_state, 0) and valid(next_state, 1):
            if self.get_map_value(next_state) != -1:
                self.current_state = next_state

        treasure_value = self.get_map_value(self.current_state)
        if treasure_value == 0 or treasure_value == -1:
            treasure_value = 0.0
        else:
            treasure_value /= self.max_reward
            self.terminal = True
        time_penalty = -1.0 / self.max_reward
        reward = np.array([treasure_value, time_penalty])

        return self.current_state, reward, self.terminal
