from __future__ import absolute_import, division, print_function
import numpy as np
from .deep_sea_treasure import DeepSeaTreasure
from .fruit_tree import FruitTree


class MultiObjectiveEnv(object):

    def __init__(self, env_name="deep_sea_treasure"):
        if env_name == "dst":
            self.env = DeepSeaTreasure()
            self.state_spec = self.env.state_spec
            self.action_spec = self.env.action_spec
            self.reward_spec = self.env.reward_spec
        if env_name == "ft":
            self.env = FruitTree()
            self.state_spec = self.env.state_spec
            self.action_spec = self.env.action_spec
            self.reward_spec = self.env.reward_spec
        if env_name == "ft5":
            self.env = FruitTree(5)
            self.state_spec = self.env.state_spec
            self.action_spec = self.env.action_spec
            self.reward_spec = self.env.reward_spec
        if env_name == "ft7":
            self.env = FruitTree(7)
            self.state_spec = self.env.state_spec
            self.action_spec = self.env.action_spec
            self.reward_spec = self.env.reward_spec

    def reset(self, env_name=None):
        ''' reset the enviroment '''
        self.env.reset()

    def observe(self):
        ''' reset the enviroment '''
        return self.env.current_state

    def step(self, action):
        ''' process one step transition (s, a) -> s'
            return (s', r, terminal)
        '''
        return self.env.step(action)


if __name__ == "__main__":
    '''
        Test ENVs
    '''
    dst_env = MultiObjectiveEnv("ft7")
    dst_env.reset()
    terminal = False
    print("DST STATE SPEC:", dst_env.state_spec)
    print("DST ACTION SPEC:", dst_env.action_spec)
    print("DST REWARD SPEC:", dst_env.reward_spec)
    while not terminal:
        state = dst_env.observe()
        action = np.random.choice(2, 1)[0]
        next_state, reward, terminal = dst_env.step(action)
        print("s:", state, "\ta:", action, "\ts':", next_state, "\tr:", reward)
    print("AN EPISODE ENDS")
