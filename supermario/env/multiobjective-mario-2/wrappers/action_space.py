import gym

__all__ = [ 'ToDiscrete', 'ToBox' ]

def ToDiscrete():

    class ToDiscreteWrapper(gym.Wrapper):
        """
            Wrapper to convert MultiDiscrete action space to Discrete

            Only supports one config, which maps to the most logical discrete space possible
        """
        def __init__(self, env):
            super(ToDiscreteWrapper, self).__init__(env)
            mapping = {
                0: [0, 0, 0, 0, 0, 0],  # NOOP
                1: [1, 0, 0, 0, 0, 0],  # Up
                2: [0, 0, 1, 0, 0, 0],  # Down
                3: [0, 1, 0, 0, 0, 0],  # Left
                4: [0, 1, 0, 0, 1, 0],  # Left + A
                5: [0, 1, 0, 0, 0, 1],  # Left + B
                6: [0, 1, 0, 0, 1, 1],  # Left + A + B
                7: [0, 0, 0, 1, 0, 0],  # Right
                8: [0, 0, 0, 1, 1, 0],  # Right + A
                9: [0, 0, 0, 1, 0, 1],  # Right + B
                10: [0, 0, 0, 1, 1, 1],  # Right + A + B
                11: [0, 0, 0, 0, 1, 0],  # A
                12: [0, 0, 0, 0, 0, 1],  # B
                13: [0, 0, 0, 0, 1, 1],  # A + B
            }
            self.action_space = gym.spaces.multi_discrete.DiscreteToMultiDiscrete(self.action_space, mapping)
        def _step(self, action):
            return self.env._step(self.action_space(action))

    return ToDiscreteWrapper

def ToBox():

    class ToBoxWrapper(gym.Wrapper):
        """
            Wrapper to convert MultiDiscrete action space to Box

            Only supports one config, which allows all keys to be pressed
        """
        def __init__(self, env):
            super(ToBoxWrapper, self).__init__(env)
            self.action_space = gym.spaces.multi_discrete.BoxToMultiDiscrete(self.action_space)
        def _step(self, action):
            return self.env._step(self.action_space(action))

    return ToBoxWrapper
