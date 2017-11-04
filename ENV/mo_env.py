import numpy as np
from deep_sea_treasure import DeepSeaTreasure

class MultiObjectiveEnv(object):

	def __init__(self, env_name="deep_sea_treasure"):
		if env_name == "dst":
			self.env = DeepSeaTreasure()
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
	dst_env = MultiObjectiveEnv("dst")
	dst_env.reset()
	terminal = False
	print "DST STATE SPEC:", dst_env.state_spec
	print "DST ACTION SPEC:", dst_env.action_spec
	print "DST REWARD SPEC:", dst_env.reward_spec
	while not terminal:
		state = dst_env.observe()
		action = np.random.choice(4,1)[0]
		next_state, reward, terminal = dst_env.step(action)
		print "s:", state, "\ta:", action, "\ts':", next_state, "\tr:", reward
	print "AN EPISODE ENDS"
		
