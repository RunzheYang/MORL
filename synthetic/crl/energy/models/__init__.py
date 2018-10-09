from .linear import EnergyLinearCQN

def get_new_model(name, state_size, action_size, reward_size):
	if name == 'linear':
		return EnergyLinearCQN(state_size, action_size, reward_size)
	else:
		print("model %s doesn't exist."%(name))
		return None