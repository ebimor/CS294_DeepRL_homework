import numpy as np
from cost_functions import trajectory_cost_fn
import time

class Controller():
	def __init__(self):
		pass

	# Get the appropriate action(s) for this state(s)
	def get_action(self, state):
		pass


class RandomController(Controller):
	def __init__(self, env):
		""" YOUR CODE HERE """
		self.env = env
		#pass

	def get_action(self, state):
		""" YOUR CODE HERE """
		""" Your code should randomly sample an action uniformly from the action space """
		return self.env.action_space.sample()


class MPCcontroller(Controller):
	""" Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """
	def __init__(self,
				 env,
				 dyn_model,
				 horizon=5,
				 cost_fn=None,
				 num_simulated_paths=10,
				 ):
		self.env = env
		self.dyn_model = dyn_model
		self.horizon = horizon
		self.cost_fn = cost_fn
		self.num_simulated_paths = num_simulated_paths

	def get_action(self, state):
		""" YOUR CODE HERE """
		""" Note: be careful to batch your simulations through the model for speed """

		states, next_states, inputs = [], [], []
		observations = [state for _ in range(self.num_simulated_paths)]

	    for it in range(self.horizon):
			actions =  [self.env.action_space.sample() for _ in range(self.num_simulated_paths)]
			next_observations = self.dyn_model.predict(observations, actions)
			states.append(observations)
			next_states.append(next_observations)
			inputs.append(actions)
			observations = next_observations

		costs = trajectory_cost_fn(self.cost_fn, states, inputs, next_states)
		best_action_index = np.argmin(costs)
		return inputs[costs,0]
