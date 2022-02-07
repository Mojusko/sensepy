import torch
from typing import Callable, Type, Union, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from stpy.borel_set import BorelSet, HierarchicalBorelSets
from stpy.kernels import KernelFunction
from stpy.point_processes.poisson_rate_estimator import PoissonRateEstimator
from stpy.point_processes.poisson.poisson import PoissonPointProcess


class SensingAlgorithm:
	"""
	Sensing algorithm main class
	"""

	def __init__(self,
				 process: PoissonPointProcess,
				 w: Callable[..., float],
				 dt: float = 1.,
				 topk: int = 1) -> None:
		"""

		:param process: the poisson process class
		:param w: cost function evaluating an action
		:param dt: the sensing duration
		:param topk: number of top arms
		"""
		self.process = process
		self.w = w
		self.dt = dt
		self.topk = topk

	def sense(self, action: BorelSet) -> (BorelSet, torch.tensor, float):
		"""
		Senses a action for default duration
		:param action: Borel set
		:return:
		"""
		obs = self.process.sample(action, dt=self.dt)
		return (action, obs, self.dt)

	def acquisition_function(self, actions: list) -> torch.tensor:
		"""
		Calculates the acquisition function for the actions

		:param actions:
		:return:
		"""
		return torch.randn(size=(len(actions), 1)).double()

	def fit_estimator(self):
		"""
		Fits the estimator
		:return:
		"""
		pass

	def count_events(self) -> int:
		"""
		Counts the number of events
		:return:
		"""
		total = 0
		for i in self.data:
			if i[1] is not None:
				total += i[1].size()[0]
		return total

	def step(self, actions: List,
			 verbose: bool = False,
			 points: bool = False):
		"""

		:param actions:
		:param verbose: levle of verbosity
		:param points:
		:return:
		"""

		# fits the estimator
		self.fit_estimator()

		# acquisiton function
		scores = self.acquisition_function(actions)

		# picks the actions
		best_regions, best_indices = torch.topk(scores, self.topk)

		if self.topk == 1:
			sensed_actions = [actions[best_indices]]
		else:
			sensed_actions = [actions[i] for i in best_indices]

		if verbose == True:
			print("Scores:", scores)
			print("Sensing:", [action.description() for action in sensed_actions])
			print("Total events so far:", self.count_events())

		# sense
		data_point = []
		cost = 0
		points_loc = None
		for action in sensed_actions:
			data_point = self.sense(action)
			self.add_data(data_point)
			cost += self.w(data_point[0])

			if points_loc is None and data_point[1] is not None:
				points_loc = data_point[1]
			elif points_loc is not None and data_point[1] is not None:
				points_loc = torch.cat((points_loc, data_point[1]), dim=0)

		if points == False:
			if points_loc is not None:
				return (cost, points_loc.size()[0], sensed_actions, best_indices)
			else:
				return (cost, 0, sensed_actions, best_indices)
		else:
			if points_loc is not None:
				return (cost, points_loc, sensed_actions, best_indices)
			else:
				return (cost, None, sensed_actions, best_indices)

	def best_point_so_far(self,
						  D : BorelSet,
						  n : int)->torch.Tensor:
		"""
		return the argmax given the current estimator
		:param D:
		:param n:
		:return:
		"""
		xx = D.return_discretization(n)
		map = self.estimator.mean_rate(D, n)
		return xx[torch.argmax(map), :].view(1, -1)


class EpsilonGreedySense(SensingAlgorithm):

	def __init__(self,
				 process: PoissonPointProcess,
				 estimator: PoissonRateEstimator,
				 w: Callable,
				 initial_data = None,
				 epsilon: Callable = lambda t: 1. - 1. / np.sqrt(t), # eps-probability decay
				 dt: float = 10.,
				 topk: int = 1) ->None:
		"""
		Construct a class for Epsilon greedy algorithm
		:param process:
		:param estimator:
		:param w:
		:param initial_data:
		:param epsilon:
		:param dt:
		:param topk:
		"""
		super().__init__(process, w, dt, topk)
		self.process = process
		self.estimator = estimator
		self.dt = dt
		self.w = w
		self.data = initial_data
		self.estimator.load_data(self.data)
		self.epsilon = epsilon
		self.topk = topk
		self.t = 0

	def fit_estimator(self):
		"""
		Fits the estimator
		:return:
		"""
		self.estimator.fit_gp()

	def acquisition_function(self, actions: List)->torch.Tensor:
		"""
		Acquisiton function
		:param actions:
		:return:
		"""
		self.t = self.t + 1
		p = np.random.uniform(0, 1)
		scores = []
		epsilon = self.epsilon(self.t)
		if p > epsilon:
			# exploitation
			if self.estimator.rate is not None:
				for action in actions:
					map = self.estimator.mean_set(action)
					scores.append(float(map / self.w(action)))
			else:
				for action in actions:
					scores.append(np.random.randn())
		else:
			# random exploration
			for action in actions:
				scores.append(np.random.randn())

		return torch.Tensor(scores).double()

	def add_data(self, data_point: Tuple)->None:
		"""
			Add the data to the estimator
		:param data_point:
		:return:
		"""
		self.estimator.add_data_point(data_point)
		self.data.append(data_point)

