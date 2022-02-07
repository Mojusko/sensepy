import torch
import matplotlib.pyplot as plt
from stpy.borel_set import BorelSet,HierarchicalBorelSets
from stpy.kernels import KernelFunction
from typing import Callable, Type, Union, Tuple, List
from stpy.point_processes.poisson_rate_estimator import PoissonRateEstimator
from stpy.point_processes.poisson.poisson import PoissonPointProcess
from sensepy.capture_ucb import CaptureUCB
import numpy as np
from scipy import optimize

class CaptureIDS(CaptureUCB):

	def __init__(self,
				 *args,
				 actions = None,
				 original_ids = True, # original or using experimental precomputaiton
				 **kwargs)->None:
		"""
		Create IDS algorithm for poisson sesning of Mutny & Krause (2021)
		:param args: see parent class
		:param actions: set of actions to work with
		:param original_ids:
		:param kwargs:
		"""
		super().__init__(*args,**kwargs)
		self.precomputed = None
		self.original = original_ids
		if actions is not None:
			self.precomputed = {}
			for S in actions:
				ind = []
				for index, set in enumerate(self.estimator.basic_sets):
					if S.inside(set):
						ind.append(index)
				Upsilon = self.estimator.varphis[ind, :]
				self.precomputed[S] = Upsilon

	def acquisition_function(self, actions: List)->torch.Tensor:
		"""
		Calculate the acqusition function for Capture IDS without optimization
		:param actions:
		:return:
		"""
		if self.original == True:
			return self.acquisition_function_original(actions)
		else:
			self.estimator.ucb_identified = False
			gaps = [self.estimator.gap(action, actions, self.w, dt=self.dt) for action in actions]
			inf = [self.estimator.information(action, self.dt, precomputed=self.precomputed) for action in actions]
			gaps = np.array(gaps)
			inf  = np.array(inf)
			index = np.argmin((gaps**2)/inf)
			return index

	def acquisition_function_original(self, actions):
		"""
		Calculate the acqusition function for Capture IDS with optimized distribution
		:param actions:
		:return:
		"""
		gaps = []
		inf = []
		self.estimator.ucb_identified = False

		for action in actions:
			gaps.append(self.estimator.gap(action,actions,self.w, dt = self.dt))
			inf.append(self.estimator.information(action,self.dt,precomputed=self.precomputed))

		gaps = np.array(gaps)
		inf  = np.array(inf)

		index1 = np.argmin(gaps)
		index2 = np.argmin(gaps/inf)

		gaps_squared = gaps**2

		ratio = lambda p:  (gaps_squared[index1]*p + gaps_squared[index2]*(1-p))/(inf[index1]*p + inf[index2]*(1-p))
		res = optimize.minimize_scalar(ratio, bounds = (0,1), method = "bounded")
		p = res.x

		if np.random.uniform() < p:
			print ("greedy.")
			return index1
		else:
			print ("informative.")
			return index2




	def step(self, actions,
			 verbose: bool = False,
			 points: bool = False):
		"""

		:param actions: set of actions
		:param verbose: verobiste level (T/F)
		:param points: returns also location of the points (T/F)
		:return: see parent class
		"""
		self.fit_estimator()

		# acquisiton function
		best_region = self.acquisition_function(actions)
		best_indices = [best_region]
		if verbose == True:
			print ("Sensing:", actions[best_region].bounds)

		sensed_actions = [actions[best_region]]
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

