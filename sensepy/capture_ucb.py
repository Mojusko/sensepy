import torch
import matplotlib.pyplot as plt
from typing import Callable, Type, Union, Tuple, List
from stpy.borel_set import BorelSet,HierarchicalBorelSets
from stpy.kernels import KernelFunction
from stpy.point_processes.poisson_rate_estimator import PoissonRateEstimator
from stpy.point_processes.poisson.poisson import PoissonPointProcess
from sensepy.sense import SensingAlgorithm

class CaptureUCB(SensingAlgorithm):

	def __init__(self,
				 process: PoissonPointProcess,
				 estimator: PoissonRateEstimator,
				 w: Callable,
				 initial_data,
				 dt: float = 10.,
				 topk:int = 1):
		"""
		Create a handler for Capture-UCB Algorithm of Mutny & Krause 2021.
		:param process: internal ref to poisson process
		:param estimator: internal ref to estimator
		:param w: cost function
		:param initial_data:
		:param dt: sensign time
		:param topk: number of elements played in single round
		"""
		self.process = process
		self.estimator = estimator
		self.dt = dt
		self.w = w
		self.data = initial_data
		self.estimator.load_data(self.data)
		self.topk = topk

	def fit_estimator(self)->None:
		"""
		Fits the estimator and update the variance of it
		:return:
		"""
		self.estimator.update_variances()
		self.estimator.fit_gp()

	def acquisition_function(self, actions: List)->torch.Tensor:
		"""
		Caculates the acqusition function
		:param actions:
		:return:
		"""
		scores = []
		for action in actions:
			scores.append(self.estimator.ucb(action, dt = self.dt)/self.w(action))
		return torch.Tensor(scores).double()

	def add_data(self,data_point: Tuple[BorelSet,torch.Tensor,float])->None:
		"""
		Add the datapoint and updates the internal estimaotr
		:param data_point:
		"""
		self.estimator.add_data_point(data_point)
		self.data.append(data_point)

