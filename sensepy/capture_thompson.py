import torch
import matplotlib.pyplot as plt
from typing import Callable, Type, Union, Tuple, List
from stpy.borel_set import BorelSet,HierarchicalBorelSets
from stpy.kernels import KernelFunction
from stpy.point_processes.poisson_rate_estimator import PoissonRateEstimator
from stpy.point_processes.poisson.poisson import PoissonPointProcess
from sensepy.capture_ucb import CaptureUCB

class CaptureThompson(CaptureUCB):
	"""
	Create Cox-Thompson sampling algorithm of Mutny & Krause (2022) using
	"""

	def acquisition_function(self, actions: List)->torch.Tensor:
		"""
		Calculates the acquisiton function for the actions
		:return scores
		"""
		scores = []
		self.estimator.sample()
		for action in actions:
			score = self.estimator.sample_value(action)*self.dt/self.w(action)
			scores.append(score)
		return torch.Tensor(scores).double()

