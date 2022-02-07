import torch
from typing import Callable, Type, Union, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from stpy.borel_set import BorelSet,HierarchicalBorelSets
from stpy.point_processes.poisson.poisson import PoissonPointProcess
from sensepy.sense import SensingAlgorithm
import scipy.stats

class GammaPriorThopsonSampling(SensingAlgorithm):

	def __init__(self,
				process: PoissonPointProcess,
				basic_sets,
				w,
				initial_data =None,
				dt: float=10.,
				alpha: float =1.,
				beta: float=1.,
				topk: int =1):
		"""
		Implements the method of Grant (2019-2020) which uses Gamma prior for discretized subsets of Borelsets.
		Implemented only for 1D intervals.

		:param process: poisson process implementing sample method
		:param basic_sets: discretization of the domain to small sets
		:param w:  cost function
		:param initial_data: intiial data
		:param dt: sensing duration
		:param alpha: prior of gamma distribution params
		:param beta:  prior of gamma distribution params
		:param topk: as in constructor
		"""
		super().__init__(process, w, dt, topk)
		self.process = process
		self.dt = dt
		self.w = w
		self.data = initial_data
		self.alphas = torch.ones(len(basic_sets)).double()*alpha
		self.betas = torch.ones(len(basic_sets)).double()*beta
		self.basic_sets = basic_sets
		self.topk = topk

		data_basic = [[] for _ in range(len(basic_sets))]
		sensing_times = [[] for _ in range(len(basic_sets))]
		counts = torch.zeros(len(basic_sets)).int()
		total_data = 0
		# use initial data
		for sample in self.data:
			S, obs, dt = sample
			if obs is not None:
				total_data = total_data + obs.size()[0]
				for index, elementary in enumerate(self.basic_sets):
					mask = elementary.is_inside(obs)
					if S.inside(elementary) == True:
						data_basic[index].append(obs[mask])
						counts[index] += 1
					sensing_times[index].append(dt)
			else:
				for index, elementary in enumerate(self.basic_sets):
					if S.inside(elementary) == True:
						data_basic[index].append(torch.Tensor([]))
						counts[index] += 1
					sensing_times[index].append(dt)

		for index, elementary in enumerate(self.basic_sets):
			arr = np.array([elem.size()[0] for elem in data_basic[index]])
			self.alphas[index] += np.sum(arr)
			self.betas[index] += arr.shape[0]


	def fit_estimator(self)->None:
		pass

	def add_data(self,
				 new_data: Tuple[BorelSet,torch.Tensor,float]):
		"""
		Adds a data point
		:param new_data:
		:return:
		"""
		index = 0

		for set in self.basic_sets:
			if set.inside(new_data[0]):
				break
			index +=1
		if new_data[1] is not None:
			self.alphas[index] += new_data[1].size()[0]
		self.betas[index] += 1

		self.data.append(new_data)

	def acquisition_function(self,actions: List):
		"""
		Calculate the acquisition function
		:param actions:
		:return:
		"""

		# decide whether explore or exploit
		scores_basic = []
		for id_region, region in enumerate(self.basic_sets):
			scores_basic.append(scipy.stats.gamma.rvs(self.alphas[id_region],scale = 1./self.betas[id_region]))
		scores = []

		for action in actions:
			score = 0.
			for id_region, region in enumerate(self.basic_sets):
				if action.inside(region) == True:
					score = float(score) + float(scores_basic[id_region])
			scores.append(score/float(self.w(action)))

		return torch.Tensor(scores).double()

