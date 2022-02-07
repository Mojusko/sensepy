import torch
from typing import Callable, Type, Union, Tuple, List
from stpy.borel_set import BorelSet,HierarchicalBorelSets
from stpy.kernels import KernelFunction
from stpy.point_processes.poisson_rate_estimator import PoissonRateEstimator
from stpy.point_processes.poisson.poisson import PoissonPointProcess
from sensepy.sense_entropy import SenseEntropy

class Top2Thompson(SenseEntropy):

	def __init__(self, *args,**kwargs):
		"""
		Implements the top-2 algorithm for level set/maximum identification
		:param args:
		:param kwargs: see parent
		"""
		super().__init__(*args,**kwargs)
		self.t = 0
		self.topk = 1

	def acquisition_function(self, actions: List):
		"""
		calculate the acqusition function for the top2 algorithm for the given actions
		:param actions:
		:return:
		"""
		scores = torch.zeros(len(actions)).double()
		diff = False

		while diff == False:

			self.estimator.sample()
			path1 = self.estimator.sample_path(self.D,self.n).clone()

			self.estimator.sample()
			path2 = self.estimator.sample_path(self.D,self.n).clone()

			if self.level_set is not None:
				R1 = path1 > self.level_set
				R2 = path2 > self.level_set
				difference = torch.abs(path1-path2)
				R = torch.logical_xor(R1.view(-1),R2.view(-1))
			else:
				x1 = torch.argmax(path1)
				x2 = torch.argmax(path2)
				R1 = torch.zeros(size = path1.view(-1).size()).bool()
				R2 = torch.zeros(size = path2.view(-1).size()).bool()
				R1[x1] = True
				R2[x2] = True
				R = torch.logical_xor(R1, R2)

			if torch.sum(R) > 1 or self.t == 0:
				diff = True
			else:
				print (self.t, "Resampling.")
				if self.level_set is not None:
					print ("Above:",torch.sum(R1)/float(len(R1)))
					print (path1.T)
					print ("Above:",torch.sum(R2)/float(len(R1)))
					print (path2.T)
				else:
					print ("Maxima values:",path1[x1],path2[x2])

		xtest = self.D.return_discretization(self.n)
		x = xtest[R]
		for id_action, action in enumerate(actions):
			if self.level_set is not None:
				diff = difference[R].view(-1)
				scores[id_action] = torch.sum(diff*action.is_inside(x).view(-1)) #+ (2*float(torch.randint(0,2, size =(1,1)))-1)/2.
			else:
				val = torch.sum(action.is_inside(x))
				if val > 0.5:
					val = val + torch.randint(1,100,size = (1,1)).double()
				scores[id_action] = val
		print (scores)
		return scores


	def sense(self, action:BorelSet)->Tuple[BorelSet,torch.Tensor,float]:
		"""
		Senses the action using the interval poisson process
		:param action:
		:return:
		"""
		obs = self.process.sample(action, dt = self.dt)
		print ("Sensing duration:",self.dt)
		self.t = self.t + 1
		return (action, obs, self.dt)

