import torch
import pickle
from abc import abstractmethod, ABC
import matplotlib.pyplot as plt
from typing import Callable, Type, Union, Tuple, List
from stpy.point_processes.poisson.poisson import PoissonPointProcess
from stpy.point_processes.poisson_rate_estimator import PoissonRateEstimator
from stpy.borel_set import HierarchicalBorelSets, BorelSet
from stpy.kernels import KernelFunction


class SpatialProblem(ABC):

	def __init__(self, m: int=50,
				 estimator_basis="triangle",
				 B : float = 10e06,
				 jitter: float = 1e-5,
				 d : int = 2):
		"""
		Implemenets abstract spatial problem class for benchamarks
		:param m: basis size for the approximation
		:param estimator:
		"""

		gamma = 0.15
		# domain
		self.D = BorelSet(2, bounds=torch.Tensor([[-1., 1.], [-1, 1]]).double())

		# hierarchical sets
		self.hs2d = HierarchicalBorelSets(d=d, interval=[(-1, 1), (-1, 1)], levels=5)
		k = KernelFunction(gamma=gamma, d=d)

		if estimator_basis == "psd":
			raise NotImplementedError("Not implemented yet")
		else:
			self.estimator = PoissonRateEstimator(None, self.hs2d, basis = estimator_basis,
												  d=d, kernel_object=k, B=B, m=m, jitter=jitter)

	@abstractmethod
	def load_data(self, prefix:str = '../data/'):
		pass

	def save_processed_data(self, filename):
		with open(filename, 'wb') as handle:
			pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)

	def load_processed_data(self, filename):
		with open(filename, 'rb') as handle:
			self.data = pickle.load(handle)

	def load_saved_model(self, name:Union[None,str]=None)->None:
		if name is None:
			theta = torch.load('model.pt')
		else:
			theta = torch.load(name)
		self.estimator.rate = theta
		print("Model loaded.")

	def save_model(self, name:Union[None,str]=None)->None:
		if name is None:
			torch.save(self.estimator.rate, 'model.pt')
		else:
			torch.save(self.estimator.rate, name)

	def rate(self, x: torch.Tensor)->torch.Tensor:
		"""
		gives rate evaluated at x
		:param x:
		:return:
		"""
		return self.estimator.rate_value(x)

	def return_process(self, n:int=50):
		"""
		:param n: discretization to define maximum and minimum
		:return:
		"""

		rate = lambda x: self.rate(x)
		rate_volume = lambda S: self.estimator.mean_set(S)

		xtest = self.D.return_discretization(n)
		vals = rate(xtest)

		b = torch.min(vals)
		B = torch.max(vals)

		print("Maximum: ", B, "minimum", b)
		P = PoissonPointProcess(d=2, B=B, b=b, rate=rate, rate_volume=rate_volume)
		print("Process created.")
		return P

	def plot_mean(self, estimator: PoissonRateEstimator, n:int = 50, show = True):
		F = lambda x: estimator.rate_value(x)
		from stpy.random_process import RandomProcess
		R = RandomProcess()
		xtest = self.D.return_discretization(n)
		R.visualize_function_contour(xtest, F, levels=20)
		obs = estimator.get_observations()
		plt.plot(obs[:, 0], obs[:, 1], 'ro', markersize=3)
		# plt.savefig(name,dpi=100,bbox_inches = 'tight',pad_inches = 0)
		if show:
			plt.show()

	def plot(self, name="name.png", show=True, n=40):
		F = lambda x: self.estimator.rate_value(x)
		from stpy.random_process import RandomProcess
		R = RandomProcess()
		xtest = self.D.return_discretization(n)
		R.visualize_function_contour(xtest, F, levels=20)
		plt.plot(self.obs[:, 0], self.obs[:, 1], 'ro', markersize=3)
		# plt.savefig(name,dpi=100,bbox_inches = 'tight',pad_inches = 0)
		if show:
			plt.show()
