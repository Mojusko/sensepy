import torch
import numpy as np
from typing import Callable, Type, Union, Tuple, List
import matplotlib.pyplot as plt
from sensepy.capture_ucb import SensingAlgorithm
from stpy.borel_set import BorelSet, HierarchicalBorelSets
from stpy.kernels import KernelFunction
from stpy.point_processes.poisson_rate_estimator import PoissonRateEstimator
from stpy.point_processes.poisson.poisson import PoissonPointProcess


class SenseEntropy(SensingAlgorithm):

	def __init__(self,
				 process: Union[PoissonPointProcess,None], #inherent process
				 estimator: PoissonRateEstimator, # estimator
				 w: Callable, # cost function
				 D: BorelSet,
				 n: int, # discretization of D
				 initial_data = None,
				 dt: float = 10., #time duration
				 level_set: Union[None,float] = None,
				 design_type: str = "V", #type of the design
				 repeats: int = 3, # number of samples from the integral to approximate integration over posterior
				 beta: float = 3) -> None: # confidence parameter used for ucb,lcb calculation
		"""
		Algorithm that executes classical experiment design pipeline
		in order to learn level/sets and/or maxima.  Level sets specified by the flag, if None, maxima is
		searched for.
		"""
		super().__init__(process, w, dt)
		self.process = process
		self.estimator = estimator
		self.dt = dt
		self.w = w
		self.data = initial_data
		self.level_set = level_set
		self.beta = beta
		self.repeats = repeats
		self.design_type = design_type
		self.n = n
		self.D = D
		self.approx = self.estimator.approx
		self.topk = 1

	def get_ucb_lcb(self, D: BorelSet, n: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		Gets ucb,lcb over the set D with discretization n, it uses internal rate estimator
		:param D: borel set where want to evaluate ucb, lcb and mean
		:param n: discretization of the borel set
		:return: (mean, lcb, ucb) over D
		"""
		if self.approx is None:
			self.mean, self.lcb, self.ucb = self.estimator.map_lcb_ucb(D, n, beta=self.beta)
			return self.mean, self.lcb, self.ucb

		elif self.approx == "ellipsoid" and self.estimator.data is not None:
			self.mean, self.lcb, self.ucb = self.estimator.map_lcb_ucb_approx(D, n)
			return self.mean, self.lcb, self.ucb

		else:
			self.mean, self.lcb, self.ucb = self.estimator.map_lcb_ucb(D, n, beta=self.beta)
			return self.mean, self.lcb, self.ucb

	def get_region(self, D: BorelSet, n: int) -> torch.Tensor:
		"""
		Gives a region of interest, i.e. where we still cannot clasify level sets, maxima
		:param D:
		:param n:
		:return: torch mask in 1D
		"""
		mean, lcb, ucb = self.get_ucb_lcb(D, n)

		if self.level_set is not None:
			R = ucb > self.level_set
			R2 = lcb < self.level_set
			R = R * R2
		else:
			R = ucb > torch.max(lcb)

		return R.view(-1)

	def fit_estimator(self) -> None:
		"""
		fit the internal estimator
		"""
		self.estimator.fit_gp()

	def classify_region(self, D: BorelSet, n: int)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
		"""
		Method classifies +,-1 the sets as below and above using the current lcb, ucb estimates
		:return: above , not_known, below
		"""
		mean, lcb, ucb = self.get_ucb_lcb(D, n)

		above = lcb > self.level_set
		below = ucb < self.level_set

		not_known = above * False + True
		not_known = not_known * (~ above) * (~ below)
		return (above.view(-1), not_known.view(-1), below.view(-1))

	def classify_region_map(self, D: BorelSet, n:int)->Tuple[torch.Tensor,torch.Tensor]:
		"""
		Method classifies +,-1 the sets as below and above using the current map estimate
		:return: above, below
		"""
		xtest = D.return_discretization(n)
		mean = self.estimator.rate_value(xtest)

		above = mean > self.level_set
		below = mean < self.level_set
		return (above.view(-1), below.view(-1))

	def true_classify_region(self, D:BorelSet, n:int)->Tuple[torch.Tensor,torch.Tensor]:
		"""
		Uses the internal poisson process rate [if given] to classify the level sets
		:return:
		"""
		xtest = D.return_discretization(n)
		rate = self.process.rate(xtest)
		above = rate > self.level_set
		below = rate < self.level_set
		return (above.view(-1), below.view(-1))

	def per_action_acquisiton_function_V(self, action: BorelSet)->float:
		"""
		Calculate the value of acquisiton function for V-experimental design.
		:param action: Borelset
		:return: return the value
		"""
		cost = self.w(action)
		numerator = 0.
		jitter = 10e-5

		for i in range(self.repeats):

			if self.new_data[i] is not None:
				mask = action.is_inside(self.new_data[i])
				obs = self.new_data[i][mask, :]
				if obs.size()[0] == 0:
					obs = None
			else:
				obs = None

			if obs is not None:
				Phi = self.estimator.packing.embed(obs)
				norm = self.estimator.mean_rate_points(obs)
				U = torch.diag(1. / np.sqrt(torch.abs(norm.view(-1)) + jitter)) @ Phi
				Sigma = self.Phi_R.T @ self.Phi_R @ \
						(self.invSigma - self.invSigma @ U.T @ torch.pinverse(
							torch.eye(U.size()[0]).double() + U @ self.invSigma @ U.T) @ U @ self.invSigma)

				numerator = numerator + torch.trace(Sigma)
			else:
				numerator = numerator

		return -numerator / (self.repeats * cost)

	def per_action_acquisiton_function_V_approx(self, action):
		"""
		Calculate the value of acquisiton function for V-experimental design,
		which uses of inversion lemma (essentially same as the other)
		:param action: Borelset
		:return: return the value
		"""
		cost = self.w(action)
		numerator = 0.

		for i in range(self.repeats):
			if self.new_data[i] is not None:
				mask = action.is_inside(self.new_data[i])
				obs = self.new_data[i][mask, :]
				if obs.size()[0] == 0:
					obs = None
			else:
				obs = None

			if obs is not None:
				Phi = self.newPhi[i][mask, :]  # estimator.packing.embed(obs)
				norm = self.estimator.mean_rate_points(obs)
				U = torch.diag(1. / np.sqrt(torch.abs(norm.view(-1)) + 10e-5)) @ Phi

				Sigma = self.Phi_R.T @ self.Phi_R @ (self.invSigma @ U.T @ torch.pinverse(
					torch.eye(U.size()[0]).double() + U @ self.invSigma @ U.T) @ U @ self.invSigma)

				numerator = numerator + torch.trace(Sigma)
			else:
				numerator = numerator

		return numerator / (self.repeats * cost)


	def acquisition_function(self, actions: List)->torch.Tensor:
		"""
		Calculates the acquisiton function over actions using the specified strategy
		:param actions: list of actions (Borelsets)
		:return: returns 1D array of scores corresponding to acquisiton function of that action
		"""
		scores = torch.zeros(len(actions)).double()

		if self.design_type != "Random":  # if not random do the below

			R = self.get_region(self.D, self.n)
			xtest = self.D.return_discretization(self.n)
			self.Phi = self.estimator.packing.embed(xtest).double()
			self.Phi_R = self.Phi[R.view(-1), :]

			self.Sigma = self.estimator.construct_covariance_matrix_laplace()
			self.invSigma = torch.pinverse(self.Sigma)
			self.theta = self.estimator.rate

			ucb = self.estimator.map_lcb_ucb_approx(self.D, self.n)[2]
			self.new_data = []
			self.newPhi = []
			surogate_process = PoissonPointProcess(d=self.process.d, B=self.process.B, b=self.process.b)

			for i in range(self.repeats):
				obs = surogate_process.sample_discretized_direct(xtest, ucb)
				self.new_data.append(obs)
				self.newPhi.append(self.estimator.packing.embed(obs))

		for id_action, action in enumerate(actions):
			if self.design_type == "V":
				scores[id_action] = self.per_action_acquisiton_function_V(action)

			elif self.design_type == "V-approx":
				scores[id_action] = self.per_action_acquisiton_function_V_approx(action)

			elif self.design_type == "Random":
				scores[id_action] = np.random.randn()
			else:
				raise NotImplementedError("The requested acquisiton function is not implemented.")
		return scores

	def add_data(self, data_point:Tuple[BorelSet,torch.Tensor,float])->None:
		"""
		Add data to the internal estimator
		:param data_point: data point in form (S,obs,delta_t)
		"""
		self.estimator.add_data_point(data_point)

