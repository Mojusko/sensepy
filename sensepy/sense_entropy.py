import torch
import numpy as np
import matplotlib.pyplot as plt
from sensepy.capture_ucb import SensingAlgorithm
from stpy.borel_set import BorelSet,HierarchicalBorelSets
from stpy.kernels import KernelFunction
from stpy.point_processes.poisson_rate_estimator import PositiveRateEstimator
from stpy.point_processes.poisson import PoissonPointProcess
#from multiprocessing import Pool

import copy

class SenseEntropy(SensingAlgorithm):

	def __init__(self, process, estimator,w,D,n,initial_data = None,
				 dt = 10., level_set = None, design_type = "D", repeats = 10, beta = 3):
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

	def get_ucb_lcb(self, D, n):
		if self.approx is None:
			self.mean, self.lcb, self.ucb = self.estimator.map_lcb_ucb(D, n, beta=self.beta)
			return self.mean, self.lcb, self.ucb

		elif self.approx == "ellipsoid" and self.estimator.data is not None:
			self.mean, self.lcb, self.ucb = self.estimator.map_lcb_ucb_approx(D, n)
			return self.mean, self.lcb, self.ucb

		else:
			self.mean, self.lcb, self.ucb = self.estimator.map_lcb_ucb(D, n, beta=self.beta)
			return self.mean, self.lcb, self.ucb

	def get_region(self,D,n):
		mean, lcb, ucb = self.get_ucb_lcb(D,n)

		if self.level_set is not None:
			R = ucb > self.level_set
			R2 = lcb < self.level_set
			R = R*R2
		else:
			R = ucb > torch.max(lcb)

		return R.view(-1)

	def fit_estimator(self):
		self.estimator.fit_gp()

	def classify_region(self,D,n):
		mean, lcb, ucb = self.get_ucb_lcb(D,n)

		above = lcb > self.level_set
		below = ucb < self.level_set

		not_known = above*False + True
		not_known = not_known*(~ above)*(~ below)
		return (above.view(-1),not_known.view(-1),below.view(-1))


	def classify_region_map(self,D,n):
		xtest = D.return_discretization(n)
		mean = self.estimator.rate_value(xtest)

		above = mean > self.level_set
		below = mean < self.level_set
		return (above.view(-1),below.view(-1))


	def true_classify_region(self,D,n):
		xtest = D.return_discretization(n)
		rate = self.process.rate(xtest)
		above = rate > self.level_set
		below = rate < self.level_set
		return (above.view(-1), below.view(-1))


	def per_action_acquisiton_function_V(self, action, threads = 0):
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
				Phi = self.estimator.packing.embed(obs)
				norm = self.estimator.mean_rate_points(obs)

				U = torch.diag(1. / np.sqrt(torch.abs(norm.view(-1)) + 10e-5)) @ Phi
				Sigma = self.Phi_R.T @ self.Phi_R @ \
						(self.invSigma - self.invSigma @ U.T @ torch.pinverse(
							torch.eye(U.size()[0]).double() + U @ self.invSigma @ U.T) @ U @ self.invSigma)

				numerator = numerator + torch.trace(Sigma)
			else:
				numerator = numerator

		return -numerator / (self.repeats * cost)

	def per_action_acquisiton_function_V_approx(self, action):
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
				Phi = self.estimator.packing.embed(obs)
				norm = self.estimator.mean_rate_points(obs)
				U = torch.diag(1./np.sqrt(torch.abs(norm.view(-1))+10e-5)) @ Phi


				Sigma = self.Phi_R.T @ self.Phi_R @ ( self.invSigma @ U.T @ torch.pinverse(
							torch.eye(U.size()[0]).double() + U @ self.invSigma @ U.T) @ U @ self.invSigma)

				numerator = numerator + torch.trace(Sigma)
			else:
				numerator = numerator

		return  numerator/(self.repeats*cost)



	def acquisition_function(self, actions):

		scores = torch.zeros(len(actions)).double()

		if self.design_type != "Random":
			R = self.get_region(self.D,self.n)
			xtest = self.D.return_discretization(self.n)
			self.Phi = self.estimator.packing.embed(xtest).double()
			self.Phi_R = self.Phi[R.view(-1),:]

			self.Sigma = self.estimator.construct_covariance_matrix_laplace()
			self.invSigma = torch.pinverse(self.Sigma)
			self.theta = self.estimator.rate

			ucb = self.estimator.map_lcb_ucb_approx(self.D,self.n)[2]
			self.new_data = []
			surogate_process = PoissonPointProcess(d = self.process.d, B = self.process.B, b= self.process.b)
			for i in range(self.repeats):
				self.new_data.append(surogate_process.sample_discretized_direct(xtest,ucb))

		for id_action, action in enumerate(actions):
			if self.design_type == "V":
				scores[id_action] = self.per_action_acquisiton_function_V(action,threads=4)
			elif  self.design_type == "V-approx":
				scores[id_action] = self.per_action_acquisiton_function_V_approx(action)
			elif self.design_type == "Random":
				scores[id_action] = np.random.randn()
			else:
				raise AssertionError("Acquisiton Function not implemented.")
		return scores

	def add_data(self,data_point):
		self.estimator.add_data_point(data_point)


if __name__ == "__main__":
	d = 1
	gamma = 0.1
	n = 64
	B = 4.
	b = 1.
	m = 128

	process = PoissonPointProcess(d=1, B=B, b=b)
	Sets = []
	levels = 3
	hierarchical_structure = HierarchicalBorelSets(d=1, interval=(-1, 1), levels=levels)
	Sets = hierarchical_structure.get_all_sets()

	D = BorelSet(1, bounds=torch.Tensor([[-1., 1.]]).double())

	k = KernelFunction(gamma=gamma, kappa=B).kernel
	estimator = PositiveRateEstimator(process, hierarchical_structure, kernel=k, B=B + b, b = b, m=m, jitter=10e-4, approx="ellipsoid")

	min_vol, max_vol = estimator.get_min_max()
	dt = (b)/min_vol
	dt = 200*dt
	xtest = D.return_discretization(n = n)

	w = lambda s: s.volume()*1000
	data = []
	Bandit = SenseEntropy(process,estimator,w,D,n, initial_data = data,
						  dt = dt, level_set = None, repeats=1, design_type="V-approx")
	T = 50
	plt.ion()
	rate = process.rate(xtest)

	for t in range(T):
		Bandit.fit_estimator()
		rate_mean = Bandit.estimator.mean_rate(D, n=n)
		scores = Bandit.acquisition_function(Sets)

		plt.clf()
		#plt.plot(Bandit.data[-1][1],Bandit.data[-1][1]*0,'ko')
		for index_set,score in enumerate(scores):
			xx = Sets[index_set].return_discretization(n)
			plt.plot(xx,score+xx*0,'g')

		max_set = Sets[torch.argmax(scores)]
		xx = max_set.return_discretization(n)
		if Bandit.level_set is not None:
			above,not_known,below = Bandit.classify_region(D,n)

			plt.plot(xtest[above],xtest[above]*0 + 1,'r*-', label = 'region_of_interest - above')
			plt.plot(xtest[not_known],xtest[not_known]*0 , 'r*-', label='region_of_interest - not known')
			plt.plot(xtest[below], xtest[below]*0 - 1, 'r*-', label='region_of_interest - below')
		else:
			x = Bandit.best_point_so_far(D,n)
			plt.plot(x,1,'ro', label = "best-point")
			print ("Simple regret",torch.max(process.rate(xtest)) - process.rate(x) )
			R = Bandit.get_region(D,n)

			plt.plot(xtest[R],xtest[R]*0,'r*-', label = 'region_of_interest')

		plt.plot(xx, torch.max(scores) + xx * 0, "--", color='yellow', lw=2)
		plt.plot(xtest,rate_mean,color = 'blue', label = 'likelihood estimate') # normalized to dt = 1
		plt.plot(xtest, rate,color = 'orange', label='rate', lw=3)

		plt.plot(xtest, xtest*0 + 2, "k--")

		_ , lcb, ucb =  Bandit.estimator.map_lcb_ucb(D, n, beta = 3.)
		plt.fill_between(xtest.numpy().flatten(), lcb.numpy().flatten(), ucb.numpy().flatten(), alpha = 0.4, color = 'gray', label = "uncertainty")

		plt.draw()
		plt.legend()
		plt.pause(0.5)


		Bandit.step(Sets)
		print (t)

