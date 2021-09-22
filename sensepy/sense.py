import torch
import numpy as np
import matplotlib.pyplot as plt
from stpy.borel_set import BorelSet,HierarchicalBorelSets
from stpy.kernels import KernelFunction
from stpy.point_processes.poisson_rate_estimator import PositiveRateEstimator
from stpy.point_processes.poisson import PoissonPointProcess


class SensingAlgorithm:

	def __init__(self, process,w, dt=1, topk = 1):
		self.process = process
		self.w = w
		self.dt = dt
		self.topk = topk

	def sense(self, action):
		print ("Sensing duration:",self.dt)
		obs = self.process.sample(action, dt = self.dt)
		return (action, obs, self.dt)

	def acquisition_function(self,actions):
		return torch.randn(size = (len(actions,1))).double()

	def fit_estimator(self):
		pass


	def count_events(self):
		total = 0
		for i in self.data:
			if i[1] is not None:
				total += i[1].size()[0]
		return total

	def step(self, actions, verbose = False, index = False, points = False):

		self.fit_estimator()

		# acquisiton function
		scores = self.acquisition_function(actions)

		best_regions, best_indices = torch.topk(scores,self.topk)

		if self.topk == 1:
			sensed_actions = [actions[best_indices]]
		else:
			sensed_actions = [actions[i] for i in best_indices]

		if verbose == True:
			print("Scores:",scores)
			print ("Sensing:", [action.bounds for action in sensed_actions])
			print ("Total events so far:", self.count_events())

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
				points_loc = torch.cat((points_loc,data_point[1]), dim = 0)

		if points == False:
			if points_loc is not None:
				return (cost,points_loc.size()[0],sensed_actions,best_indices)
			else:
				return (cost,0,sensed_actions, best_indices)
		else:
			if points_loc is not None:
				return (cost,points_loc,sensed_actions,best_indices)
			else:
				return (cost,None,sensed_actions, best_indices)

	def best_point_so_far(self, D, n):
		xx = D.return_discretization(n)
		map = self.estimator.mean_rate(D,n)
		return xx[torch.argmax(map),:].view(1,-1)





class EpsilonGreedySense(SensingAlgorithm):

	def __init__(self, process, estimator,w,initial_data = None, epsilon = lambda t: 1. - 1./np.sqrt(t), dt = 10., topk = 1):
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
		self.estimator.fit_gp()

	def acquisition_function(self,actions):
		self.t = self.t + 1
		p = np.random.uniform(0,1)
		scores = []
		epsilon = self.epsilon(self.t)
		if p > epsilon:
			# exploitation
			if self.estimator.rate is not None:
				for action in actions:
					map = self.estimator.mean_set(action)
					scores.append(float(map/self.w(action)))
			else:
				for action in actions:
					scores.append(np.random.randn())
		else:
			# random exploration
			for action in actions:
				scores.append(np.random.randn())

		return torch.Tensor(scores).double()

	def add_data(self,data_point):
		self.estimator.add_data_point(data_point)
		self.data.append(data_point)

if __name__ == "__main__":
	d = 1
	gamma = 0.1
	n = 32
	B = 4.
	b = 1.

	process = PoissonPointProcess(d=1, B=B, b=b)
	Sets = []
	levels = 6
	hierarchical_structure = HierarchicalBorelSets(d=1, interval=(-1, 1), levels=levels)
	Sets = hierarchical_structure.get_all_sets()

	D = BorelSet(1, bounds=torch.Tensor([[-1., 1.]]).double())

	k = KernelFunction(gamma=gamma, kappa=B).kernel
	estimator = PositiveRateEstimator(process, hierarchical_structure, kernel=k, B=B + b, b=b, m=100, jitter=10e-4)

	min_vol, max_vol = estimator.get_min_max()
	dt = (1 * b) / min_vol

	small_dt = 1.
	sample_D = process.sample_discretized(D, small_dt)
	data = []
	data.append((D, sample_D, small_dt))

	estimator.load_data(data)
	estimator.penalized_likelihood()
	rate_mean = estimator.mean_rate(D, n=n)

	xtest = D.return_discretization(n=n)
	plt.plot(xtest, rate_mean, color='blue', label='likelihood - locations known')  # normalized to dt = 1

	for j in data:
		if j[1] is not None:
			plt.plot(j[1], j[1] * 0, 'ko')

	process.visualize(D, samples=0, n=n, dt=1.)  # normalized to dt = 1

	w = lambda s: s.volume() * B
	Bandit = EpsilonGreedySense(process, estimator, w, epsilon=0.5,initial_data=data, dt=dt)
	T = 50
	plt.ion()
	rate = process.rate(xtest)

	for t in range(T):

		rate_mean = Bandit.estimator.mean_rate(D, n=n)
		plt.clf()

		# scores = Bandit.acquisition_function(Bandit.estimator.basic_sets)
		#
		# for index_set,score in enumerate(scores):
		# 	xx = Bandit.estimator.basic_sets[index_set].return_discretization(n)
		# 	plt.plot(xx,score+xx*0,'g')

		plt.plot(xtest, rate_mean, color='blue', label='likelihood - locations known')  # normalized to dt = 1
		plt.plot(xtest, rate, color='orange', label='rate', lw=3)

		plt.draw()
		plt.pause(0.5)

		Bandit.step(Sets)
