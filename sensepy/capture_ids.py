import torch
import matplotlib.pyplot as plt
from stpy.borel_set import BorelSet,HierarchicalBorelSets
from stpy.kernels import KernelFunction
from stpy.point_processes.poisson_rate_estimator import PositiveRateEstimator
from stpy.point_processes.poisson import PoissonPointProcess
from sensepy.sense import SensingAlgorithm
from sensepy.capture_ucb import CaptureUCB
import numpy as np
from scipy import optimize

class CaptureIDS(CaptureUCB):

	def __init__(self,*args, actions = None,original_ids = True,**kwargs):
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

	def acquisition_function(self, actions):
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




	def step(self, actions, verbose = False, index = False, points = False):

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
if __name__ == "__main__":
	d = 1
	gamma = 0.1
	n = 32
	B = 4.
	b = 1.
	m = 32

	process = PoissonPointProcess(d=1, B=B, b=b)
	Sets = []
	levels = 5
	action_level = 5
	hierarchical_structure = HierarchicalBorelSets(d=1, interval=(-1, 1), levels=levels)

	Sets = hierarchical_structure.get_all_sets()
	basic_sets = hierarchical_structure.get_sets_level(hierarchical_structure.levels)
	actions = hierarchical_structure.get_sets_level(action_level)

	actions = []
	for level in range(action_level):
		actions = actions + hierarchical_structure.get_sets_level(level)

	D = BorelSet(1, bounds=torch.Tensor([[-1., 1.]]).double())

	k = KernelFunction(gamma=gamma, kappa=B)

	estimator = PositiveRateEstimator(process, hierarchical_structure, offset = 0.5, kernel_object=k, B=B + b, b = b, m=m, jitter=10e-5,
									  estimator='least-sq', uncertainty='least-sq', feedback = 'count-record', approx = "ellipsoid",
									  beta = 2.)
	vol = basic_sets[0].volume()
	dt = 1./(vol*b)

	data = []

	estimator.load_data(data)
	estimator.fit_gp()
	xtest = D.return_discretization(n = n)

	w = lambda s: s.volume()
	#w = lambda s: s.perimeter()

	Bandit = CaptureIDS(process,estimator,w, initial_data = data, dt = dt)
	T = 50
	rate = process.rate(xtest)
	delta = 0.05

	for t in range(T):

		Bandit.fit_estimator()
		rate_mean = Bandit.estimator.mean_rate(D, n=n)
		_ , lcb, ucb =  estimator.map_lcb_ucb(D,n, beta = 4.)
		scores = Bandit.acquisition_function(actions)

		plt.clf()
		# for index,action in enumerate(actions):
		# 	xx = action.return_discretization(n)
		# 	if index == 0:
		# 		plt.plot(xx,xx*0+(scores[index]*w(action))/action.volume()/dt,'g',lw = 2, label = 'IDS')
		# 	else:
		# 		plt.plot(xx, xx * 0 + (scores[index] * w(action)) / action.volume() / dt, 'g', lw=2)

		plt.title("Iteration: "+str(t))
		plt.plot(xtest,rate_mean,color = 'blue', label = 'rate estimate') # normalized to dt = 1
		plt.plot(xtest, rate,color = 'orange', label='rate', lw=3)
		plt.fill_between(xtest.numpy().flatten(), lcb.numpy().flatten(), ucb.numpy().flatten(), alpha = 0.4, color = 'gray', label = 'confidence')

		plt.legend()
		plt.show()
		#plt.savefig("experiments/video/"+str(t)+".png",dpi = 50)
		#plt.draw()
		#plt.pause(0.1)

		cost, events, _, _ = Bandit.step(actions, verbose=True)
		#Bandit.estimator.update_variances()



