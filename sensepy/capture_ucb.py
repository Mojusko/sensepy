import torch
import matplotlib.pyplot as plt
from stpy.borel_set import BorelSet,HierarchicalBorelSets
from stpy.kernels import KernelFunction
from stpy.point_processes.poisson_rate_estimator import PositiveRateEstimator
from stpy.point_processes.poisson import PoissonPointProcess
from sensepy.sense import SensingAlgorithm

class CaptureUCB(SensingAlgorithm):

	def __init__(self, process, estimator,w,initial_data = None, dt = 10., topk = 1):
		self.process = process
		self.estimator = estimator
		self.dt = dt
		self.w = w
		self.data = initial_data
		self.estimator.load_data(self.data)
		self.topk = topk

	def fit_estimator(self):
		self.estimator.update_variances()
		self.estimator.fit_gp()

	def acquisition_function(self, actions):
		scores = []
		for action in actions:
			scores.append(self.estimator.ucb(action, dt = self.dt)/self.w(action))
		return torch.Tensor(scores).double()

	def add_data(self,data_point):
		self.estimator.add_data_point(data_point)
		self.data.append(data_point)

if __name__ == "__main__":
	d = 1
	gamma = 0.1
	n = 64
	B = 4.
	b = 1.
	m = 64

	process = PoissonPointProcess(d=1, B=B, b=b)
	Sets = []
	levels = 6
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
	estimator = PositiveRateEstimator(process, hierarchical_structure, kernel_object=k, B=B + b, b = b, m=m, jitter=10e-3,
									  estimator='least-sq', uncertainty='least-sq', approx = 'ellipsoid', feedback = 'histogram')
	estimator_laplace = PositiveRateEstimator(process, hierarchical_structure, kernel_object=k, B=B + b, b = b, m=m, jitter=10e-3)

	vol = basic_sets[0].volume()
	dt = 1./(vol*b)



	data = []

	estimator.load_data(data)
	estimator.fit_gp()
	xtest = D.return_discretization(n = n)

	w = lambda s: s.volume()
	w = lambda s: s.perimeter()

	Bandit = CaptureUCB(process,estimator,w, initial_data = data, dt = dt, topk = 2)
	T = 50
	plt.ion()
	rate = process.rate(xtest)
	delta = 0.05

	for t in range(T):

		Bandit.fit_estimator()
		rate_mean = Bandit.estimator.mean_rate(D, n=n)
		_ , lcb, ucb =  estimator.map_lcb_ucb(D,n, beta = 4.)
		scores = Bandit.acquisition_function(actions)

		plt.clf()
		for index,action in enumerate(actions):
			xx = action.return_discretization(n)
			if index == 0:
				plt.plot(xx,xx*0+(scores[index]*w(action))/action.volume()/dt,'g',lw = 2, label = 'UCB')
			else:
				plt.plot(xx, xx * 0 + (scores[index] * w(action)) / action.volume() / dt, 'g', lw=2)

		plt.title("Iteration: "+str(t))
		plt.plot(xtest,rate_mean,color = 'blue', label = 'rate estimate') # normalized to dt = 1
		plt.plot(xtest, rate,color = 'orange', label='rate', lw=3)
		plt.fill_between(xtest.numpy().flatten(), lcb.numpy().flatten(), ucb.numpy().flatten(), alpha = 0.4, color = 'gray', label = 'confidence')

		plt.legend()
		plt.savefig("experiments/video/"+str(t)+".png",dpi = 50)
		plt.draw()
		plt.pause(0.1)

		cost, events, _, _ = Bandit.step(actions, verbose=True)
		Bandit.estimator.update_variances()

		print (len(Bandit.estimator.data))




