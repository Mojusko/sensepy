import torch
import numpy as np
import matplotlib.pyplot as plt
from sensepy.capture_ucb import SensingAlgorithm
from stpy.borel_set import BorelSet,HierarchicalBorelSets
from stpy.kernels import KernelFunction
from stpy.point_processes.poisson_rate_estimator import PositiveRateEstimator
from stpy.point_processes.poisson import PoissonPointProcess
from sensepy.sense_entropy import SenseEntropy
#from multiprocessing import Pool

class Top2Thompson(SenseEntropy):

	def __init__(self, *args,**kwargs):
		super().__init__(*args,**kwargs)
		self.t = 0
		self.topk = 1

	def get_region(self,D,n):
		mean, lcb, ucb = self.get_ucb_lcb(D,n)

		if self.level_set is not None:
			R = ucb > self.level_set
			R2 = lcb < self.level_set
			R = R*R2
		else:
			R = ucb > torch.max(lcb)

		return R.view(-1)


	def acquisition_function(self, actions):
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
				print ("Resampling.")
				if self.level_set is not None:
					print ("Above:",torch.sum(R1)/float(len(R1)))
					print (path1)
					print ("Above:",torch.sum(R2)/float(len(R1)))
					print (path2)
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


	def sense(self, action):
		obs = self.process.sample(action, dt = self.dt)
		print ("Sensing duration:",self.dt)
		self.t = self.t + 1
		return (action, obs, self.dt)


if __name__ == "__main__":
	d = 1
	gamma = 0.1
	n = 10
	B = 4.
	b = 0.
	m = 32

	process = PoissonPointProcess(d=1, B=B, b=b)
	Sets = []
	levels = 3
	hierarchical_structure = HierarchicalBorelSets(d=1, interval=(-1, 1), levels=levels)
	Sets = hierarchical_structure.get_sets_level(levels)

	D = BorelSet(1, bounds=torch.Tensor([[-1., 1.]]).double())

	k = KernelFunction(gamma=gamma, kappa=B)
	estimator = PositiveRateEstimator(process, hierarchical_structure, kernel_object=k, B=B + b, b = b, m=m, jitter=10e-6, approx="ellipsoid")

	dt = 5
	xtest = D.return_discretization(n = n)

	w = lambda s: s.volume()*1000
	data = []
	Bandit = Top2Thompson(process,estimator,w,D,n, initial_data = data, dt = dt, level_set = 2)
	T = 50
	rate = process.rate(xtest)

	for t in range(T):
		Bandit.step(actions=Sets, verbose = True)
		print (t)