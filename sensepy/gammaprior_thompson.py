import torch
import numpy as np
import matplotlib.pyplot as plt
from stpy.borel_set import BorelSet,HierarchicalBorelSets
from stpy.point_processes.poisson import PoissonPointProcess
from sensepy.sense import SensingAlgorithm
import scipy.stats

class GammaPriorThopsonSampling(SensingAlgorithm):


	def __init__(self, process,basic_sets, w,initial_data = None, dt = 10., alpha = 1., beta = 1., topk =1):
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


	def fit_estimator(self):
		pass

	def add_data(self,new_data):
		index = 0

		for set in self.basic_sets:
			if set.inside(new_data[0]):
				break
			index +=1
		#print (new_data[0].bounds)
		#print (self.basic_sets[index].bounds)
		#print ('---------')
		if new_data[1] is not None:
			self.alphas[index] += new_data[1].size()[0]
		self.betas[index] += 1

		self.data.append(new_data)

	def acquisition_function(self,actions):
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




if __name__ == "__main__":
	d = 1
	gamma = 0.1
	n = 32
	B = 4.
	b = 1.

	process = PoissonPointProcess(d=1, B=B, b=b)
	Sets = []
	levels = 4
	hierarchical_structure = HierarchicalBorelSets(d=1, interval=(-1, 1), levels=levels)


	Sets = hierarchical_structure.get_all_sets()
	basic_sets = hierarchical_structure.get_sets_level(hierarchical_structure.levels)

	D = BorelSet(1, bounds=torch.Tensor([[-1., 1.]]).double())

	dt = 1.
	vol = basic_sets[0].volume()

	small_dt = 1.
	sample_D = process.sample_discretized(D, small_dt)
	data = []
	data.append((D, sample_D, small_dt))

	xtest = D.return_discretization(n=n)

	# for j in data:
	# 	if j[1] is not None:
	# 		plt.plot(j[1], j[1] * 0, 'ko')
	#
	# process.visualize(D, samples=0, n=n, dt=1.)  # normalized to dt = 1

	w = lambda s: s.volume() * B
	Bandit = GammaPriorThopsonSampling(process, basic_sets, w, initial_data=data,alpha=dt*B*vol, beta=10, dt=dt)
	T = 50
	plt.ion()
	rate = process.rate(xtest)

	for t in range(T):

		plt.clf()
		plt.plot(xtest, rate, color='orange', label='rate', lw=3)
		for index_set, set in enumerate(basic_sets):
			xx = set.return_discretization(n)
			r = process.rate_volume(set)
			plt.plot(xx,Bandit.alphas[index_set]/Bandit.betas[index_set]/vol+xx*0,'--',color = 'k')
			plt.plot(xx, r/vol + xx * 0, color = 'orange',lw = 3)

		plt.draw()
		plt.pause(0.5)
		Bandit.step(basic_sets)
