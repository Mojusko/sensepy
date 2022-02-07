import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.cluster import AgglomerativeClustering
from stpy.helpers.transformations import transform
from stpy.borel_set import HierarchicalBorelSets, BorelSet
from stpy.point_processes.poisson_rate_estimator import PoissonRateEstimator
from stpy.kernels import KernelFunction
from sensepy.benchmarks.spatial_problem import SpatialProblem
from stpy.continuous_processes.gauss_procc import GaussianProcess

class BeilschmiediaProblem(SpatialProblem):

	def __init__(self, m=20, levels=4, b=0.1, gamma=0.1, basis="triangle"):
		super().__init__(m)
		self.D = BorelSet(2, bounds=torch.Tensor([[-1., 1.], [-1, 1]]).double())
		self.hs2d = HierarchicalBorelSets(d=2, interval=[(-1, 1), (-1, 1)], levels=levels)
		self.basis = basis
		self.b = b
		self.gamma = gamma
		self.m = m

	def load_data(self, clusters = 100, prefix = "../data/"):

		print ("loading begins.")

		# load elevation values
		dt = pd.read_csv(prefix+"elev.csv")
		X = torch.from_numpy(dt[['x', 'y']].values).double().numpy()
		y = torch.from_numpy(dt['value'].values).double().view(-1, 1).numpy()

		dt = pd.read_csv(prefix+"grad.csv")
		X2 = torch.from_numpy(dt[['x', 'y']].values).double().numpy()
		y2 = torch.from_numpy(dt['value'].values).double().view(-1, 1).numpy()

		xx = pd.read_csv(prefix+"x.csv")
		yy = pd.read_csv(prefix+"y.csv")

		print ("data loaded.")

		centers = clusters
		kmeans = AgglomerativeClustering(n_clusters=centers).fit(X)

		cluster_centers = np.zeros(shape = (centers,X.shape[1]))
		for i in range(centers):
			cluster_centers[i,:] = np.average(X[kmeans.labels_ == i,:],axis = 0)


		Xsub = np.zeros(shape=(centers, 2))
		ysub = np.zeros(shape=(centers))
		X2sub = np.zeros(shape=(centers, 2))
		y2sub = np.zeros(shape=(centers))

		for i in range(centers):
			center = cluster_centers[i,:]
			dist = np.sum((X - np.tile(center, (X.shape[0], 1))) ** 2, axis=1)
			index = np.argmin(dist)
			Xsub[i, :] = X[index, :]
			ysub[i] = y[index]

			X2sub[i, :] = X2[index, :]
			y2sub[i] = y2[index]

		# transform to [-1,1]
		self.Xtsub, trans, inv_trans, tr, _ = transform(torch.from_numpy(Xsub), offsets=[(0, 1000), (0, 500)])
		self.X2tsub, trans, inv_trans, tr, _ = transform(torch.from_numpy(X2sub), offsets=[(0, 1000), (0, 500)])

		self.ytsub = transform(torch.from_numpy(ysub).view(-1, 1), functions=False)
		self.y2tsub = transform(torch.from_numpy(y2sub).view(-1, 1), functions=False)

		xx = np.apply_along_axis(tr[0], 0, xx['x'].values)
		yy = np.apply_along_axis(tr[1], 0, yy['x'].values)
		obs = torch.stack([torch.from_numpy(xx), torch.from_numpy(yy)]).T.double()

		self.obs = obs
		print ("Number of events:",self.obs.size())
		self.dt = 1.
		self.data = [(self.D, self.obs, self.dt)]


	def fit_model(self):

		GP = GaussianProcess(kappa=1, gamma=0.1, d=2)
		GP2 = GaussianProcess(kappa=1, gamma=0.1, d=2)

		GP.fit_gp(self.Xtsub, self.ytsub)
		GP2.fit_gp(self.X2tsub, self.y2tsub)

		self.height = lambda x: GP.mean(x)  # [-1,1]
		self.slope = lambda x: GP2.mean(x)

		# custom kernel function
		self.kernel1 = KernelFunction(d = 1, kernel_name="squared_exponential", gamma = self.gamma, kappa = 1)
		self.kernel2 = KernelFunction(d = 1, kernel_name="squared_exponential", gamma = self.gamma, kappa = 1)
		kernel = lambda x, y, kappa, group: kappa*1000*self.kernel1.kernel(self.height(x),self.height(y))*self.kernel1.kernel(self.slope(x),self.slope(y))
		self.kernel = KernelFunction(kernel_function = kernel, d = 2)

		# fit estimator
		self.estimator = PoissonRateEstimator(None, self.hs2d, d=2, basis=self.basis, kernel_object=self.kernel, B=10e10, b=self.b, m=self.m, jitter=1e-5, opt='cvxpy')
		self.estimator.load_data(self.data)
		self.estimator.fit_gp()


if __name__ == "__main__":

	torch.manual_seed(24)
	torch.use_deterministic_algorithms(True)
	np.random.seed(24)
	random.seed(0)

	Problem = BeilschmiediaProblem(m = 20, basis = "triangle")
	Problem.load_data(clusters = 100)
	Problem.fit_model()
	Problem.return_process(n=20)
	Problem.plot()

