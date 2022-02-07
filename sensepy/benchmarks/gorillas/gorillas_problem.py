import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from stpy.borel_set import HierarchicalBorelSets, BorelSet
from stpy.point_processes.poisson_rate_estimator import PoissonRateEstimator
from stpy.kernels import KernelFunction
from sensepy.benchmarks.spatial_problem import SpatialProblem


class GorillasProblem(SpatialProblem):


	def __init__(self, n=30, m=30, b=0.):
		super().__init__(m)
		self.n = n
		self.m = m
		gamma = 0.2
		self.D = BorelSet(2, bounds=torch.Tensor([[-1., 1.], [-1, 1]]).double())
		self.hs2d = HierarchicalBorelSets(d=2, interval=[(-1, 1), (-1, 1)], levels=2)
		k = KernelFunction(gamma=gamma, d=2)
		self.estimator = PoissonRateEstimator(None, self.hs2d, d=2, kernel_object=k, B=100., m=m, jitter=10e-6)

	def load_data(self, size = 10, prefix = '../data/'):
		df = pd.read_csv(prefix+"gorillas_data.csv")
		df = df[['Longitude', 'Lattitude']]
		print("data loaded")
		obs = df.values[:, 0:2].astype(float)
		obs = obs[~np.isnan(obs)[:, 0], :]

		x_max = np.max(obs[:, 0])  # longitude
		x_min = np.min(obs[:, 0])

		y_max = np.max(obs[:, 1])  # lattitude
		y_min = np.min(obs[:, 1])

		lat = df['Lattitude']
		long = df['Longitude']

		self.left, self.right = long.min(), long.max()
		self.down, self.up = lat.min(), lat.max()

		self.transform_x = lambda x: (2 / (x_max - x_min)) * x + (1 - (2 * x_max / (x_max - x_min)))
		self.transform_y = lambda y: (2 / (y_max - y_min)) * y + (1 - (2 * y_max / (y_max - y_min)))

		obs[:, 0] = np.apply_along_axis(self.transform_x, 0, obs[:, 0])
		obs[:, 1] = np.apply_along_axis(self.transform_y, 0, obs[:, 1])

		dt = 1.
		obs = torch.from_numpy(obs)

		# estimate
		gamma = 0.2
		k = KernelFunction(gamma=gamma, d=2)

		hs2d = HierarchicalBorelSets(d = 2 , interval=[(-1,  1),(-1, 1)], levels = 5)
		self.estimator = PoissonRateEstimator(None, hs2d, d=2, kernel_object = k, b = 0.0, B=1000000., m=self.m, jitter=10e-6)

		D = BorelSet(2, bounds= torch.Tensor([[-1.,1.],[-1,1]]).double())
		self.D = D
		data = [(D,obs,dt)]
		self.obs = obs
		self.estimator.load_data(data)

	def fit_model(self):
		self.theta = self.estimator.fit_gp()


if __name__ == "__main__":
	Problem = GorillasProblem()
	Problem.load_data()
	Problem.fit_model()
	Problem.return_process()
	Problem.plot()