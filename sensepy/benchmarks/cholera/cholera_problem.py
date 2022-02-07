import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy.interpolate import griddata
from shapely.geometry import Polygon, Point
from stpy.borel_set import HierarchicalBorelSets, BorelSet
from stpy.point_processes.poisson_rate_estimator import PoissonRateEstimator
from stpy.kernels import KernelFunction
from sensepy.benchmarks.spatial_problem import SpatialProblem
from stpy.continuous_processes.nystrom_fea import NystromFeatures
import copy
import geopandas


class CholeraProblem(SpatialProblem):

	def __init__(self, n=40, m=30, b=0., gamma=0.5):
		super().__init__(m)
		self.n = n
		self.m = m
		self.b = b
		self.gamma = gamma
		self.D = BorelSet(2, bounds=torch.Tensor([[-1., 1.], [-1, 1]]).double())

	def load_data(self, name_prefix = '../data/'):
		pumps = pd.read_csv(name_prefix + 'pumps.csv')
		pumps = pumps.drop([3])
		locations_pumps = pumps[['X coordinate', 'Y coordinate']]

		deaths = pd.read_csv(name_prefix +'deaths.csv')
		locations_deaths = deaths[['X coordinate', 'Y coordinate']]

		together = pd.concat((locations_pumps, locations_deaths))

		gdf = geopandas.GeoDataFrame(
			locations_deaths,
			geometry=geopandas.points_from_xy(locations_deaths['Y coordinate'], locations_deaths['X coordinate']))
		self.gdf = copy.deepcopy(gdf)
		gdf2 = geopandas.GeoDataFrame(
			locations_pumps,
			geometry=geopandas.points_from_xy(locations_pumps['Y coordinate'], locations_pumps['X coordinate']))

		obs = together.values[:, 0:2]
		# transform data to [-1,1]
		x_max = np.max(obs[:, 0])
		x_min = np.min(obs[:, 0])

		y_max = np.max(obs[:, 1])
		y_min = np.min(obs[:, 1])


		self.left = y_min
		self.right = y_max

		self.down = x_min
		self.up = x_max

		self.transform_x = lambda x: (2 / (x_max - x_min)) * x + (1 - (2 * x_max / (x_max - x_min)))
		self.transform_y = lambda y: (2 / (y_max - y_min)) * y + (1 - (2 * y_max / (y_max - y_min)))

		obs = gdf.values[:, 0:2]

		obs[:, 0] = np.apply_along_axis(self.transform_x, 0, obs[:, 0])
		obs[:, 1] = np.apply_along_axis(self.transform_y, 0, obs[:, 1])

		self.obs = torch.from_numpy(obs.astype(float))

		self.dt = 40. # lenght of the epidemic
		data = [(self.D, self.obs, self.dt)]
		self.hs2d = HierarchicalBorelSets(d=2, interval=[(-1, 1), (-1, 1)], levels=5)

		# estimate
		kernel = KernelFunction(gamma=self.gamma, d=2, kappa = 16.)

		# definekernel function on the inducing points
		obs_pumps = gdf2.values[:, 0:2]

		obs_pumps[:, 0] = np.apply_along_axis(self.transform_x, 0, obs_pumps[:, 0])
		obs_pumps[:, 1] = np.apply_along_axis(self.transform_y, 0, obs_pumps[:, 1])
		obs_pumps = obs_pumps.astype(float)

		m = obs_pumps.shape[0]
		GP = NystromFeatures(kernel, m=torch.Tensor([m]), s=10e-6, approx="svd")

		obs_pumps = torch.from_numpy(obs_pumps)
		y = torch.zeros(size=(obs_pumps.size()[0], 1))
		GP.fit_gp(obs_pumps, y)

		def k_func(x, y, kappa=1., group=None):
			return GP.embed(x) @ GP.embed(y).T

		self.k = KernelFunction(d=2, kernel_function=k_func)

		self.estimator = PoissonRateEstimator(None, self.hs2d, d=2, kernel_object=self.k,
											  B=1000000., m=self.m, jitter=1e-5, b = self.b)
		self.estimator.load_data(data)

	def fit_model(self):
		self.theta = self.estimator.fit_gp()


if __name__ == "__main__":
	Problem = CholeraProblem(m = 20)
	Problem.load_data()
	Problem.fit_model()
	Problem.plot()

