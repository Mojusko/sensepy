# This source code is part of the sensepy package and is distributed
# under the License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Mojmir Mutny"
__copyright__ = "Copyright (c) 2022 Mojmir Mutny, ETH Zurich"
__credits__ = ["Mojmir Mutny"]
__license__ = "MIT Licence"
__version__ = "0.1"
__email__ = "mojmir.mutny@inf.ethz.ch"
__status__ = "DEV"

from sensepy import PoissonRateEstimator
from sensepy import PoissonPointProcess
from sensepy import EpsilonGreedySense, SenseEntropy, Top2Thompson
from stpy import HierarchicalBorelSets, BorelSet
from stpy import KernelFunction
import torch
import pytest



def test_capture_epsilon_greedy():
	gamma = 0.1
	n = 32
	B = 4.
	b = 1.

	process = PoissonPointProcess(d=1, B=B, b=b)
	levels = 6
	hierarchical_structure = HierarchicalBorelSets(d=1, interval=(-1, 1), levels=levels)
	Sets = hierarchical_structure.get_all_sets()

	k = KernelFunction(gamma=gamma, kappa=B)
	estimator = PoissonRateEstimator(process, hierarchical_structure, kernel_object=k, B=B + b, b=b, m=100, jitter=1e-4)

	min_vol, max_vol = estimator.get_min_max()
	dt = (1 * b) / min_vol
	data = []
	w = lambda s: s.volume() * B
	Bandit = EpsilonGreedySense(process, estimator, w, epsilon=lambda x: 0.5, initial_data=data, dt=dt)
	T = 5
	for t in range(T):
		Bandit.step(Sets)

	assert (t+1 == T)



@pytest.fixture
def example_setup_count_record():
	gamma = 0.1
	B = 4.
	b = 1.
	m = 128
	n = 64
	process = PoissonPointProcess(d=1, B=B, b=b)
	levels = 3
	hierarchical_structure = HierarchicalBorelSets(d=1, interval=(-1, 1), levels=levels)
	actions = hierarchical_structure.get_all_sets()

	D = BorelSet(1, bounds=torch.Tensor([[-1., 1.]]).double())

	k = KernelFunction(gamma=gamma, kappa=B)
	estimator = PoissonRateEstimator(process, hierarchical_structure, kernel_object=k,
					B=B + b, b=b, m=m, jitter=10e-4, approx="ellipsoid")

	min_vol, max_vol = estimator.get_min_max()
	dt = (b) / min_vol
	dt = dt
	w = lambda s: s.volume() * 1000

	return [process, estimator, actions, w, D, dt, n]


def test_v_design_approx_count_record(example_setup_count_record):
	[process, estimator, actions, w, D, dt, n] = example_setup_count_record
	data = []
	Bandit = SenseEntropy(process, estimator, w, D, n, initial_data=data,
						  dt=dt, level_set=2, repeats=1, design_type="V-approx")
	T = 5
	for t in range(T):
		above, not_known, below = Bandit.classify_region(D, n)
		x = Bandit.best_point_so_far(D, n)
		Bandit.step(actions)
	assert (t+1 == T)

def test_v_design_count_record(example_setup_count_record):
	[process, estimator, actions, w, D, dt, n] = example_setup_count_record
	data = []
	Bandit = SenseEntropy(process, estimator, w, D, n, initial_data=data,
						  dt=dt, level_set=2, repeats=1, design_type="V")
	T = 5
	for t in range(T):
		above, not_known, below = Bandit.classify_region(D, n)
		x = Bandit.best_point_so_far(D, n)
		Bandit.step(actions)
	assert (t+1 == T)


def test_top2_count_record(example_setup_count_record):
	[process, estimator, actions, w, D, dt, n] = example_setup_count_record
	data = []
	estimator.steps = 10
	Bandit = Top2Thompson(process, estimator, w, D, n, initial_data=data, dt=dt, level_set=2)
	T = 5
	for t in range(T):
		Bandit.step(actions)
		above, not_known, below = Bandit.classify_region(D, n)
		x = Bandit.best_point_so_far(D, n)

	assert (t+1 == T)