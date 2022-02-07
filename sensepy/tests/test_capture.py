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
from sensepy import CaptureUCB, CaptureThompson, CaptureIDS
from stpy import HierarchicalBorelSets, BorelSet
from stpy import KernelFunction
import torch
import pytest


@pytest.fixture
def example_setup_count_record():
	d = 1
	gamma = 0.1
	B = 4.
	b = 1.
	m = 64

	process = PoissonPointProcess(d=1, B=B, b=b)
	levels = 6
	action_level = 5
	hierarchical_structure = HierarchicalBorelSets(d=1, interval=(-1, 1), levels=levels)

	basic_sets = hierarchical_structure.get_sets_level(hierarchical_structure.levels)
	actions = hierarchical_structure.get_sets_level(action_level)
	k = KernelFunction(gamma=gamma, kappa=B, d = d)
	estimator = PoissonRateEstimator(process, hierarchical_structure, kernel_object=k, B=B + b, b=b, m=m, jitter=10e-3,
									 estimator='likelihood', uncertainty='laplace', approx='ellipsoid',
									 feedback='count-record')
	vol = basic_sets[0].volume()
	dt = 1. / (vol * b)
	data = []
	estimator.load_data(data)
	estimator.fit_gp()
	w = lambda s: s.volume()
	return [process, estimator, w, actions, data, dt]

@pytest.fixture
def example_setup_histogram():
	d = 1
	gamma = 0.1
	B = 4.
	b = 1.
	m = 64

	process = PoissonPointProcess(d=1, B=B, b=b)
	levels = 6
	action_level = 5
	hierarchical_structure = HierarchicalBorelSets(d=1, interval=(-1, 1), levels=levels)

	basic_sets = hierarchical_structure.get_sets_level(hierarchical_structure.levels)
	actions = hierarchical_structure.get_sets_level(action_level)
	k = KernelFunction(gamma=gamma, kappa=B, d = d)
	estimator = PoissonRateEstimator(process, hierarchical_structure, kernel_object=k, B=B + b, b=b, m=m, jitter=10e-3,
									 estimator='likelihood', uncertainty='laplace', approx='ellipsoid',
									 feedback='histogram')
	vol = basic_sets[0].volume()
	dt = 1. / (vol * b)
	data = []
	estimator.load_data(data)
	estimator.fit_gp()
	w = lambda s: s.volume()
	return [process, estimator, w, actions, data, dt]



def test_capture_ucb_count_record(example_setup_count_record):
	[process, estimator, w, actions, data, dt] = example_setup_count_record
	Bandit = CaptureUCB(process, estimator, w, initial_data=data, dt=dt, topk=1)
	T = 10
	for t in range(T):
		Bandit.fit_estimator()
		cost, events, _, _ = Bandit.step(actions, verbose=False)

	assert (t+1==T)


def test_capture_ids_count_record(example_setup_count_record):
	[process, estimator, w, actions, data, dt] = example_setup_count_record
	Bandit = CaptureIDS(process, estimator, w, initial_data=data, dt=dt, topk=1)
	T = 10
	for t in range(T):
		Bandit.fit_estimator()
		cost, events, _, _ = Bandit.step(actions, verbose=False)

	assert (t+1==T)

def test_capture_thompson_count_record(example_setup_count_record):
	[process, estimator, w, actions, data, dt] = example_setup_count_record
	estimator.steps = 5 # set steps to low number.
	Bandit = CaptureThompson(process, estimator, w, initial_data=data, dt=dt, topk=1)
	T = 10
	for t in range(T):
		Bandit.fit_estimator()
		cost, events, _, _ = Bandit.step(actions, verbose=False)

	assert (t+1==T)



def test_capture_ucb_histogram(example_setup_histogram):
	[process, estimator, w, actions, data, dt] = example_setup_histogram
	Bandit = CaptureUCB(process, estimator, w, initial_data=data, dt=dt, topk=1)
	T = 10
	for t in range(T):
		Bandit.fit_estimator()
		cost, events, _, _ = Bandit.step(actions, verbose=False)

	assert (t+1==T)


def test_capture_ids_histogram(example_setup_histogram):
	[process, estimator, w, actions, data, dt] = example_setup_histogram
	Bandit = CaptureIDS(process, estimator, w, initial_data=data, dt=dt, topk=1)
	T = 10
	for t in range(T):
		Bandit.fit_estimator()
		cost, events, _, _ = Bandit.step(actions, verbose=False)

	assert (t+1==T)

def test_capture_ucb_histogram_batch(example_setup_histogram):
	[process, estimator, w, actions, data, dt] = example_setup_histogram
	Bandit = CaptureUCB(process, estimator, w, initial_data=data, dt=dt, topk=2)
	T = 10
	for t in range(T):
		Bandit.fit_estimator()
		cost, events, _, _ = Bandit.step(actions, verbose=False)

	assert (t + 1 == T)