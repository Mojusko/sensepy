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
from sensepy import GammaPriorThopsonSampling
from stpy import HierarchicalBorelSets, BorelSet
from stpy import KernelFunction
import torch
import pytest


def test_thompson_gammaprior():
	B = 4.
	b = 1.
	process = PoissonPointProcess(d=1, B=B, b=b)
	levels = 4
	hierarchical_structure = HierarchicalBorelSets(d=1, interval=(-1, 1), levels=levels)
	basic_sets = hierarchical_structure.get_sets_level(hierarchical_structure.levels)
	dt = 1.
	vol = basic_sets[0].volume()
	data = []
	w = lambda s: s.volume() * B
	Bandit = GammaPriorThopsonSampling(process, basic_sets, w, initial_data=data,alpha=dt*B*vol, beta=10, dt=dt)
	T = 5
	for t in range(T):
		Bandit.step(basic_sets)
	assert t+1 == T