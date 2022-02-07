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

import sensepy
import pytest


def test_version_number():
    version = sensepy.__version__
    assert hasattr(sensepy, "__version__")