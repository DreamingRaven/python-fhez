# @Author: George Onoufriou <archer>
# @Date:   2021-09-16T12:27:58+01:00
# @Last modified by:   archer
# @Last modified time: 2021-09-16T12:31:08+01:00

import time
import unittest
import numpy as np
from fhez.nn.graph.prefab import cnn_classifier, basic
from fhez.nn.traverse.firing import Firing


class PrefabTester(unittest.TestCase):
    """Test neural network prefab graphs functionality."""

    def setUp(self):
        """Start timer and init variables."""
        self.start_time = time.time()

    def tearDown(self):
        """Calculate and print time delta."""
        t = time.time() - self.start_time
        print('%s: %.3f' % (self.id(), t))

    @property
    def data_shape(self):
        """Define desired data shape."""
        return (28, 28)

    @property
    def data(self):
        """Get some generated data."""
        array = np.random.rand(*self.data_shape)
        return array

    @property
    def reseal_args(self):
        """Get some reseal arguments for encryption."""
        return {
            "scheme": 2,  # seal.scheme_type.CKK,
            "poly_modulus_degree": 8192*2,  # 438
            # "coefficient_modulus": [60, 40, 40, 60],
            "coefficient_modulus":
                [45, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 45],
            "scale": pow(2.0, 30),
            "cache": True,
        }

    def test_todo(self):
        """Todo note to fail tests so it cant be forgotten."""
        raise NotImplementedError("FHE prefab tests not completed.")
