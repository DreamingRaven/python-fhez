# @Author: George Onoufriou <archer>
# @Date:   2021-08-23T17:19:31+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-24T08:47:15+01:00

import time
import unittest
import numpy as np

from fhez.nn.graph.prefab import cnn_classifier
from fhez.nn.traverse.firing import Firing


class FiringTest(unittest.TestCase):
    """Test linear activation function."""

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

    @property
    def graph(self):
        """Get neuron/ computational graph to test against."""
        return cnn_classifier(10)

    def test_stimulate_forward(self):
        """Check neuronal firing algorithm forward stimulation of graph."""
        graph = self.graph
        data = self.data
        f = Firing(graph=graph)
        f.stimulate(neurons=["x", "y"], signals=[data, 1])
