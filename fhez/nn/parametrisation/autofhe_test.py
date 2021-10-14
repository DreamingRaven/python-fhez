# @Author: George Onoufriou <archer>
# @Date:   2021-09-14T11:51:45+01:00
# @Last modified by:   archer
# @Last modified time: 2021-10-14T16:26:57+01:00

import time
import unittest
import numpy as np
from fhez.nn.graph.prefab import cnn_classifier, basic
from fhez.nn.parametrisation.autofhe import autoHE, temp_encryptor_generator
from fhez.nn.traverse.firing import Firing
from fhez.rearray import ReArray


class AutoHE(unittest.TestCase):
    """Test automatic homomorphic parametreisation."""

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
        return basic()

    def test_parametrisation(self):
        """Check autohe can auto parametrise as expected."""
        graph = self.graph
        # auto parametrise all encrypted input nodes and their paths
        groups = autoHE(graph=graph, nodes=["x_0", "x_1", "y_0"])
        np.testing.assert_array_almost_equal(groups[1], [4, 4],
                                             decimal=4,
                                             verbose=True)
        nf = Firing(graph)
        nf.stimulate(neurons=np.array([]), signals=np.array([]),
                     receptor="forward")

    # def test_parametrisation_large(self):
    #     """Check autohe parameterisation works on much larger graphs."""
    #     graph = self.graph
    #     autoHE(graph=graph, node="x_0")
    #     autoHE(graph=graph, node="x_1")
    #     nf = Firing(graph)
    #     nf.stimulate(neurons=np.array([]), signals=np.array([]),
    #                  receptor="forward")

    def test_temp_encryptor_generator(self):
        cost = 6
        parms = temp_encryptor_generator(cost=cost)
        print(parms)
        # x = self.data.flatten()[:10]
        x = np.array([0.5, 0.3, 0.7, 0.9, 0.1])
        cyphertxt = ReArray(x, **parms)
        for _ in range(cost):
            cyphertxt = np.multiply(cyphertxt, 0.5)
            print("RUN")
        print(x)
        np.testing.assert_array_almost_equal(cyphertxt, x/(cost*2),
                                             decimal=6,
                                             verbose=True)

    def test_todo(self):
        """Todo note to fail tests so it cant be forgotten."""
        raise NotImplementedError("Autofhe incomplete.")
