"""Sum operation tests."""
# @Author: George Onoufriou <archer>
# @Date:   2021-08-17T09:53:32+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-24T13:05:52+01:00

import time
import unittest
import numpy as np
from fhez.nn.operations.topsum import TopSum


class TopSumTest(unittest.TestCase):
    """Test Sum operation node."""

    @property
    def data_shape(self):
        """Define desired data shape."""
        return (3, 32, 32, 3)

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

    def setUp(self):
        """Start timer and init variables."""
        self.weights = (1,)  # if tuple allows cnn to initialise itself

        self.startTime = time.time()

    def tearDown(self):
        """Calculate and print time delta."""
        t = time.time() - self.startTime
        print('%s: %.3f' % (self.id(), t))

    def test_test(self):
        """Check our testing values meet requirements."""
        # check data is the shape we desire/ gave it to generate
        self.assertEqual(self.data.shape, self.data_shape)

    def test_init(self):
        """Check object initialisation."""
        TopSum()

    def test_forward(self):
        """Check sum forward pass activation correct."""
        s = TopSum()
        d = self.data
        a = s.forward(x=d)
        truth = np.sum(d, axis=0)
        np.testing.assert_array_almost_equal(a, truth,
                                             decimal=1,
                                             verbose=True)

    def test_backward(self):
        """Check sum backward pass gradients correct."""
        s = TopSum()
        d = self.data
        s.forward(x=d)
        grads = s.backward(gradient=1)
        self.assertEqual(len(grads), len(d))
        truth = np.ones((len(d),))
        np.testing.assert_array_almost_equal(grads, truth,
                                             decimal=1,
                                             verbose=True)
