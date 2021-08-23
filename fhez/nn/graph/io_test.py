# @Author: George Onoufriou <archer>
# @Date:   2021-08-23T10:14:13+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-23T10:23:21+01:00

import time
import unittest
import numpy as np

from fhez.nn.graph.io import IO


class IOTest(unittest.TestCase):
    """Test IO node."""

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
        """Check IO initialises properly."""
        i = IO()
        self.assertIsInstance(i, IO)

    def test_forward(self):
        """Check IO maps input to output."""
        i = IO()
        data = self.data
        out = i.forward(data)
        np.testing.assert_array_almost_equal(out, data,
                                             decimal=2,
                                             verbose=True)

    def test_backward(self):
        """Check IO maps gradients to input."""
        i = IO()
        data = self.data
        i.forward(data)
        out = i.backward(data)
        np.testing.assert_array_almost_equal(out, data,
                                             decimal=2,
                                             verbose=True)
