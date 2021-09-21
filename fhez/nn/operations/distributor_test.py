# @Author: George Onoufriou <archer>
# @Date:   2021-09-21T14:42:26+01:00
# @Last modified by:   archer
# @Last modified time: 2021-09-21T14:54:53+01:00

import time
import unittest
import numpy as np
from fhez.nn.operations.distributor import Distributor


class DistributorTest(unittest.TestCase):
    """Test enqueue operation node."""

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

    def test_init(self):
        """Test enque initialisation."""
        Distributor()

    def test_forward(self):
        """Check distributor distributes as expected."""
        node = Distributor()
        out = node.forward([1, 2, 3, 4, 5])
        np.testing.assert_array_almost_equal(out, [1, 2, 3, 4, 5],
                                             decimal=1,
                                             verbose=True)

    def test_backward(self):
        """Check distributor distributes as expected."""
        node = Distributor()
        out = node.backward([1, 2, 3, 4, 5])
        self.assertEqual(out, 15)
        np.testing.assert_array_almost_equal(out, np.sum([1, 2, 3, 4, 5],
                                                         axis=0),
                                             decimal=1,
                                             verbose=True)
