# @Author: George Onoufriou <archer>
# @Date:   2021-08-23T10:27:37+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-23T10:57:40+01:00


import time
import unittest
import numpy as np

from fhez.nn.operations.one_hot_encode import OneHotEncode


class OneHotEncodeTest(unittest.TestCase):
    """Test one hot encoder node."""

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

    def test_init(self):
        """Check object is initialised properly."""
        classes = 2
        encoder = OneHotEncode(length=classes)
        self.assertIsInstance(encoder, OneHotEncode)
        self.assertEqual(encoder.length, classes)

    def test_forward(self):
        """Check encoding forward pass working as expected."""
        encoder = OneHotEncode(length=5)
        clss = 2
        encoded = encoder.forward(clss)
        truth = np.zeros(5)
        truth[clss] = 1
        np.testing.assert_array_almost_equal(encoded, truth,
                                             decimal=1,
                                             verbose=True)

    def test_backward(self):
        """Check only one hot gradient is being returned."""
        encoder = OneHotEncode(length=5)
        clss = 2
        encoder.forward(clss)
        gradient = np.zeros(5)
        gradient[clss] = 1
        out = encoder.backward(gradient)
        np.testing.assert_array_almost_equal(out, np.array([1]),
                                             decimal=1,
                                             verbose=True)
