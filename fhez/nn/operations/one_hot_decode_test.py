# @Author: George Onoufriou <archer>
# @Date:   2021-08-23T10:27:46+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-23T12:16:16+01:00

import time

import unittest
import numpy as np

from fhez.nn.operations.one_hot_decode import OneHotDecode


class OneHotDecodeTest(unittest.TestCase):
    """Test one hot decoder node."""

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
        decoder = OneHotDecode()
        self.assertIsInstance(decoder, OneHotDecode)

    def test_forward(self):
        """Check decoding forward pass working as expected."""
        decode = OneHotDecode()
        one_hot = np.eye(5)[3]
        not_hot = decode.forward(one_hot)
        self.assertEqual(not_hot, np.argmax(one_hot))

    def test_backward(self):
        """Check only one hot gradient is chained."""
        decode = OneHotDecode()
        one_hot = np.eye(5)[3]
        not_hot = decode.forward(one_hot)
        self.assertEqual(not_hot, np.argmax(one_hot))
        local_grad = decode.backward(2)
        np.testing.assert_array_almost_equal(local_grad, one_hot * 2,
                                             decimal=1,
                                             verbose=True)
