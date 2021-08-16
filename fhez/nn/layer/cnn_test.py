# @Author: George Onoufriou <archer>
# @Date:   2021-08-10T14:36:02+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-16T16:08:47+01:00

import time
import unittest
import numpy as np

from fhez.nn.layer.cnn import CNN


class CNNTest(unittest.TestCase):
    """Test CNN node abstraction."""

    @property
    def data_shape(self):
        """Define desired data shape."""
        return (4, 4, 3)

    @property
    def filter_shape(self):
        """Get a filter shape corresponding to data."""
        return (3, 3, 3)

    @property
    def data(self):
        """Get some generated data."""
        array = np.ones(self.data_shape)
        return array

    @property
    def filt(self):
        """Generate some filter for the data."""
        filt = np.ones(self.filter_shape)/2
        return filt

    @property
    def bias(self):
        """Get a bias term."""
        return 0.5

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
        """Check that initialisation has occured properly."""
        cnn = CNN()
        self.assertIsInstance(cnn, CNN)

    def test_forward(self):
        """Test CNN filter and sum applied correctly."""
        weights = self.filt
        data = self.data
        bias = self.bias

        cnn = CNN(weights=weights, bias=bias)
        a = cnn.forward(x=data)
        # check that number of windows matches what we expect
        self.assertEqual(len(cnn.windows), 4)
        first = a[0]
        second = a[1]
        last = a[-1]

        first_truth = np.ones(self.data_shape)/2
        first_truth[0:3, 0:3, 0:3] = 1

        second_truth = np.ones(self.data_shape)/2
        second_truth[0:3, 1:4, 0:3] = 1

        last_truth = np.ones(self.data_shape)/2
        last_truth[1:4, 1:4, 0:3] = 1

        np.testing.assert_array_almost_equal(first, first_truth,
                                             decimal=1,
                                             verbose=True)
        np.testing.assert_array_almost_equal(second, second_truth,
                                             decimal=1,
                                             verbose=True)
        np.testing.assert_array_almost_equal(last, last_truth,
                                             decimal=1,
                                             verbose=True)

    def test_backward(self):
        """Test CNN gradient calculated correctly."""
        weights = self.filt
        data = self.data
        bias = self.bias

        cnn = CNN(weights=weights, bias=bias)
        cnn.forward(x=data)
        cnn.backward(gradient=1)
        grads = cnn.gradients.pop()
        # check bias gradient
        self.assertEqual(grads["dfdb"], self.filt.size*4)
        # check weights gradient
        np.testing.assert_array_almost_equal(grads["dfdw"],
                                             np.ones(self.filt.shape)*4,
                                             decimal=1,
                                             verbose=True)
        # check input (x) gradient
        dfdx_truth = [[[0.5, 0.5, 0.5],
                       [1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0],
                       [0.5, 0.5, 0.5]],
                      [[1.0, 1.0, 1.0],
                       [2.0, 2.0, 2.0],
                       [2.0, 2.0, 2.0],
                       [1.0, 1.0, 1.0]],
                      [[1.0, 1.0, 1.0],
                       [2.0, 2.0, 2.0],
                       [2.0, 2.0, 2.0],
                       [1.0, 1.0, 1.0]],
                      [[0.5, 0.5, 0.5],
                       [1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0],
                       [0.5, 0.5, 0.5]]]
        dfdx_truth = np.array(dfdx_truth)
        np.testing.assert_array_almost_equal(grads["dfdx"],
                                             dfdx_truth,
                                             decimal=1,
                                             verbose=True)
