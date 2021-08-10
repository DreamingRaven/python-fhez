# @Author: George Onoufriou <archer>
# @Date:   2021-08-09T16:48:31+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-09T17:20:21+01:00

import time
import unittest
import numpy as np

from fhez.nn.activation.sigmoid import Sigmoid


class Sigmoid_Test(unittest.TestCase):

    def setUp(self):
        """Start timer and init variables."""
        self.start_time = time.time()

    def tearDown(self):
        """Calculate and print time delta."""
        t = time.time() - self.start_time
        print('%s: %.3f' % (self.id(), t))

    @property
    def x(self):
        x = [1, -1, 0.7]
        return np.array(x)

    def test_forward(self):
        """Check sigmoid forward pass against known truth, plaintext."""
        print(self.x.shape)
        activation_function = Sigmoid()
        activation = activation_function.forward(self.x)
        # activation_truth = activation_function.sigmoid(self.x)
        activation_truth = np.array([0.693, 0.307, 0.637])
        np.testing.assert_array_almost_equal(activation, activation_truth,
                                             decimal=3, verbose=True)

    def test_backward(self):
        """Check sigmoid backward pass against known truth, plaintext."""
        activation_function = Sigmoid()
        activation_function.forward(self.x)
        df_dx = activation_function.backward(np.array([1]))
        df_dx_truth = np.array([0.185, 0.185, 0.1911])
        np.testing.assert_array_almost_equal(df_dx, df_dx_truth,
                                             decimal=3, verbose=True)
