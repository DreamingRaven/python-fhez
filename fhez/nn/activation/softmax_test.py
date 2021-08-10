# @Author: George Onoufriou <archer>
# @Date:   2021-08-02T22:01:04+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-09T16:04:55+01:00

import time
import unittest
import numpy as np

from fhez.nn.activation.softmax import Softmax
from fhez.nn.loss.cce import CCE


class SoftmaxTest(unittest.TestCase):
    """Test Softmax."""

    def setUp(self):
        """Start timer and init variables."""
        self.start_time = time.time()

    def tearDown(self):
        """Calculate and print time delta."""
        t = time.time() - self.start_time
        print('%s: %.3f' % (self.id(), t))

    def test_init(self):
        """Check SGD can be initialised using defaults."""
        activation = Softmax()
        self.assertIsInstance(activation, Softmax)

    def test_backward(self):
        """Test softmax backward pass with known values."""
        a = [1.42, -0.4, 0.23]
        softmax = Softmax()
        py_hat = softmax.forward(a)
        py_hat_truth = [0.682, 0.111, 0.207]
        np.testing.assert_array_almost_equal(py_hat, py_hat_truth,
                                             decimal=3, verbose=True)
        grad = softmax.backward(gradient=np.array([1, 0, 0]))
        grad_truth = np.array([0.217, -0.075, -0.142])
        print("SOFTMAX GRADIENT", grad)
        np.testing.assert_array_almost_equal(grad, grad_truth,
                                             decimal=3, verbose=True)

    def test_forward(self):
        """Test softmax forward pass with known values."""
        a = [1.42, -0.4, 0.23]
        softmax = Softmax()
        py_hat = softmax.forward(a)
        py_hat_truth = [0.682, 0.111, 0.207]
        np.testing.assert_array_almost_equal(py_hat, py_hat_truth,
                                             decimal=3, verbose=True)
