# @Author: George Onoufriou <archer>
# @Date:   2021-08-02T22:01:04+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-23T14:30:35+01:00

import time
import unittest
import numpy as np

from fhez.nn.activation.argmax import Argmax


class ArgmaxTest(unittest.TestCase):
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
        activation = Argmax()
        self.assertIsInstance(activation, Argmax)

    def test_backward(self):
        """Test argmax backward pass with known values."""
        a = [1.42, -0.4, 0.23]
        argmax = Argmax()
        py_hat = argmax.forward(a)
        py_hat_truth = [1, 0, 0]
        np.testing.assert_array_almost_equal(py_hat, py_hat_truth,
                                             decimal=3, verbose=True)
        grad = argmax.backward(gradient=np.array([1, 0, 0]))
        grad_truth = np.array([1, 0, 0])
        np.testing.assert_array_almost_equal(grad, grad_truth,
                                             decimal=3, verbose=True)

    def test_forward(self):
        """Test argmax forward pass with known values."""
        a = [1.42, -0.4, 0.23]
        argmax = Argmax()
        py_hat = argmax.forward(a)
        py_hat_truth = [1, 0, 0]
        np.testing.assert_array_almost_equal(py_hat, py_hat_truth,
                                             decimal=3, verbose=True)
