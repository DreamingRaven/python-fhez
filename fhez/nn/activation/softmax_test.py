# @Author: George Onoufriou <archer>
# @Date:   2021-08-02T22:01:04+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-09T15:24:27+01:00

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
        a = [1.0, 2.0, 3.0]
        softmax = Softmax()
        py_hat = softmax.forward(a)
        truth = [0.024, 0.064, 0.175]
        softmax.backward(gradient=np.array([0, 0, 0, 1, 0, 0, 0]))

    def test_forward(self):
        x = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]
        softmax = Softmax()
        a = softmax.forward(x)
        truth = [0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175]
        np.testing.assert_array_almost_equal(a, truth,
                                             decimal=3, verbose=True)
