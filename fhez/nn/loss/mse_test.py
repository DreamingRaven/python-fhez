# @Author: George Onoufriou <archer>
# @Date:   2021-07-30T12:07:12+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-01T20:48:10+01:00

import time
import unittest
import numpy as np
from fhez.nn.loss.mse import MSE
from fhez.nn.activation.linear import Linear


class mseTest(unittest.TestCase):

    def setUp(self):
        """Start timer and init variables."""
        self.start_time = time.time()

    def tearDown(self):
        """Calculate and print time delta."""
        t = time.time() - self.start_time
        print('%s: %.3f' % (self.id(), t))

    def test_test(self):
        """Check this test is being called."""

    def test_init(self):
        """Test init."""
        MSE()

    def test_loss_equal(self):
        """Check when completeley accurate that gradient is 0."""
        mse = MSE()
        node = Linear()
        x = np.array([1, 0.2, 3])
        y_hat = np.sum(node.forward(x))
        y = y_hat
        loss = mse.forward(y=y, y_hat=y_hat)
        out = mse.backward(gradient=loss)
        self.assertEqual(out, np.array([0]))

    def test_loss_different(self):
        """Check gradient calculation when not 100% accurate."""
        mse = MSE()
        y_hat = np.array([0.7, 0.5, 0.3])
        y = np.array([0.3, 0.5, 0.7])
        loss = mse.forward(y=y, y_hat=y_hat)
        out = mse.backward(gradient=loss)
        # since inputs difference are equidistent the average gradient is 0
        self.assertEqual(out, np.array([0]))

    def test_forward(self):
        """Check forward loss calculation."""
        mse = MSE()
        y_hat = np.array([1, 3])
        y = np.array([6, 6])
        loss = mse.forward(y=y, y_hat=y_hat)
        self.assertEqual(loss, np.array([17]))
