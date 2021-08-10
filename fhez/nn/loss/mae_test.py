# @Author: George Onoufriou <archer>
# @Date:   2021-07-30T12:07:12+01:00
# @Last modified by:   archer
# @Last modified time: 2021-07-30T14:36:26+01:00

import time
import unittest
import numpy as np
from fhez.nn.loss.mae import MAE
from fhez.nn.activation.linear import Linear


class MAETest(unittest.TestCase):

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
        mae = MAE()

    def test_loss_equal(self):
        """Check when completeley accurate that gradient is 0."""
        mae = MAE()
        node = Linear()
        x = np.array([1, 0.2, 3])
        y_hat = np.sum(node.forward(x))
        y = y_hat
        loss = mae.forward(y=y, y_hat=y_hat)
        out = mae.backward(gradient=loss)
        self.assertEqual(out, np.array([0]))

    def test_loss_different(self):
        """Check gradient calculation when not 100% accurate."""
        mae = MAE()
        y_hat = np.array([0.7, 0.5, 0.3])
        y = np.array([0.3, 0.5, 0.7])
        loss = mae.forward(y=y, y_hat=y_hat)
        out = mae.backward(gradient=loss)
        # since inputs difference are equidistent the average gradient is 0
        self.assertEqual(out, np.array([0]))
