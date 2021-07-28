"""Test Loss Functions."""
# @Author: George Onoufriou <archer>
# @Date:   2021-07-28T21:43:54+01:00
# @Last modified by:   archer
# @Last modified time: 2021-07-28T22:05:56+01:00


import time
import unittest
import numpy as np


class LossTest(unittest.TestCase):
    """Test loss functions."""

    def setUp(self):
        """Start timer and init variables."""
        self.start_time = time.time()

    def tearDown(self):
        """Calculate and print time delta."""
        t = time.time() - self.start_time
        print('%s: %.3f' % (self.id(), t))

    def linear(self, x, m, c):
        """Calculate standard linear function for testing against."""
        return (m * x) + c

    @property
    def y(self):
        """Get gaff truth for comparison."""
        return np.array([
            1,
            2,
            3,
        ])

    @property
    def y_hat(self):
        """Get gaff truth for comparison."""
        return np.array([
            3,
            -2,
            3

        ])

    def test_test(self):
        """Check this test is being called."""
        pass

    def test_mae(self):
        """Check MAE loss function working properly."""
        pass
        # from fhez.nn.loss.loss import mae
        # mae(y=self.y, y_hat=self.y_hat)
