"""Test Loss Functions."""
# @Author: George Onoufriou <archer>
# @Date:   2021-07-28T21:43:54+01:00
# @Last modified by:   archer
# @Last modified time: 2021-07-29T10:51:38+01:00


import time
import unittest
import numpy as np

from fhez.nn.loss.loss import mae, mse, rmse


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
        self.assertEqual(self.y.shape, self.y_hat.shape)

    def test_mae(self):
        """Check MAE loss function working properly."""
        loss = mae(y=self.y, y_hat=self.y_hat)
        loss_truth = np.mean(np.abs(self.y - self.y_hat))
        # self.assertEqual(loss_truth, 2)
        self.assertEqual(loss, loss_truth)

    def test_mse(self):
        """Check MSE loss function working properly."""
        loss = mse(y=self.y, y_hat=self.y_hat)
        loss_truth = np.mean(np.square(self.y - self.y_hat))
        # np.testing.assert_almost_equal(loss_truth, 6.666, decimal=3)
        self.assertEqual(loss, loss_truth)
        return loss

    def test_rmse(self):
        """Check RMSE loss function working properly."""
        loss = rmse(y=self.y, y_hat=self.y_hat)
        loss_truth = np.sqrt(self.test_mse())
        # np.testing.assert_almost_equal(loss_truth, 2.581, decimal=3)
        self.assertEqual(loss, loss_truth)
