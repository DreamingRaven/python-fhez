# @Author: George Onoufriou <archer>
# @Date:   2021-07-30T12:07:12+01:00
# @Last modified by:   archer
# @Last modified time: 2021-07-30T12:13:12+01:00

import time
import unittest
import numpy as np
from fhez.nn.loss.mae import MAE


class MAETest(unittest.TestCase):

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

    def test_init(self):
        """Test init."""
        mae = MAE()
