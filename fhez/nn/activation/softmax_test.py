# @Author: George Onoufriou <archer>
# @Date:   2021-08-02T22:01:04+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-02T22:04:07+01:00

import time
import unittest
import numpy as np

from fhez.nn.activation.softmax import Softmax


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

    def test_completed_softmax(self):
        raise NotImplementedError("Softmax has not been completed.")
