# @Author: George Onoufriou <archer>
# @Date:   2021-08-02T22:04:55+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-02T22:08:02+01:00

import time
import unittest
import numpy as np

from fhez.nn.loss.categorical_crossentropy import CategoricalCrossentropy


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
        activation = CategoricalCrossentropy()
        self.assertIsInstance(activation, CategoricalCrossentropy)

    def test_completed_cce(self):
        raise NotImplementedError(
            "CategoricalCrossentropy has not been completed.")
