# @Author: George Onoufriou <archer>
# @Date:   2021-08-02T22:04:55+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-04T16:13:51+01:00

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
        loss = CategoricalCrossentropy()
        self.assertIsInstance(loss, CategoricalCrossentropy)

    @property
    def y(self):
        """Ground truth classification."""
        return np.array([
            1,
            0,
            0,
            0
        ])

    @property
    def y_hat(self):
        """Classification probability score (must add up to 1)."""
        t = np.array([
            0.3,
            0.6,
            0.05,
            0.05
        ])
        # output of a softmax layer will always be normalised to add up to 1
        self.assertEqual(np.sum(t), 1)
        return t

    def test_forward(self):
        loss_func = CategoricalCrossentropy()
        loss = loss_func.forward(y=self.y, y_hat=self.y_hat)
        print("CC LOSS:", loss)

    def test_backward(self):
        loss_func = CategoricalCrossentropy()
        loss = loss_func.forward(y=self.y, y_hat=self.y_hat)
        class_grads = loss_func.backward(loss)
        self.assertEqual(len(class_grads), len(self.y_hat))

    def test_completed_cce(self):
        raise NotImplementedError(
            "CategoricalCrossentropy has not been completed.")
