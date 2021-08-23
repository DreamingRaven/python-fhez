# @Author: George Onoufriou <archer>
# @Date:   2021-08-23T17:19:31+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-23T20:20:13+01:00

import time
import unittest
import numpy as np

from fhez.nn.graph.prefab import cnn_classifier


class FiringTest(unittest.TestCase):
    """Test linear activation function."""

    def setUp(self):
        """Start timer and init variables."""
        self.start_time = time.time()

    def tearDown(self):
        """Calculate and print time delta."""
        t = time.time() - self.start_time
        print('%s: %.3f' % (self.id(), t))

    @property
    def graph(self):
        """Get neuron/ computational graph to test against."""
        return cnn_classifier(10)
