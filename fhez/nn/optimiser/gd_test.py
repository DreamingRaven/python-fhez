# @Author: George Onoufriou <archer>
# @Date:   2021-07-27T05:13:25+01:00
# @Last modified by:   archer
# @Last modified time: 2021-07-27T05:16:59+01:00

import time
import unittest
import numpy as np

from fhez.nn.optimiser.gd import GD
from fhez.rearray import ReArray as Erray  # aliasing for later adjust


class GDTest(unittest.TestCase):
    """Test stocastic gradient descent."""

    def setUp(self):
        """Start timer and init variables."""
        self.start_time = time.time()

    def tearDown(self):
        """Calculate and print time delta."""
        t = time.time() - self.start_time
        print('%s: %.3f' % (self.id(), t))

    def test_init(self):
        """Check SGD can be initialised using defaults."""
        optimiser = GD()
        self.assertIsInstance(optimiser, GD)
