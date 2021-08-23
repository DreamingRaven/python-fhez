# @Author: George Onoufriou <archer>
# @Date:   2021-08-22T13:53:17+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-23T09:55:16+01:00

import time
import unittest


class FHEz_Test(unittest.TestCase):
    """Test FHEZ top level package."""

    def setUp(self):
        """Start timer and init variables."""
        self.startTime = time.time()

    def tearDown(self):
        """Calculate and print time delta."""
        t = time.time() - self.startTime
        print('%s: %.3f' % (self.id(), t))

    def test_import(self):
        """Test basic import of FHEZ."""
        import fhez
