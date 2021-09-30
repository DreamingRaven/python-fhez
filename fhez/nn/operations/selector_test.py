# @Author: George Onoufriou <archer>
# @Date:   2021-09-21T12:44:39+01:00
# @Last modified by:   archer
# @Last modified time: 2021-09-21T13:19:22+01:00

import time
import unittest
import numpy as np
from fhez.nn.operations.selector import Selector


class SelectorTest(unittest.TestCase):
    """Test Selector operation node."""

    @property
    def data_shape(self):
        """Define desired data shape."""
        return (3, 32, 32, 3)

    @property
    def data(self):
        """Get some generated data."""
        array = np.random.rand(*self.data_shape)
        return array

    @property
    def reseal_args(self):
        """Get some reseal arguments for encryption."""
        return {
            "scheme": 2,  # seal.scheme_type.CKK,
            "poly_modulus_degree": 8192*2,  # 438
            # "coefficient_modulus": [60, 40, 40, 60],
            "coefficient_modulus":
                [45, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 45],
            "scale": pow(2.0, 30),
            "cache": True,
        }

    def setUp(self):
        """Start timer and init variables."""
        self.weights = (1,)  # if tuple allows cnn to initialise itself

        self.startTime = time.time()

    def tearDown(self):
        """Calculate and print time delta."""
        t = time.time() - self.startTime
        print('%s: %.3f' % (self.id(), t))

    def test_forward_selection(self):
        """Check selection of forward only passing desired."""
        node = Selector(forward=[False, True])
        out = node.forward(x=[5, 6])
        self.assertEqual(np.array(out).shape, tuple())
        np.testing.assert_array_almost_equal(out, 6,
                                             decimal=1,
                                             verbose=True)

    def test_forward_selection_multidim(self):
        """Check selection of forward only passing desired."""
        node = Selector(forward=[False, True])
        out = node.forward(x=[[4, 5], [6, 7]])
        self.assertEqual(np.array(out).shape, (2,))
        np.testing.assert_array_almost_equal(out, [6, 7],
                                             decimal=1,
                                             verbose=True)

    def test_backward_selection(self):
        """Check selection of backward signals only passing desired."""
        node = Selector(backward=[False, True])
        node.forward(x=[5.0, 6.0])
        out = node.backward(gradient=[7.0, 8.0])
        self.assertEqual(np.array(out).shape, tuple())
        np.testing.assert_array_almost_equal(out, 8,
                                             decimal=1,
                                             verbose=True)

    def test_forward_selection_none(self):
        """Check selection of forward only passing desired."""
        node = Selector()
        out = node.forward(x=[5.0, 6.0])
        self.assertEqual(np.array(out).shape, (2,))
        np.testing.assert_array_almost_equal(out, [5, 6],
                                             decimal=1,
                                             verbose=True)

    def test_backward_selection_none(self):
        """Check selection of backward signals only passing desired."""
        node = Selector()
        node.forward(x=[5, 6])
        out = node.backward(gradient=[7, 8])
        self.assertEqual(np.array(out).shape, (2,))
        np.testing.assert_array_almost_equal(out, [7, 8],
                                             decimal=1,
                                             verbose=True)
