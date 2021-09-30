"""Linear activation tests."""
# @Author: George Onoufriou <archer>
# @Date:   2021-07-26T17:00:08+01:00
# @Last modified by:   archer
# @Last modified time: 2021-09-10T16:50:27+01:00

import time
import unittest
import numpy as np

from fhez.nn.activation.linear import Linear
from fhez.rearray import ReArray as Erray  # aliasing for later adjust


class LinearTest(unittest.TestCase):
    """Test linear activation function."""

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

        self.start_time = time.time()

    def tearDown(self):
        """Calculate and print time delta."""
        t = time.time() - self.start_time
        print('%s: %.3f' % (self.id(), t))

    def test_forward(self):
        """Check linear activation function works with enc and plaintext."""
        node = Linear()
        data = self.data
        plain_out = node.forward(x=data)
        enc_out = node.forward(Erray(data, **self.reseal_args))
        # check answers match
        np.testing.assert_array_almost_equal(plain_out, np.array(enc_out),
                                             decimal=2, verbose=True)

        # check the answer
        np.testing.assert_array_almost_equal(plain_out, data,
                                             decimal=2, verbose=True)

    def test_backward(self):
        """Check backward propogation giving correct gradients."""
        node = Linear()
        data = self.data
        enc_out = node.forward(Erray(data, **self.reseal_args))
        loss = 0.5 - np.array(enc_out)  # pretend target is 0.5
        dfdx = node.backward(loss)
        dfdx_truth = 1 * loss  # whatever the calculation for dfdx should be
        np.testing.assert_array_almost_equal(dfdx, dfdx_truth,
                                             decimal=2, verbose=True)

    def test_update(self):
        """Check updates occured or did not occur properly."""
        node = Linear()
        data = self.data
        enc_out = node.forward(Erray(data, **self.reseal_args))
        loss = 0.5 - np.array(enc_out)  # pretend target is 0.5
        node.backward(loss)
        node.update()

    def test_updates(self):
        """Check updates occured or did not occur properly."""
        node = Linear()
        data = self.data
        enc_out = node.forward(Erray(data, **self.reseal_args))
        loss = 0.5 - np.array(enc_out)  # pretend target is 0.5
        node.backward(loss)
        node.updates()

    def test_getstate_setstate(self):
        """Check setstate getstate functionality."""
        obj_dump = Linear(m=np.array([5]), c=np.array([3]))
        obj_load = Linear()
        # getting simple dictionary representation of class
        d = obj_dump.__getstate__()
        # check is dict properly
        self.assertIsInstance(d, dict)
        # check repr works properly returning a string
        self.assertIsInstance(repr(obj_dump), str)
        # recreate original object in new object
        obj_load.__setstate__(d)
        # manually comparing each part of our dictionaries as we cant rely on
        # assertEqual to do the whole dictionary when it comes to multidim
        # numpy arrays
        for key, value in obj_dump.__dict__.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_almost_equal(obj_dump.__dict__[key],
                                                     value,
                                                     decimal=1,
                                                     verbose=True)
            else:
                self.assertEqual(obj_dump.__dict__[key], value)
