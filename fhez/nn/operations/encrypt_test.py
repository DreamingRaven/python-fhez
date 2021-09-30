# @Author: George Onoufriou <archer>
# @Date:   2021-08-18T15:35:00+01:00
# @Last modified by:   archer
# @Last modified time: 2021-09-14T13:51:53+01:00

import time
import unittest
import numpy as np
from fhez.rearray import ReArray
from fhez.nn.operations.encrypt import Encrypt


class EncryptTest(unittest.TestCase):
    """Test encrypt operation node."""

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

    def test_forward(self):
        """Check encryption is successfull with given params."""
        x = self.data
        node = Encrypt(provider=ReArray, **self.reseal_args)
        cyphertext = node.forward(x)
        # check parameters are exactly as provided
        # check cyphertext is expected type
        self.assertIsInstance(cyphertext, ReArray)
        # decyrpt again manually
        plaintext = np.array(cyphertext)
        np.testing.assert_array_almost_equal(plaintext, x,
                                             decimal=4,
                                             verbose=True)

    def test_forward_plain(self):
        """Check lack of encryption without params."""
        x = self.data
        node = Encrypt()
        plaintext = node.forward(x)
        self.assertIsInstance(plaintext, np.ndarray)
        np.testing.assert_array_almost_equal(plaintext, x,
                                             decimal=4,
                                             verbose=True)

    def test_backward(self):
        """Check gradients are mapped."""
        node = Encrypt()
        grad = np.array([1, 2, 3])
        local_grad = node.backward(grad)
        # gradient input should be the same as gradient output as mapped
        np.testing.assert_array_almost_equal(local_grad, grad,
                                             decimal=1,
                                             verbose=True)
