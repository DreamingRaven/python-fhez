# @Author: George Onoufriou <archer>
# @Date:   2021-08-18T15:35:00+01:00
# @Last modified by:   archer
# @Last modified time: 2021-10-21T12:57:25+01:00

import time
import unittest
import numpy as np
from fhez.rearray import ReArray
from fhez.nn.operations.rotate import Rotate


class RotateTest(unittest.TestCase):
    """Test rotate operation node."""

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
        node = Rotate(provider=ReArray, **self.reseal_args)
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
        node = Rotate()
        plaintext = node.forward(x)
        self.assertIsInstance(plaintext, np.ndarray)
        np.testing.assert_array_almost_equal(plaintext, x,
                                             decimal=4,
                                             verbose=True)

    def test_forward_plain_sum(self):
        """Check rotation sums desired axis."""
        x = self.data
        node = Rotate(sum_axis=1)
        plaintext = node.forward(x)
        self.assertIsInstance(plaintext, np.ndarray)
        np.testing.assert_array_almost_equal(plaintext, x,
                                             decimal=4,
                                             verbose=True)

    def test_forward_decrypt(self):
        """Decrypt forward pass without re-encryption."""
        x = self.data
        cypher = ReArray(x, self.reseal_args)
        node = Rotate()
        plaintext = node.forward(cypher)
        self.assertIsInstance(plaintext, np.ndarray)
        np.testing.assert_array_almost_equal(plaintext, x,
                                             decimal=4,
                                             verbose=True)

    def test_forward_with_encryptor(self):
        """Check cyphertext is generated from encryptor properly."""
        plaintext = self.data
        encryptor = ReArray(np.array([1]), **self.reseal_args)
        node = Rotate(encryptor=encryptor)
        cyphertext = node.forward(plaintext)
        self.assertIsInstance(cyphertext, ReArray)
        x = np.array(cyphertext)
        np.testing.assert_array_almost_equal(x, plaintext,
                                             decimal=4,
                                             verbose=True)

    def test_forward_rotation(self):
        """Check that encryption parameters have been changed, on axis."""
        x = self.data
        encryptor = ReArray(np.array([1]), {
            "scheme": 2,  # seal.scheme_type.CKK,
            "poly_modulus_degree": 8192*2,  # 438
            # "coefficient_modulus": [60, 40, 40, 60],
            "coefficient_modulus":
                [45, 30, 30, 30, 30, 45],  # coefficient mod length of 6
            "scale": pow(2.0, 30),
            "cache": True,
        })
        node = Rotate(encryptor=encryptor)
        cyphertext_in = ReArray(x, **self.reseal_args)
        cyphertext_out = node.forward(cyphertext_in)
        self.assertIsInstance(cyphertext_out, ReArray)
        out_cm = cyphertext_out.cyphertext[0].coefficient_modulus
        in_cm = cyphertext_in.cyphertext[0].coefficient_modulus
        self.assertNotEqual(in_cm, out_cm)

    def test_forward_encrypt_axis(self):
        x = self.data
        axis = 1
        encryptor = ReArray(np.array([1]), {
            "scheme": 2,  # seal.scheme_type.CKK,
            "poly_modulus_degree": 8192*2,  # 438
            # "coefficient_modulus": [60, 40, 40, 60],
            "coefficient_modulus":
                [45, 30, 30, 30, 30, 45],  # coefficient mod length of 6
            "scale": pow(2.0, 30),
            "cache": True,
        })
        node = Rotate(encryptor=encryptor, axis=1)
        cyphertext_in = ReArray(x, **self.reseal_args)
        cyphertext_lst_out = node.forward(cyphertext_in)
        self.assertIsInstance(cyphertext_lst_out, list)
        self.assertIsInstance(cyphertext_lst_out[0], ReArray)
        np.testing.assert_array_almost_equal(cyphertext_lst_out, cyphertext_in,
                                             decimal=4,
                                             verbose=True)

    def test_backward(self):
        """Check gradients are mapped."""
        node = Rotate()
        grad = np.array([1, 2, 3])
        local_grad = node.backward(grad)
        # gradient input should be the same as gradient output as mapped
        np.testing.assert_array_almost_equal(local_grad, grad,
                                             decimal=1,
                                             verbose=True)

    def test_flatten(self):
        """Check that flattening occurs as expected."""
        node = Rotate(flatten=True)
        x = np.ones((28, 28))
        flattened = node.forward(x)
        np.testing.assert_array_almost_equal(flattened, x.flatten(),
                                             decimal=4,
                                             verbose=True)
