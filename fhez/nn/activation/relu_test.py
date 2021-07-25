# @Author: George Onoufriou <archer>
# @Date:   2021-07-25T15:40:17+01:00
# @Last modified by:   archer
# @Last modified time: 2021-07-25T18:36:57+01:00
import time
import unittest
import numpy as np

from fhez.nn.activation.relu import RELU
from fhez.rearray import ReArray as Erray  # aliasing for later adjust


class Relu_Test(unittest.TestCase):
    """Test RELU approximation node."""

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

    def test_test(self):
        """Check our testing values meet requirements."""
        # check data is the shape we desire/ gave it to generate
        self.assertEqual(self.data.shape, self.data_shape)

    def test_init(self):
        """Check initialisation of RELU."""
        self.assertIsInstance(RELU(), RELU, "RELU() not creating object RELU")
        self.assertEqual(RELU(q=5).q, 5, "dynamic range 'q' not set properly")

    def test_forward(self, x=None, q=None):
        """Check RELU forward pass producing good approximations."""
        # set input
        x = x if x is not None else np.array([1])
        # set range of approximation
        q = q if q is not None else 1
        r = RELU(q=q)
        acti = r.forward(x)
        truth = x if x >= 0 else 0
        self.assertEqual(
            acti.round(decimals=1), truth.round(decimals=1),
            "RELU-apx activation {} is not close to true RELU {}".format(
                acti, truth))

    def test_forward_ndarray(self, x=None, q=None):
        """Check that RELU operates on multidimensional arrays properly."""
        # set input
        x = x if x is not None else self.data[0]  # get only first batch
        # set range of approximation
        q = q if q is not None else 1
        r = RELU(q=q)
        acti = r.forward(x)
        # truth = np.maximum(x, 0)
        truth = x * (x > 0)
        # TODO: make this not ERROR, but instead FAIL if it does not pass!
        # like:
        # self.assertTrue(numpy.allclose(array1, array2,
        #                                rtol=1e-05, atol=1e-08))
        np.testing.assert_array_almost_equal(acti, truth,
                                             decimal=1,
                                             verbose=True)

    def test_forward_ndarray_encrypted(self, x=None, q=None):
        """Check Encrypted cyphertext passess through RELU forward."""
        x = x if x is not None else Erray(self.data[0], **self.reseal_args)
        print(x)
