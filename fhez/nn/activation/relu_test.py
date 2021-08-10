# @Author: George Onoufriou <archer>
# @Date:   2021-07-25T15:40:17+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-02T16:23:48+01:00
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

    # def test_forward_ndarray_encrypted(self, x=None, q=None):
    #     """Check Encrypted cyphertext passess through RELU forward."""
    #     # this is hella slow since the features are independent cyphertexts
    #     data = self.data[0]
    #     x = x if x is not None else Erray(data, **self.reseal_args)
    #     relu = RELU()
    #     acti = np.array(relu.forward(x))  # forward and decrypt
    #     truth = data * (data > 0)  # manual RELU calculation on original data
    #     # print("acti", acti)
    #     # print("truth", acti)
    #     # confirm if they are about the same
    #     np.testing.assert_array_almost_equal(acti, truth,
    #                                          decimal=1,
    #                                          verbose=True)

    def test_forwards_ndarray_encrypted(self, x=None, q=None):
        """Check Encrypted cyphertext passess through RELU forward."""
        # ok so it looks like we have a quirk in rearray since its treating
        # each cyphertext independentley, while the result is the same,
        # the compute time is orders of magnitude higher. from 4->42 seconds
        data = self.data
        x = x if x is not None else Erray(data, **self.reseal_args)
        relu = RELU(q=q)
        acti = relu.forward(x)  # forward and decrypt
        truth = data * (data > 0)  # manual RELU calculation on original data
        # print("acti", acti)
        # print("truth", acti)
        # confirm if they are about the same
        np.testing.assert_array_almost_equal(np.array(acti), truth,
                                             decimal=1,
                                             verbose=True)

    def test_backward_encrypted(self, x=None, q=None):
        """Check backward pass working properly calculating gradients."""
        data = self.data
        # data = np.array([1.0, 1.0, 0.0])
        x = x if x is not None else Erray(data, **self.reseal_args)
        relu = RELU(q=q)
        relu.forward(np.array(x).flat[0])  # second forward for second backward
        acti = relu.forward(x)  # forward
        acti_truth = data * (data > 0)
        self.assertEqual(acti_truth.shape, acti.shape)  # check shape unchanged
        plain_acti = np.array(acti)

        dfdx = relu.backward(0.5 - plain_acti)
        dfdx_single = relu.backward(0.5 - plain_acti.flat[0])
        self.assertEqual(dfdx.shape, plain_acti.shape)  # check shape unchanged
        self.assertEqual(dfdx.flat[0].shape, tuple())
        self.assertEqual(dfdx_single.shape, tuple())
        self.assertEqual(dfdx.flat[0],
                         dfdx_single)

        self.assertEqual(len(relu.gradients), 2)  # check is only set of grads
        self.assertIsInstance(relu.gradients[0], dict)

    def test_gradients(self):
        """Check gradients are calculated properly to known correct."""
        node = RELU()
        # dfdx check
        dfdx = node.local_dfdx(x=5, q=2)
        dfdx_truth = 2.622  # from manual calculation
        np.testing.assert_array_almost_equal(dfdx, dfdx_truth, decimal=3,
                                             verbose=True)
        # dfxq check
        dfdq = node.local_dfdq(x=5, q=2)
        dfdq_truth = -2.546
        np.testing.assert_array_almost_equal(dfdq, dfdq_truth, decimal=3,
                                             verbose=True)

    def test_updates(self):
        """Check that updates are atleast happening.

        Tests for the optimizer are part of the respective optimizer not us.
        """
        node = RELU()
        acti = node.forward(self.data)
        node.backward(0.5 - acti)
        q_original = node.q
        node.updates()
        # checking if weight has been changed we dont care if correct as
        # that is a test for the optimiser itself
        # NOTE: since the data is multidimensional Q will also have
        # become multidimensional unless there is a savetey guard on the setter
        self.assertNotEqual(q_original, node.q.flatten()[0])
