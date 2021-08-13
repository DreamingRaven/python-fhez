# @Author: George Onoufriou <archer>
# @Date:   2021-08-10T14:36:02+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-13T12:54:39+01:00

import time
import unittest
import numpy as np

from fhez.nn.layer.cnn import CNN


class CNNTest(unittest.TestCase):
    """Test CNN node abstraction."""

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
        cnn = CNN()

    def test_forward(self):
        """Test CNN filter and sum applied correctly."""
        weights = np.ones((3, 3, 3))/2
        cnn = CNN(weights=weights)
        a = cnn.forward(x=self.data)
        print("cnn.forward", a)

    def test_backward(self):
        """Test CNN gradient calculated correctly."""
        weights = np.ones((3, 3, 3))/2
        cnn = CNN(weights=weights)
        cnn.forward(x=self.data)
        grad = cnn.backward(gradient=1)
        print("cnn.backward", grad)

# class cnn_tests(unittest.TestCase):
#     """Unit test class aggregating all tests for the cnn class"""
#
#     @property
#     def data(self):
#         array = np.random.rand(2, 15, 15, 3)
#         return array
#
#     @property
#     def reseal_args(self):
#         return {
#             "scheme": seal.scheme_type.CKKS,
#             "poly_modulus_degree": 8192*2,  # 438
#             # "coefficient_modulus": [60, 40, 40, 60],
#             "coefficient_modulus":
#                 [45, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 45],
#             "scale": pow(2.0, 30),
#             "cache": True,
#         }
#
#     def setUp(self):
#         import time
#
#         self.weights = (1, 3, 3, 3)  # tuple allows cnn to initialise itself
#         self.stride = [1, 3, 3, 3]  # stride list per-dimension
#         self.bias = 0  # assume no bias at first
#
#         self.startTime = time.time()
#
#     def tearDown(self):
#         import time  # dont want time to be imported unless testing as unused
#         t = time.time() - self.startTime
#         print('%s: %.3f' % (self.id(), t))
#
#     def test_cnn_whole(self):
#         from fhez.nn.layer.ann import Layer_ANN
#
#         # CREATE IDENTICAL CNN LAYERS
#         cnn = Layer_CNN(weights=self.weights,
#                         bias=self.bias,
#                         stride=self.stride,
#                         )
#         cnn_copy = copy.deepcopy(cnn)
#
#         # CREATE PLACEHOLDER FOR ANN
#         dense = None
#         dense_copy = None
#         previous_loss_np = None
#
#         # DEFINE DATA (DONT REGENERATE)
#         data = self.data
#
#         for i in range(10):
#
#             # FORWARD PASS CNN
#             re_acti = cnn.forward(x=ReArray(data, **self.reseal_args))
#             np_acti = cnn_copy.forward(x=data)
#             self.assertEqual(re_acti.shape, (25,)+data.shape)
#             self.assertEqual(np_acti.shape, (25,)+data.shape)
#
#             # CREATE IDENTICAL ANN LAYERS
#             if dense is None:
#                 dense = Layer_ANN(weights=(len(re_acti),), bias=0)
#                 dense_copy = copy.deepcopy(dense)
#
#             # FORWARD PASS ANN
#             re_fwd = dense.forward(re_acti)
#             np_fwd = dense_copy.forward(np_acti)
#             self.assertEqual(re_fwd.shape, data.shape)
#             self.assertEqual(np_fwd.shape, data.shape)
#             y_hat_re = np.sum(re_fwd, axis=tuple(range(1, re_fwd.ndim)))
#             y_hat_np = np.sum(np_fwd, axis=tuple(range(1, re_fwd.ndim)))
#
#             # CALCULATE AND DISPLAY LOSS
#             re_loss = 1 - y_hat_re.mean()
#             np_loss = 1 - y_hat_np.mean()
#             print("loss_re", re_loss)
#             print("y_hat_re avg", y_hat_re.mean(), "val:", y_hat_re)
#             print("loss_np", np_loss)
#             print("y_hat_np avg", y_hat_np.mean(), "val:", y_hat_np)
#             if previous_loss_np is not None:
#                 txt = "loss somehow more inacurate activations".format()
#                 # self.assertLess(abs(np_loss), abs(previous_loss_np), txt)
#             previous_loss_np = re_loss
#
#             # BACKWARD PASS CNN AND ANN
#             re_gradient = cnn.backward(dense.backward(re_loss))
#             np_gradient = cnn_copy.backward(dense_copy.backward(np_loss))
#             self.assertEqual(re_gradient.shape, (data.shape[0],))
#             self.assertEqual(np_gradient.shape, (data.shape[0],))
#
#             # UPDATE CNN AND ANN
#             dense.update()
#             dense_copy.update()
#             cnn.update()
#             cnn_copy.update()
#
#         # RESEAL VS NUMPY NOISE DIFFERENCE TESTING
#         self.assertListEqual(
#             np.around(np.array(re_acti), decimals=2).flatten().tolist(),
#             np.around(np.array(np_acti), decimals=2).flatten().tolist(),
#         )
#         self.assertListEqual(
#             np.around(np.array(re_fwd), decimals=2).flatten().tolist(),
#             np.around(np.array(np_fwd), decimals=2).flatten().tolist(),
#         )
#         self.assertListEqual(
#             np.around(np.array(y_hat_re), decimals=2).flatten().tolist(),
#             np.around(np.array(y_hat_np), decimals=2).flatten().tolist(),
#         )
#         self.assertListEqual(
#             np.around(np.array(re_gradient), decimals=2).flatten().tolist(),
#             np.around(np.array(np_gradient), decimals=2).flatten().tolist(),
#         )
#
#
# if __name__ == "__main__":
#     logger.basicConfig(  # filename="{}.log".format(__file__),
#         level=logger.INFO,
#         format="%(asctime)s %(levelname)s:%(message)s",
#         datefmt="%Y-%m-%dT%H:%M:%S")
#     # run all the unit-tests
#     print("now testing:", __file__, "...")
#     unittest.main()
