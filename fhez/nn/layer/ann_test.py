"""Artificial Neural Network (ANN) tests."""
# @Author: George Onoufriou <archer>
# @Date:   2021-07-24T15:33:14+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-09T20:56:07+01:00
import unittest
import logging as logger
import numpy as np
import time

import seal
from fhez.rearray import ReArray
from fhez.nn.layer.ann import ANN


class Ann_Tests(unittest.TestCase):

    @property
    def data_shape(self):
        return (3, 32, 32, 3)

    @property
    def data(self):
        """Get some generated data."""
        # array = np.arange(1*32*32*3)
        # array.shape = (1, 32, 32, 3)
        array = np.random.rand(*self.data_shape)
        return array

    @property
    def reseal_args(self):
        """Get some reseal arguments for encryption."""
        return {
            "scheme": seal.scheme_type.CKKS,
            "poly_modulus_degree": 8192*2,  # 438
            # "coefficient_modulus": [60, 40, 40, 60],
            "coefficient_modulus":
                [45, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 45],
            "scale": pow(2.0, 30),
            "cache": True,
        }

    def setUp(self):
        """Start timer and init variables."""
        self.weights = (3,)  # if tuple allows cnn to initialise itself
        self.stride = [1, 3, 3, 3]  # stride list per-dimension
        self.bias = 0  # assume no bias at first

        self.startTime = time.time()

    def tearDown(self):
        """Calculate and print time delta."""
        t = time.time() - self.startTime
        print('%s: %.3f' % (self.id(), t))

    def test_test(self):
        """Check our testing values meet requirements."""
        ann = ANN(weights=self.weights, bias=self.bias)
        # check data is the shape we desire/ gave it to generate
        self.assertEqual(self.data.shape, self.data_shape)
        # check weights length matches first dim of data
        self.assertEqual(len(ann.weights), self.data_shape[0])

    def test_init(self):
        """Check object initialisation works."""
        ANN(weights=self.weights, bias=self.bias)

    def test_forward(self, data=None):
        """Check forward pass works as expected."""
        ann = ANN(weights=self.weights, bias=self.bias)
        data = self.data if data is None else data
        acti = ann.forward(x=data)
        return acti

    def test_forward_enc(self):
        """Check encrypted forward pass works as expected."""
        data = self.data[0]
        edata = ReArray(data, **self.reseal_args)
        self.assertTrue(np.array(edata).shape == data.shape)
        self.assertIsInstance(edata, ReArray)
        acti = self.test_forward(data=edata)
        self.assertIsInstance(acti, ReArray)
        plain = np.array(acti)
        self.assertIsInstance(plain, np.ndarray)


if __name__ == "__main__":
    logger.basicConfig(  # filename="{}.log".format(__file__),
        level=logger.INFO,
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S")
    # run all the unit-tests
    print("now testing:", __file__, "...")
    unittest.main()
