#!/usr/bin/env python3

# @Author: GeorgeRaven <archer>
# @Date:   2020-09-16T11:33:51+01:00
# @Last modified by:   archer
# @Last modified time: 2021-07-23T16:16:50+01:00
# @License: please see LICENSE file in project root

import unittest
import logging as logger
import numpy as np

from tqdm import tqdm
from fhez.rearray import ReArray
from fhez.nn.layer.layer import Layer
from fhez.nn.graph.node import Node


class ANN(Node):
    """Dense artificial neural network as computational graph."""

    def __init__(self, weights: np.array = None, bias: int = None):
        """Initialise dense net."""
        if weights:
            self.weights = weights
        if bias:
            self.bias = bias

    @property
    def weights(self):
        """Get the current weights."""
        if self.__dict__.get("_weights") is None:
            self._weights = np.array([])
        return self._weights

    @weights.setter
    def weights(self, weights: np.ndarray):
        """Set the NN weights or let it self initialise."""
        if isinstance(weights, tuple):
            # if given a tuple will self initialise weights
            # for now this is done at random
            weights = np.random.rand(*weights)
        self._weights = weights

    @property
    def bias(self):
        """Get ANN sum of products bias."""
        pass

    @bias.setter
    def bias(self):
        """Set ANN sum of products bias."""
        pass

    def forward(self, x):
        """Compute forward pass of neural network."""
        # check that first dim matches so they can loop together
        if len(x) != len(self.weights):
            raise ValueError("Mismatched shapes {}, {}".format(
                len(x),
                self.weights[0]))
        # map - product of weight
        weighted = x * self.weights
        # reduce - sum of products
        sum = np.sum(weighted, axis=0)  # sum over only first axis
        self.inputs.append(x)
        return sum

    def backward(self, gradient):
        """Compute backward pass of neural network."""
        return gradient

    def update(self):
        """Update weights and bias of the network stocastically."""

    def updates(self):
        """Update weights and bias as one batch all together."""

    @property
    def cost(self):
        """Get no cost of a this node."""
        return 2


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
        import time

        self.weights = (3,)  # if tuple allows cnn to initialise itself
        self.stride = [1, 3, 3, 3]  # stride list per-dimension
        self.bias = 0  # assume no bias at first

        self.startTime = time.time()

    def tearDown(self):
        """Calculate and print time delta."""
        import time  # dont want time to be imported unless testing as unused
        t = time.time() - self.startTime
        print('%s: %.3f' % (self.id(), t))

    def test_test(self):
        """Check our testing values meet requirements."""
        # check data is the shape we desire/ gave it to generate
        self.assertEqual(self.data.shape, self.data_shape)
        # check weights length matches first dim of data
        self.assertEqual(len(self.weights), self.data_shape[0])
        # check data is between 0-1
        self.assertLessEqual(self.data[0], 1)

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
        data = ReArray(self.data, **self.reseal_args)
        self.assertIsInstance(data, ReArray)
        acti = self.test_forward(data=data)
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
