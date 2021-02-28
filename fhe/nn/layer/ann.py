#!/usr/bin/env python3

# @Author: GeorgeRaven <archer>
# @Date:   2020-09-16T11:33:51+01:00
# @Last modified by:   archer
# @Last modified time: 2021-02-27T02:00:53+00:00
# @License: please see LICENSE file in project root

import logging as logger
import numpy as np
import unittest

import seal
from fhe.rearray import ReArray
from fhe.nn.layer.layer import Layer


class Layer_ANN(Layer):

    @Layer.fwd
    def forward(self, x: (np.array, ReArray)):
        """Take numpy array of objects or ReArray object to calculate y_hat."""
        # check that first dim matches so they can loop together
        if len(x) != len(self.weights):
            raise ValueError("Mismatched shapes {}, {}".format(
                len(x),
                self.weights[0]))

        self.x = x
        sum = None
        for i in range(len(x)):
            t = x[i] * self.weights[i]
            if sum is None:
                sum = t
            else:
                sum = sum + t
        print("********LAYER_ANN_AF***********")
        return self.activation_function.forward(sum)

    @Layer.bwd
    def backward(self, gradient):
        """Calculate the local gradient of this CNN.

        Given the gradient that precedes us,
        what is the local gradient after us.
        """
        # if gradient not given like if its the start of the chain then 1
        gradient = gradient if gradient is not None else 1
        # calculate gradient of activation function
        activation_gradient = self.activation_function.backward(gradient)
        # save gradients of parameters with respect to output
        self.bias_gradient = 1 * activation_gradient
        # self.weights_gradient = self.weights * \
        #     np.reshape(self.x,
        #                (self.x.shape[0], self.x.size/self.x.shape[0])) * \
        activation_gradient
        # calculate gradient with respect to fully connected ANN
        local_gradient = 1 * self.weights
        # return local gradient
        return local_gradient * activation_gradient

    def update(self):
        self.cc.update()


class ann_tests(unittest.TestCase):
    """Unit test class aggregating all tests for the cnn class"""

    @property
    def data(self):
        array = np.arange(1*32*32*3)
        array.shape = (1, 32, 32, 3)
        return array

    @property
    def reseal_args(self):
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
        import time

        self.weights = (1, 3, 3, 3)  # if tuple allows cnn to initialise itself
        self.stride = [1, 3, 3, 3]  # stride list per-dimension
        self.bias = 0  # assume no bias at first

        self.startTime = time.time()

    def tearDown(self):
        import time  # dont want time to be imported unless testing as unused
        t = time.time() - self.startTime
        print('%s: %.3f' % (self.id(), t))

    def test_numpy_matrix(self):
        ann = Layer_ANN(weights=self.weights,
                        bias=self.bias)
        ann.forward(x=self.data)

    def test_rearray(self):
        ann = Layer_ANN(weights=self.weights,
                        bias=self.bias)
        activations = ann.forward(x=ReArray(self.data, **self.reseal_args))
        accumulator = []
        for i in range(len(activations)):
            if(i % 10 == 0) or (i == len(activations) - 1):
                logger.debug("decrypting: {}".format(len(activations)))
            t = np.array(activations.pop(0))
            accumulator.append(t)
        plaintext_activations = np.around(np.array(accumulator), 2)
        compared_activations = np.around(ann.forward(x=self.data), 2)
        self.assertListEqual(plaintext_activations.flatten()[:200].tolist(),
                             compared_activations.flatten()[:200].tolist())
        self.assertListEqual


if __name__ == "__main__":
    logger.basicConfig(  # filename="{}.log".format(__file__),
        level=logger.DEBUG,
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S")
    # run all the unit-tests
    print("now testing:", __file__, "...")
    unittest.main()
