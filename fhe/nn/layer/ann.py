#!/usr/bin/env python3

# @Author: GeorgeRaven <archer>
# @Date:   2020-09-16T11:33:51+01:00
# @Last modified by:   archer
# @Last modified time: 2021-03-15T12:56:14+00:00
# @License: please see LICENSE file in project root

import logging as logger
import numpy as np
import unittest

from tqdm import tqdm

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

        sum = None
        for i in tqdm(range(len(x)), desc="{}.{}".format(
            self.__class__.__name__, "forward"),
            ncols=80, colour="blue"
        ):
            t = x[i] * self.weights[i]
            if sum is None:
                sum = t
            else:
                sum = sum + t
        # sum is not a single number, it is a multidimensional array
        # if you just add to this values will be broadcast and added to each
        # element individually, which makes the maths wrong I.E
        # 2 + (1+2+3) == (1+2/3) + (2+2/3) + (3+2/3) == 8 != (1+2)+(2+2)+(3+2)
        # we must divide by the number of elements in ONE batch
        # or else sum explodes
        elements_in_batch = sum.size/len(sum)
        sum += self.bias/elements_in_batch
        return self.activation_function.forward(sum)

    @Layer.bwd
    def backward(self, gradient, x):
        """Calculate the local gradient of this CNN.

        Given the gradient that precedes us,
        what is the local gradient after us.
        """
        # activation gradient already calculated for us
        ag = gradient
        # iterate over inputs and batches to get per-input-per-batch sums
        x = np.array(x)
        per_input_batch_sums = []
        for i in tqdm(range(len(x)), desc="{}.{}".format(
                self.__class__.__name__, "backward"),
            ncols=80, colour="blue"
        ):
            batch_sums = []
            for j in range(len(x[i])):
                sum = np.sum(x[i][j])
                batch_sums.append(sum)
            per_input_batch_sums.append(batch_sums)
        x = np.array(per_input_batch_sums)

        # save gradients of parameters with respect to output
        self.bias_gradient = 1 * ag
        self.weights_gradients = x * ag
        # calculate the average of these gradient between batches
        self.bias_gradient = np.sum(
            self.bias_gradient, axis=1)/self.bias_gradient.shape[1]
        self.weights_gradients = np.sum(
            self.weights_gradients, axis=1)/self.weights_gradients.shape[1]

        # calculate gradient with respect to fully connected ANN
        df_dx = np.array(list(map(lambda a: a * np.squeeze(ag, axis=0),
                                  self.weights)))
        return df_dx


class ann_tests(unittest.TestCase):
    """Unit test class aggregating all tests for the cnn class"""

    @property
    def data(self):
        # array = np.arange(1*32*32*3)
        # array.shape = (1, 32, 32, 3)
        array = np.random.rand(2, 3, 4)
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

    def test_ann_shapes(self):
        """Test both numpy and ReArray input result in desired ann output."""
        import copy

        x_dummy = ReArray(self.data, **self.reseal_args)
        x = []
        num_inputs = 5
        weights = np.random.rand(num_inputs)
        for i in range(num_inputs):
            r = ReArray(clone=x_dummy, plaintext=self.data)
            x.append(r)
        self.assertIsInstance(x[i], ReArray)

        ann = Layer_ANN(weights=weights,
                        bias=self.bias)
        np_ann = copy.deepcopy(ann)
        previous_activation = None

        for i in range(5):
            # FORWARD PASS TEST
            activations = ann.forward(x)
            np_activations = np_ann.forward(np.array(x))
            # check that output is equal in shape to any single input ndarray
            # also check that ReArray and numpy produce the same results
            self.assertEqual(activations.shape, x_dummy.shape)
            self.assertEqual(np_activations.shape, x_dummy.shape)
            # self.assertListEqual(
            #     np.around(np.array(activations),
            #               decimals=2).flatten().tolist(),
            #     np.around(np.array(np_activations),
            #               decimals=2).flatten().tolist(),
            # )
            a = np.array(np_activations)
            # print(a, "\n", a.shape)
            for _ in range(1, a.ndim):
                a = a.sum(axis=-1)
                # print(a, "\n", a.shape)
            a = np.around(a.mean(axis=0), decimals=5)

            # CHECK IF MORE ACCURATE PREDICTION
            print(a)
            current_loss = 1-a
            if previous_activation is not None:
                previous_loss = 1-previous_activation
                txt = "loss somehow more inacurate activations".format()
                self.assertLess(abs(current_loss), abs(previous_loss), txt)
            previous_activation = a

            # BACKWARD PASS TEST
            gradient = ann.backward(1-a)
            np_gradient = np_ann.backward(1-a)
            # we desire the resultant gradient to be of shape
            # (num_inputs, num_batches)
            desired_shape = (num_inputs,) + (len(x_dummy),)
            self.assertEqual(gradient.shape, desired_shape)
            self.assertEqual(np_gradient.shape, desired_shape)
            self.assertListEqual(
                np.around(np.array(gradient),
                          decimals=2).flatten().tolist(),
                np.around(np.array(np_gradient),
                          decimals=2).flatten().tolist(),
            )

            # UPDATE ANN
            ann.update(learning_rate=0.1)
            np_ann.update(learning_rate=0.1)


if __name__ == "__main__":
    logger.basicConfig(  # filename="{}.log".format(__file__),
        level=logger.INFO,
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S")
    # run all the unit-tests
    print("now testing:", __file__, "...")
    unittest.main()
