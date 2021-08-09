# @Author: George Onoufriou <archer>
# @Date:   2021-08-09T16:48:31+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-09T17:02:24+01:00

import time
import unittest
import numpy as np

from fhez.nn.activation.sigmoid import Sigmoid


class Sigmoid_Test(unittest.TestCase):

    def setUp(self):
        """Start timer and init variables."""
        self.start_time = time.time()

    def tearDown(self):
        """Calculate and print time delta."""
        t = time.time() - self.start_time
        print('%s: %.3f' % (self.id(), t))

    @property
    def x(self):
        x = [
            [0.0, 0.2, 0.3],
            [0.8, 0.9, 1.0]
        ]
        return np.array(x)

    def test_forward(self):
        print(self.x.shape)
        activation_function = Sigmoid()
        activation = activation_function.forward(self.x)
        truth = activation_function.sigmoid(self.x)
        print(activation)
        print(truth)

    def test_backward(self):
        activation_function = Sigmoid()
        activation = activation_function.forward(self.x)
        df_dx = activation_function.backward(np.array([1]))
        print(activation.shape)
        print(df_dx.shape)

    def test_sigmoid(self):
        raise NotImplementedError("Sigmoid implementation is incomplete.")
