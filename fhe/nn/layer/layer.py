#!/usr/bin/env python3

# @Author: GeorgeRaven <archer>
# @Date:   2020-09-16T11:33:51+01:00
# @Last modified by:   archer
# @Last modified time: 2021-03-02T22:13:59+00:00
# @License: please see LICENSE file in project root

from tqdm import tqdm
import numpy as np
from fhe.nn.activation.sigmoid import Sigmoid_Approximation


class Layer():

    def __init__(self, weights, bias, stride=None, activation=None):
        self.weights = weights
        self.bias = bias
        if activation:
            self.activation_function = activation
        if stride:
            self.stride = stride

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        # initialise weights from tuple dimensions
        # TODO: properly implement xavier weight initialisation over np.rand
        if isinstance(weights, tuple):
            # https://www.coursera.org/specializations/deep-learning
            # https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78
            self._weights = np.random.rand(*weights)
        else:
            self._weights = weights

    @property
    def bias(self):
        if self.__dict__.get("_bias") is not None:
            return self._bias
        else:
            self.bias = 0
            return self.bias

    @bias.setter
    def bias(self, bias):
        self._bias = bias

    @property
    def stride(self):
        if self.__dict__.get("_stride") is not None:
            return self._stride
        else:
            self.stride = 1
            return self.stride

    @stride.setter
    def stride(self, stride):
        self._stride = stride

    @property
    def activation_function(self):
        if self.__dict__.get("_activation_function") is not None:
            return self._activation_function
        else:
            self.activation_function = Sigmoid_Approximation()
            return self.activation_function

    @activation_function.setter
    def activation_function(self, activation_function):
        self._activation_function = activation_function

    @property
    def x(self):
        if self.__dict__.get("_x") is not None:
            return self._x
        else:
            self.x = []
            return self.x

    @x.setter
    def x(self, x):
        self._x = x

    def fwd(func):
        """Forward decorator, unpacking + stashing x to use in backward."""

        def inner(self, x):
            self.x.append(x)
            return func(self, x)
        return inner

    def bwd(func):
        """Backward decorator to use decrypted or decrypt stashed x."""

        def inner(self, gradient=1):
            accumulator = []
            for i in tqdm(range(len(self.x)), desc=".bwd"):
                accumulator.append(func(self, gradient))
            return np.array(accumulator)
        return inner
