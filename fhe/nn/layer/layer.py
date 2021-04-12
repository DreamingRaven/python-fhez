#!/usr/bin/env python3

# @Author: GeorgeRaven <archer>
# @Date:   2020-09-16T11:33:51+01:00
# @Last modified by:   archer
# @Last modified time: 2021-03-11T12:55:56+00:00
# @License: please see LICENSE file in project root

import numpy as np
from fhe.nn.activation.sigmoid import Sigmoid_Approximation
from fhe.nn.block.block import Block


class Layer(Block):

    def __init__(self,
                 weights,
                 bias, stride=None,
                 activation=None,
                 branches=None):
        self.weights = weights
        self.bias = bias
        if branches is not None:
            self.branches = branches
        if activation:
            self.activation_function = activation
        if stride is not None:
            stride = np.broadcast_to(stride, self.weights.ndim)
            self.stride = stride

    @property
    def is_activation(self):
        """Are we an Activation function."""
        return False

    @property
    def is_layer(self):
        """Are we a Layer."""
        return True

    @property
    def branches(self):
        """How many branches will this layer be expected to sum.

        Branches is an int that represents how many forks/ branches/ timesteps
        the weights are expected to operate over.
        """
        if self.__dict__.get("_branches") is not None:
            return self._branches
        else:
            self.branches = 1
            return self.branches

    @branches.setter
    def branches(self, branches: int):
        if branches is not None:
            self._branches = branches
        else:
            raise ValueError("branches got {}, {}. Expected {}".format(
                type(branches), branches, int))

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
            # ensure initial product of weights * x is in range 0-1
            self._weights = self.weights / self.branches
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

    def update(self, learning_rate=None):
        """We need to update 2 things, both the biases and the weights"""
        lr = learning_rate if learning_rate is not None else 0.001
        lr = lr if isinstance(lr, np.ndarray) else np.array([lr])
        # new_parameter = old_parameter - learning_rate * gradient_of_parameter
        bias_shape_origin = self.bias.shape if isinstance(
            self.bias, np.ndarray) else np.array([self.bias]).shape
        self.bias = self.bias - (lr * self.bias_gradient)
        txt = "Shape changed: {} to: {} given LR: {} and gradient: {}".format(
            bias_shape_origin,
            np.array(self.bias).shape,
            np.array(lr).shape,
            self.bias_gradient.shape)
        assert (self.bias.shape == bias_shape_origin), txt

        weights_shape_origin = self.weights.shape
        self.weights = self.weights - (lr * self.weights_gradients)
        txt = "Shape changed: {} to: {} given LR: {} and gradient: {}".format(
            weights_shape_origin,
            self.weights.shape,
            np.array(lr).shape,
            self.weights_gradients.shape)
        assert (self.weights.shape == weights_shape_origin), txt
