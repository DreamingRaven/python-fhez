#!/usr/bin/env python3

# @Author: GeorgeRaven <archer>
# @Date:   2020-09-16T11:33:51+01:00
# @Last modified by:   archer
# @Last modified time: 2021-07-24T23:26:57+01:00
# @License: please see LICENSE file in project root

import logging as logger
import numpy as np
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
            logger.warning("{}.weights called before initialisation".format(
                self.__class__.__name__))
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
        if self.__dict__.get("_bias") is None:
            logger.warning("{}.bias called before initialisation".format(
                self.__class__.__name__))
            self._bias = 0
        return self._bias

    @bias.setter
    def bias(self, bias):
        """Set ANN sum of products bias."""
        self._bias = bias

    def forward(self, x):
        """Compute forward pass of neural network."""
        # check that first dim matches so they can loop together
        if len(x) != len(self.weights):
            raise ValueError("Mismatched shapes {}, {}".format(
                len(x),
                self.weights[0]))
        # map - product of weight
        weighted = x * self.weights
        # reduce - sum of products using dispatcher
        sum = np.sum(weighted, axis=0)  # sum over only first axis
        # now save the input we originally got since it has been processed
        self.inputs.append(x)
        return sum

    def backward(self, gradient):
        """Compute backward pass of neural network."""
        # dfdx
        # dfdw
        # dfdb
        return gradient

    def update(self):
        """Update weights and bias of the network stocastically."""
        # dfdw
        # dfdb

    def updates(self):
        """Update weights and bias as one batch all together."""
        # dfdw
        # dfdb

    @property
    def cost(self):
        """Get no cost of a this node."""
        return 2
