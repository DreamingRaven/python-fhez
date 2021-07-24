#!/usr/bin/env python3

# @Author: GeorgeRaven <archer>
# @Date:   2020-09-16T11:33:51+01:00
# @Last modified by:   archer
# @Last modified time: 2021-07-24T15:34:25+01:00
# @License: please see LICENSE file in project root

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
