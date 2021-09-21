"""Artificial Neural Network (ANN) as node abstraction."""
# @Author: GeorgeRaven <archer>
# @Date:   2020-09-16T11:33:51+01:00
# @Last modified by:   archer
# @Last modified time: 2021-09-21T20:08:33+01:00
# @License: please see LICENSE file in project root

import logging as logger
import numpy as np
from fhez.nn.graph.node import Node


class ANN(Node):
    """Dense artificial neural network as computational graph."""

    def __init__(self, weights: np.array = None, bias: int = None):
        """Initialise dense net."""
        if weights is not None:
            self.weights = weights
        if bias is not None:
            self.bias = bias

    @property
    def w(self):
        """Shorthand for weights."""
        return self.weights

    @w.setter
    def w(self, w):
        self.weights = w

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
    def b(self):
        """Shorthand for bias."""
        return self.bias

    @b.setter
    def b(self, b):
        self.bias = b

    @property
    def bias(self):
        """Get ANN sum of products bias."""
        if self.__dict__.get("_bias") is None:
            # logger.warning("{}.bias called before initialisation".format(
            #     self.__class__.__name__))
            self._bias = 0
        return self._bias

    @bias.setter
    def bias(self, bias):
        """Set ANN sum of products bias."""
        self._bias = bias

    def forward(self, x):
        r"""Compute forward pass of neural network.

        .. math::

            a^{(i)} = \sum_{t=0}^{T_x-1}(w^{<t>}x^{(i)<t>})+b

        """
        # check that first dim matches so they can loop together
        if len(x) != len(self.weights):
            raise ValueError("Mismatched shapes inp:{}, weights:{}".format(
                len(x),
                len(self.weights)))
        # map - product of weight
        weighted = np.multiply(x, self.weights)
        # reduce - sum of products using dispatcher
        sum = np.sum(weighted, axis=0)  # sum over only first axis
        # now save the input we originally got since it has been processed
        self.inputs.append(x)
        return np.add(sum, self.bias)

    def backward(self, gradient):
        r"""Compute backward pass of neural network.

        .. math::

            \frac{df}{db} = 1 \frac{dg}{dx}

            \frac{df}{dw^{<t>}} = x^{(i)<t>} \frac{dg}{dx}

            \frac{df}{dx^{(i)<t>}} = w^{<t>} \frac{dg}{dx}
        """
        x = np.array(self.inputs.pop())
        # dfdx
        dfdx = np.multiply(self.weights, gradient)
        # dfdw
        dfdw = np.multiply(x, gradient)
        # dfdb
        dfdb = np.multiply(1, gradient)
        self.gradients.append({"dfdw": dfdw, "dfdb": dfdb, "dfdx": dfdx})
        return dfdx

    def update(self):
        """Update weights and bias of the network stocastically."""
        self.updater(parm_names=["w", "b"], it=1)

    def updates(self):
        """Update weights and bias as one batch all together."""
        self.updater(parm_names=["w", "b"])

    @property
    def cost(self):
        """Get no cost of a this node."""
        return 2
