"""Neural network selector for bridging training and inference circuits."""
# @Author: George Onoufriou <archer>
# @Date:   2021-09-21T12:30:09+01:00
# @Last modified by:   archer
# @Last modified time: 2021-09-21T13:10:40+01:00

import itertools
import numpy as np
from fhez.nn.graph.node import Node


class Selector(Node):
    """Abstract selector class for selecting inputs to outputs per receptor."""

    def __init__(self, forward=None, backward=None):
        """Given lists of ones and zeroes select inputs to outputs per func."""
        self._forward = forward
        self._backward = backward

    @property
    def forward_selection(self):
        """Get a zeroes and ones list of inputs to pass forward."""
        return self.__dict__.get("_forward")

    @property
    def backward_selection(self):
        """Get a zeroes and ones list of gradients to pass backward."""
        return self.__dict__.get("_backward")

    def forward(self, x):
        """."""
        if self.forward_selection is None:
            return x
        else:
            t = list(itertools.compress(x, self.forward_selection))
            if len(t) == 1:
                return t[0]
            else:
                return t

    def backward(self, gradient):
        """."""
        if self.backward_selection is None:
            return gradient
        else:
            t = list(itertools.compress(gradient, self.backward_selection))
            if len(t) == 1:
                return t[0]
            else:
                return t

    @property
    def cost(self):
        """Get computational cost of this node."""
        return 0

    def update(self):
        """Do nothing since selector is not parameterisable."""
        return NotImplemented

    def updates(self):
        """Do nothing since selector is not parameterisable."""
        return NotImplemented
