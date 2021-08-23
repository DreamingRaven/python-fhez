"""Argmax activation as node abstraction."""
# @Author: George Onoufriou <archer>
# @Date:   2021-08-02T22:00:06+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-23T14:35:12+01:00

import numpy as np
from fhez.nn.graph.node import Node


class Argmax(Node):
    """Argmax activation, compute sparse array of highest activation."""

    def forward(self, x: np.ndarray):
        r"""Calculate the argmax of some input :math:`x` along its first axis.

        Argmax.
        """
        sparse = np.eye(len(x))[np.argmax(x, axis=0)]
        return sparse

    def backward(self, gradient: np.ndarray):
        r"""Calculate the argmax derivative with respect to each input.

        Argmax.
        """
        # assuming sparse gradient like [0,1,0] will mean this local gradient
        # is simply just the forward gradient
        return gradient

    @property
    def cost(self):
        """Get computational cost of this activation."""
        return 0

    def update(self):
        """Update parameters, so nothing for argmax."""
        return NotImplemented

    def updates(self):
        """Update parameters using average of gradients so none for argmax."""
        return NotImplemented
