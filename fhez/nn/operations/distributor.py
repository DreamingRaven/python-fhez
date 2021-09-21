# @Author: George Onoufriou <archer>
# @Date:   2021-09-21T14:33:37+01:00
# @Last modified by:   archer
# @Last modified time: 2021-09-21T14:46:38+01:00

import numpy as np
from fhez.nn.graph.node import Node


class Distributor(Node):
    """."""

    @property
    def cost(self):
        """."""
        return 0

    def forward(self, x):
        """."""
        return x

    def backward(self, gradient):
        """."""
        return np.sum(gradient, axis=0)

    def update(self):
        """Update nothing as accumulation is not parameterisable."""
        return NotImplemented

    def updates(self):
        """Update nothing as accumulation is not parameterisable."""
        return NotImplemented
