"""Sigmoid approximation as node abstraction."""
# @Author: GeorgeRaven <archer>
# @Date:   2021-02-22T11:46:18+00:00
# @Last modified by:   archer
# @Last modified time: 2021-10-12T16:16:29+01:00
# @License: please see LICENSE file in project root
import numpy as np
from fhez.nn.graph.node import Node


class Sigmoid(Node):
    """Sigmoid approximation."""

    def forward(self, x):
        """Calculate sigmoid approximation while minimising depth."""
        self.inputs.append(x)
        # return (0.5) + (0.197 * x) + ((-0.004 * x) * (x * x))
        return np.add(np.add(0.5, np.multiply(0.197, x)), np.multiply(
            np.multiply(-0.004, x),
            np.multiply(x, x)))

    def backward(self, gradient: np.array):
        """Calculate gradient of sigmoid with respect to input x."""
        x = np.array(self.inputs.pop())
        df_dx = 0.197 + (-0.012 * (x**2))
        return df_dx * gradient

    def update(self):
        """Update nothing, as sigmoid has no parameters."""
        return NotImplemented

    def updates(self):
        """Update nothing, as sigmoid has no parameters."""
        return NotImplemented

    @property
    def cost(self):
        """Get computational depth of this node."""
        return 5

    def sigmoid(self, x):
        """Calculate standard sigmoid activation."""
        return 1/(1+np.exp(-x))
