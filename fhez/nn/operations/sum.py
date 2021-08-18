"""Maths sum as computational node."""
# @Author: George Onoufriou <archer>
# @Date:   2021-08-17T09:53:22+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-17T11:09:47+01:00

import numpy as np
from fhez.nn.graph.node import Node


class Sum(Node):
    """Sum inputs and distribute backprop.

    First dim of inputs shape e.g (64,32,32,3) to sum are summed along axis=0
    resulting in an output of (32,32,3) and returning gradients of (64,)
    """

    @property
    def cost(self):
        """Get computational cost of this node."""
        return 0

    def forward(self, x: np.ndarray):
        """Sum inputs together assuming first dim is inputs."""
        self.inputs.append(len(x))
        return np.sum(x, axis=0)

    def backward(self, gradient: np.ndarray):
        """Distribute gradient to inputs."""
        length = self.inputs.pop()
        grad = np.array(gradient)
        distributed = np.broadcast_to(grad, (length,) + grad.shape)
        return distributed

    def update(self):
        """Do nothing since sum is not parameterisable."""
        return NotImplemented

    def updates(self):
        """Do nothing since sum is not parameterisable."""
        return NotImplemented
