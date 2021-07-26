# @Author: George Onoufriou <archer>
# @Date:   2021-07-26T16:53:04+01:00
# @Last modified by:   archer
# @Last modified time: 2021-07-26T19:52:49+01:00

import numpy as np
from fhez.nn.graph.node import Node


class Linear(Node):
    """Linear activation function computational graph abstraction."""

    def forward(self, x):
        """Get linear forward propogation."""
        # cache input for later re-use
        self.inputs.append(x)
        # return computed forward propogation of node
        return x

    def backward(self, gradient):
        """Get gradients of backward prop."""
        # get any cached values required
        x = np.array(self.inputs.pop())
        # calculate gradients respect to inputs and other parameters
        dfdx = 1 * gradient
        # assign gradients to dictionary for later retrieval and use
        self.gradients.append({"dfdx": dfdx})
        # return the gradient with respect to input for immediate use
        return dfdx

    def update(self):
        """Update any weights and biases for a single example."""
        return NotImplemented

    def updates(self):
        """Update any weights and biases based on an avg of all examples."""
        return NotImplemented

    @property
    def cost(self):
        """Get the computational cost of this Node."""
        return 0
