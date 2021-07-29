# @Author: George Onoufriou <archer>
# @Date:   2021-07-26T16:53:04+01:00
# @Last modified by:   archer
# @Last modified time: 2021-07-29T13:24:33+01:00

import numpy as np
from fhez.nn.graph.node import Node


class Linear(Node):
    """Linear activation function computational graph abstraction."""

    def __init__(self, m=1, c=0):
        """Initialise weighted and biased linear function."""
        self.m = m
        self.c = c

    @property
    def m(self):
        """Slope."""
        if self.__dict__.get("_m") is None:
            self._m = 1  # defaults to identity y=mx+c where c=0 m=1 so y=x
        return self._m

    @m.setter
    def m(self, m):
        self._m = m

    @property
    def c(self):
        """Intercept."""
        if self.__dict__.get("_c") is None:
            self._c = 0  # defaults to identity y=mx+c where c=0 m=1 so y=x
        return self._c

    @c.setter
    def c(self, c):
        self._c = c

    def forward(self, x):
        """Get linear forward propogation."""
        # cache input for later re-use
        self.inputs.append(x)
        # return computed forward propogation of node
        return self.m * x + self.c

    def backward(self, gradient):
        """Get gradients of backward prop."""
        # get any cached values required
        x = np.array(self.inputs.pop())
        # calculate gradients respect to inputs and other parameters
        dfdx = self.m * gradient
        dfdm = x * gradient
        dfdc = 1 * gradient
        # assign gradients to dictionary for later retrieval and use
        self.gradients.append({"dfdx": dfdx,
                               "dfdm": dfdm,
                               "dfdc": dfdc})
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
