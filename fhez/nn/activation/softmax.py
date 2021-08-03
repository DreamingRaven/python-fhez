# @Author: George Onoufriou <archer>
# @Date:   2021-08-02T22:00:06+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-03T14:52:04+01:00

import numpy as np
from fhez.nn.graph.node import Node
from fhez.nn.optimiser.adam import Adam


class Softmax(Node):
    """."""

    def forward(self, x: np.ndarray):
        r"""Calculate the soft maximum of some input :math:`x`.

        :math:`\sigma(x) = \frac{e^{x_i}}{\sum_{i=0}^{C-1}y_i\log\hat{y}}`

        where: :math:`C` is the number of classes
        """
        return np.exp(x)/np.sum(np.exp(x))

    def backward(self, gradient: np.ndarray):
        """Calculate backward pass for singular example."""
        raise NotImplementedError
        # return None

    @property
    def cost(self):
        """Get computational cost of this activation."""
        return 4

    def update(self):
        """Update parameters, so nothing for softmax."""
        return NotImplemented

    def updates(self):
        """Update parameters using average of gradients so none for softmax."""
        return NotImplemented
