"""Softmax activation node abstraction."""
# @Author: George Onoufriou <archer>
# @Date:   2021-08-02T22:00:06+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-09T15:28:06+01:00

import numpy as np
from fhez.nn.graph.node import Node
from fhez.nn.optimiser.adam import Adam


class Softmax(Node):
    """Softmax activation, normalising sum of inputs to 1, as probability."""

    def forward(self, x: np.ndarray):
        r"""Calculate the soft maximum of some input :math:`x`.

        :math:`\hat{p(y_i)} = \frac{e^{a_i}}{\sum_{j=0}^{C-1}e^{a_j}}`

        where: :math:`C` is the number of classes, and :math:`i` is the current
        class being processed.
        """
        # self.inputs.append(x)
        out = np.exp(x)/np.sum(np.exp(x))
        self.inputs.append(out)  # NOTE appending x_softmaxed not x
        return out

    def backward(self, gradient: np.ndarray):
        r"""Calculate the soft maximum derivative with respect to each input.

        .. math::

            \frac{d\textit{SMAX(a)}}{da_i} = \begin{cases} \hat{p(y_i)}
            (1 - \hat{p(y_i)}), & \text{if}\ c=i \\ -\hat{p(y_c)} *
            \hat{p(y_i)}, & \text{otherwise} \end{cases}

        where: :math:`c` is the one hot encoded index of the correct/ true
        classification, and :math:`i` is the current index for the current
        classification.
        """
        # softmax derivative does not need x it needs the x_softmaxed
        x = np.array(self.inputs.pop())  # note this is x_softmaxed
        # calculate class specific gradient
        dfdx = (x * (1-x)) * gradient
        # calculate inter class gradient
        for i in range(len(x)):
            t = -x[i] * x * gradient[i]
            t[i] = 0  # already calc class specific grad differently above
            dfdx += t
        return dfdx

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
