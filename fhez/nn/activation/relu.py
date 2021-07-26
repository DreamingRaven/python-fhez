#!/usr/bin/env python3

# @Author: GeorgeRaven <archer>
# @Date:   2021-02-22T11:46:18+00:00
# @Last modified by:   archer
# @Last modified time: 2021-07-26T16:56:27+01:00
# @License: please see LICENSE file in project root

import numpy as np
import logging as logger
import unittest
from fhez.nn.graph.node import Node
import numbers


class RELU(Node):
    """Rectified Liniar Unit (ReLU) approximation computational graph node."""

    def __init__(self, q=None):
        """Create a RELU approximation object."""
        self.q = q  # this is the approximation range of this ReLU approximator

    @property
    def q(self):
        """Get the current ReLU approximation range."""
        if self.__dict__.get("_q") is None:
            self._q = 1
        return self._q

    @q.setter
    def q(self, q):
        """Set the current ReLU approximation range."""
        self._q = q

    @property
    def cost(self):
        """Get the computational cost of traversing to this RELU node."""
        # \frac{4}{3 \pi q}x^2 + \frac{1}{2}x + \frac{q}{3 \pi}
        return 4

    def forward(self, x):
        """Calculate forward pass for singular example."""
        # storing inputs (ignored if caching is disabled)
        self.inputs.append(x)
        if isinstance(x, numbers.Number):
            N = 1
        else:
            # if not a number try to find the total number of elements
            # in this case backward should be just on a single input
            # so no batches to account for
            N = x.size
        # https://www.researchgate.net/publication/345756894_On_Polynomial_Approximations_for_Privacy-Preserving_and_Verifiable_ReLU_Networks
        # \frac{4}{3 \pi q}x^2 + \frac{1}{2}x + \frac{q}{3 \pi}
        # we have divided the constant bias function so when broadcast it does
        # not explode since it could be given a "commuted sum" array
        # t = (4/(3*np.pi*self.q)) * (x*x)
        # bias_function = ((0.5*x)+((self.q/(3*np.pi))/(x.size/len(x))))
        # t = t + bias_function
        return ((4/(3*np.pi*self.q)) * (x*x)) + \
            (((0.5*x)+((self.q/(3*np.pi))/(N))))

    def backward(self, gradient):
        """Calculate backward pass for singular example."""
        # make sure x is decrypted into a numpy array (implicitly), and summed
        # in case it is a commuted sum, but this wont make a difference if not
        x = np.array(self.inputs.pop()).sum()  # TODO not always summed

        if isinstance(x, numbers.Number):
            N = 1
        else:
            # if not a number try to find the total number of elements
            # in this case backward should be just on a single input
            # so no batches to account for
            N = x.size

        # df/dx
        dfdx = (8/(3*np.pi*self.q)) * x + (0.5/N)
        # df/dq
        dfdq = ((4/(3*np.pi))*(x*x)) + ((1/(3*np.pi))/N)
        # this function was called using a FILO popped queue
        # so we maintain the order of inputs by flipping again using a FILO que
        # again
        # x = [1, 2, 3, 4, 5] # iterate in forward order -> (matters)
        # df = [1, 2, 3, 4, 5] # working backwards for "backward" <- (matters)
        # update = [5, 4, 3, 2, 1] # update in forward order <- (arbitrary)
        self.gradients.append({"dfdq": dfdq, "dfdx": dfdx})
        return dfdx

    def update(self):
        """Update node state/ weights for a single example."""
        dfd_ = self.gradients.pop()
        print(dfd_)

    def updates(self):
        """Update node state/ weights for multiple examples simultaneously."""
        for _ in range(len(self.gradients)):
            dfd_ = self.gradients.pop()
            print(dfd_)


if __name__ == "__main__":
    logger.basicConfig(  # filename="{}.log".format(__file__),
        level=logger.INFO,
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S")
    # run all the unit-tests
    print("now testing:", __file__, "...")
    unittest.main()
