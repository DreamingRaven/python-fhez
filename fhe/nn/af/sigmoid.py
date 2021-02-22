# @Author: GeorgeRaven <archer>
# @Date:   2021-02-22T11:46:18+00:00
# @Last modified by:   archer
# @Last modified time: 2021-02-22T11:47:48+00:00
# @License: please see LICENSE file in project root
import numpy as np
from fhe.rearray import ReArray
from fhe.reseal import ReSeal


class Sigmoid_Approximation():

    def __init__(self):
        self._cache = {}

    @property
    def x_plain(self):
        """Plaintext x for backward pass"""
        return self._cache["x"]

    @x_plain.setter
    def x_plain(self, x):
        if isinstance(x, ReSeal):
            self._cache["x"] = x.plaintext
        elif isinstance(x, ReArray):
            self._cache["x"] = np.array(x)
        else:
            self._cache["x"] = x

    def forward(self, x):
        self.x_plain = x
        # sigmoid approximation in specific order to minimise depth
        # dividing 0.5 by size of x to prevent explosion when not summed
        return (0.5/x.size) + (0.197 * x) + ((-0.004 * x) * (x * x))

    def backward(self, gradient):
        # calculate local gradient but using normal sigmoid derivative
        # as this is approximate and is faster this way
        # \frac{d\sigma}{dx} = (1-\sigma(x))\sigma(x)
        x = self.x_plain  # get our cached input to calculate gradient
        local_gradient = (1 - self.sigmoid(x)) * self.sigmoid(x) * gradient
        return local_gradient

    def update(self):
        # new_parameter = old_parameter - learning_rate * gradient_of_parameter
        raise NotImplementedError(
            "Sigmoid approximation has no parameters to update")

    def sigmoid(self, x):
        return (1/(1+np.exp(-x)))
