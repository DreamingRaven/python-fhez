# @Author: GeorgeRaven <archer>
# @Date:   2021-02-22T11:46:18+00:00
# @Last modified by:   archer
# @Last modified time: 2021-02-26T09:52:44+00:00
# @License: please see LICENSE file in project root
import numpy as np
from fhe.rearray import ReArray
from fhe.reseal import ReSeal
from fhe.nn.activation.activation import Activation


class Sigmoid_Approximation(Activation):

    def forward(self, x):
        self.x.append(x)
        # sigmoid approximation in specific order to minimise depth.
        # dividing 0.5 by size of x to prevent broadcast explosion
        # when not summed yet as commuting it to later post-decryption
        return (0.5/x.size) + (0.197 * x) + ((-0.004 * x) * (x * x))

    def backward(self, gradient):
        # calculate local gradient but using normal sigmoid derivative
        # as this is approximate and is faster this way
        # \frac{d\sigma}{dx} = (1-\sigma(x))\sigma(x)
        n = len(self.x)
        if n > 0:
            sum_gradients = 0
            # averaging multiple inputs
            for _ in range(n):
                x = self.to_plaintext(self.x.pop(0))  # calculate gradient
                local_gradient = 0
                # averaging across batches
                for i in range(len(x)):
                    t = np.sum(x[i])
                    btch_grad = (1 - self.sigmoid(t)) \
                        * self.sigmoid(t) * gradient
                    local_gradient += btch_grad
                local_gradient = local_gradient / len(x)
                # local_gradient = (1 - self.sigmoid(x)) *
                # self.sigmoid(x) * gradient
                sum_gradients += local_gradient
            self.gradient = sum_gradients/n
        return self.gradient

    def update(self):
        # new_parameter = old_parameter - learning_rate * gradient_of_parameter
        raise NotImplementedError(
            "Sigmoid approximation has no parameters to update")

    def sigmoid(self, x):
        return (1/(1+np.exp(-x)))
