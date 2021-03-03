# @Author: GeorgeRaven <archer>
# @Date:   2021-02-22T11:46:18+00:00
# @Last modified by:   archer
# @Last modified time: 2021-03-03T10:52:06+00:00
# @License: please see LICENSE file in project root
import numpy as np
from fhe.nn.activation.activation import Activation
from tqdm import tqdm


class Sigmoid_Approximation(Activation):

    @Activation.fwd
    def forward(self, x):
        # sigmoid approximation in specific order to minimise depth.
        # dividing 0.5 by size of x to prevent broadcast explosion
        # when not summed yet as commuting it to later post-decryption
        return (0.5/x.size) + (0.197 * x) + ((-0.004 * x) * (x * x))

    @Activation.bwd
    def backward(self, gradient):
        # calculate local gradient but using normal sigmoid derivative
        # as this is approximate and is faster this way
        # \frac{d\sigma}{dx} = (1-\sigma(x))\sigma(x)

        # only calculate one item at a time
        x = self.to_plaintext(self.x.pop(0))
        df_dbatch_sum = 0
        for batch in tqdm(range(len(x)), desc="{}.backward.batch".format(
                self.__class__.__name__)):
            batch = np.sum(x[batch])
            df_dbatch = (1 - self.sigmoid(batch)) * self.sigmoid(batch) * \
                gradient
            df_dbatch_sum += df_dbatch
        # average out between batches to get more stable gradient
        df_dx = df_dbatch_sum / len(x)
        return df_dx

    def update(self):
        # new_parameter = old_parameter - learning_rate * gradient_of_parameter
        raise NotImplementedError(
            "Sigmoid approximation has no parameters to update")

    def sigmoid(self, x):
        return (1/(1+np.exp(-x)))
