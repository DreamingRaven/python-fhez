# @Author: George Onoufriou <archer>
# @Date:   2021-08-02T22:04:55+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-05T12:37:47+01:00
from fhez.nn.loss.loss import Loss
import numpy as np


class CategoricalCrossentropy(Loss):
    """Categorical Cross Entropy for Multi-Class Multi-Label problems."""

    def forward(self, y: np.ndarray, y_hat: np.ndarray):
        """Calculate cross entropy and save its state for backprop."""
        self.inputs.append({"y": y, "y_hat": y_hat})
        return self.loss(y=y, y_hat=y_hat)

    def loss(self, y: np.ndarray, y_hat: np.ndarray):
        r"""Calculate the categorical cross entryopy statelessley.

        :math:`-\sum_{c=0}^{C-1} y_c * \log_e(\hat{y_c})`
        """
        return -1 * np.sum(y * np.log(y_hat))

    def backward(self, gradient: np.ndarray):
        r"""Calculate gradient of loss with respect to :math:`\hat{y}`."""
        x = self.inputs.pop()  # get original potentially encrypted values
        for key, value in x.items():
            # for each value in dictionary ensure it is a numpy array
            # which also means decrypting if possible
            x[key] = np.array(value)

        # use these values to recalculate the loss as we will need this
        loss = self.loss(**x)

        # get the maximum gradient index so we know which gradients go where
        # as one gradient is different to all the others if it is the correct
        # output
        i = np.argmax(gradient)
        print("x: {}, loss: {}, i: {}".format(x, loss, i))
        raise NotImplementedError("This function is incomplete.")
