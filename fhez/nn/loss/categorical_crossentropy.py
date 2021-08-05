# @Author: George Onoufriou <archer>
# @Date:   2021-08-02T22:04:55+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-05T23:11:59+01:00
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
        assert np.sum(y) == 1.0, "sum of y should equal exactly 1"
        assert np.sum(y_hat) == 1.0, "sum of y_hat should equal exactly 1"
        return -np.sum(y * np.log(y_hat))

    def backward(self, gradient: np.ndarray):
        r"""Calculate gradient of loss with respect to :math:`\hat{y}`."""
        inp = self.inputs.pop()  # get original potentially encrypted values
        for key, value in inp.items():
            # for each value in dictionary ensure it is a numpy array
            # which also means decrypting if possible
            inp[key] = np.array(value)

        dfdpy = -1 / (inp["y_hat"])  # calculate local gradient
        dfdpy = dfdpy * inp["y"]  # multiply each by actual probability
        return dfdpy * gradient
