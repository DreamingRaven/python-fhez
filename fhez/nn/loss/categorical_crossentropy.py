# @Author: George Onoufriou <archer>
# @Date:   2021-08-02T22:04:55+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-06T11:01:36+01:00
from fhez.nn.loss.loss import Loss
import numpy as np


class CategoricalCrossentropy(Loss):
    """*Categorical* cross entropy for **multi-class** classification.

    This is also known as **softmax loss**, since it is mostly used with
    softmax activation function.

    **Not to be confused with binary cross-entropy/ log loss**, which is
    instead
    for multi-label classification, and is instead used with the sigmoid
    activation function.
    """

    def forward(self, y: np.ndarray, y_hat: np.ndarray):
        """Calculate cross entropy and save its state for backprop."""
        self.inputs.append({"y": y, "y_hat": y_hat})
        return self.loss(y=y, y_hat=y_hat)

    def loss(self, y: np.ndarray, y_hat: np.ndarray):
        r"""Calculate the categorical cross entryopy statelessley.

        .. math::

            -\sum_{i=0}^{C-1} y_i * \log_e(\hat{y_i})

        where:

        .. math::

            \sum_{i=0}^{C-1} \hat{p(y_i)} = 1
            \sum_{i=0}^{C-1} p(y_i) = 1
        """
        assert np.sum(y) == 1.0, "sum of y should equal exactly 1"
        assert np.sum(y_hat) == 1.0, "sum of y_hat should equal exactly 1"
        # CLIP values so we never get log(0) = infinity!
        # also clipping the maximum to reduce bias!
        # e.g clip([0, 1, 0]) = [1e-07, 0.9999999, 1e-07]
        y_hat_clipped = np.clip(y_hat, 1e-15, 1-1e-15)
        return -np.sum(y * np.log(y_hat_clipped))

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
