"""Categorical Cross Entropy (CCE) as node abstraction."""
# @Author: George Onoufriou <archer>
# @Date:   2021-08-02T22:04:55+01:00
# @Last modified by:   archer
# @Last modified time: 2021-10-19T14:49:26+01:00
from fhez.nn.loss.loss import Loss
import numpy as np


class CCE(Loss):
    """*Categorical* cross entropy for **multi-class** classification.

    This is also known as **softmax loss**, since it is mostly used with
    softmax activation function.

    **Not to be confused with binary cross-entropy/ log loss**, which is
    instead
    for multi-label classification, and is instead used with the sigmoid
    activation function.

    CCE Graph: https://www.desmos.com/calculator/q2dwniwjsp
    """

    def forward(self, signal=None,
                y: np.ndarray = None,
                y_hat: np.ndarray = None,
                check=False):
        """Calculate cross entropy and save its state for backprop.

        Can either be given a network signal with both y_hat and y stacked, or
        you can explicitly define y and y_hat.
        """
        if signal is None:
            msg = "if no signal provided then you must provide y and y_hat"
            assert y_hat is not None, msg
            assert y is not None, msg
        else:
            # THE ORDER IS DEPENDENT ON THE ORDER OF EDGES!
            y_hat = signal[0]
            y = signal[1]

        if check:
            assert np.sum(y) == 1.0, "sum of y should equal exactly 1"
            assert np.sum(y_hat) == 1.0, "sum of y_hat should equal exactly 1"
        # CLIP values so we never get log(0) = infinity!
        # also clipping the maximum to reduce bias!
        # e.g clip([0, 1, 0]) = [1e-07, 0.9999999, 1e-07]
        y_hat_clipped = np.clip(y_hat, 1e-07, 1-1e-07)
        self.inputs.append({"y": y, "y_hat": y_hat_clipped})
        return self.loss(y=y, y_hat=y_hat_clipped)

    def loss(self, y: np.ndarray, y_hat: np.ndarray):
        r"""Calculate the categorical cross entryopy statelessley.

        .. math::

            CCE(\hat{p(y)}) = -\sum_{i=0}^{C-1} y_i * \log_e(\hat{y_i})

        where:

        .. math::

            \sum_{i=0}^{C-1} \hat{p(y_i)} = 1

            \sum_{i=0}^{C-1} p(y_i) = 1
        """
        return -np.sum(y * np.log(y_hat))

    def backward(self, gradient: np.ndarray):
        r"""Calculate gradient of loss with respect to :math:`\hat{y}`.

        .. math::

            \frac{d\textit{CCE}(\hat{p(y)})}{d\hat{p(y_i)}} =
            \frac{-1}{\hat{p(y_i)}}p(y_i)
        """
        inp = self.inputs.pop()  # get original potentially encrypted values
        for key, value in inp.items():
            # for each value in dictionary ensure it is a numpy array
            # which also means decrypting if possible
            inp[key] = np.array(value)

        dfdpy = -1 / (inp["y_hat"])  # calculate local gradient
        dfdpy = dfdpy * inp["y"]  # multiply each by actual probability
        return dfdpy * gradient

    @property
    def cost(self):
        """Get 0 cost of plaintext loss calculation."""
        return 0


CategoricalCrossEntropy = CCE
