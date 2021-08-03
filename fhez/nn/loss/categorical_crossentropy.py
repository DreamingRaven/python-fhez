# @Author: George Onoufriou <archer>
# @Date:   2021-08-02T22:04:55+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-03T14:23:04+01:00
from fhez.nn.loss.loss import Loss
import numpy as np


class CategoricalCrossentropy(Loss):
    """Categorical Cross Entropy for Multi-Class Multi-Label problems."""

    def forward(self, y: np.ndarray, y_hat: np.ndarray):
        """Calculate loss(es) given one or more truths."""

    def backward(self, gradient: np.ndarray):
        r"""Calculate gradient of loss with respect to :math:`\hat{y}`."""
