"""Neural Network Loss Functions."""

# @Author: George Onoufriou <archer>
# @Date:   2021-07-28T21:37:24+01:00
# @Last modified by:   archer
# @Last modified time: 2021-07-28T23:01:11+01:00

import numpy as np


def mae(y: np.array, y_hat: np.array):
    r"""Calculate Mean Absolute Error (MAE).

    :math:`\text{MAE}=\frac{\sum_{i=0}^{N-1} \left\|y-\hat{y}\right\| }{N}`
    """
    return np.mean(np.absolute(y - y_hat))


def mse(y: np.array, y_hat: np.array):
    r"""Calculate the Mean of the Squared Error (MSE).

    :math:`\text{MSE}=\frac{\sum_{i=0}^{N-1} (y-\hat{y})^2 }{N}`
    """
    return np.mean(np.square(y - y_hat))
