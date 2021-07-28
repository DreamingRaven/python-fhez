"""Neural Network Loss Functions."""

# @Author: George Onoufriou <archer>
# @Date:   2021-07-28T21:37:24+01:00
# @Last modified by:   archer
# @Last modified time: 2021-07-28T22:38:49+01:00

import numpy as np


def mae(y: np.array, y_hat: np.array):
    """Calculate Mean Absolute Error (MAE)."""
    return np.mean(np.absolute(y - y_hat))


def mse(y: np.array, y_hat: np.array):
    """Calculate the Mean of the Squared Error (MSE)."""
    return np.mean(np.square(y - y_hat))
