"""Neural Network Loss Functions."""

# @Author: George Onoufriou <archer>
# @Date:   2021-07-28T21:37:24+01:00
# @Last modified by:   archer
# @Last modified time: 2021-07-28T21:45:37+01:00

import numpy as np


def mae(y: np.array, y_hat: np.array):
    """Calculate Mean Absolute Error (MAE)."""
    return np.mean(np.absolute(y - y_hat))
