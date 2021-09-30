# @Author: George Onoufriou <archer>
# @Date:   2021-07-30T14:52:55+01:00
# @Last modified by:   archer
# @Last modified time: 2021-09-10T14:32:50+01:00

import numpy as np
from fhez.nn.loss.loss import Loss


class MSE(Loss):
    """Loss function to node wrapper."""

    def forward(self, signal=None,
                y: np.ndarray = None,
                y_hat: np.ndarray = None):
        r"""Calculate the loss of the output given the ground truth.

        This will take multiple values for both :math:`y` and :math:`\hat{y}`,
        and return a
        single value that is the mean of their absolute difference.

        :math:`\text{MSE}=\frac{\sum_{i=0}^{N-1} (y-\hat{y})^2 }{N}`
        """
        if signal is None:
            msg = "if no signal provided then you must provide y and y_hat"
            assert y_hat is not None, msg
            assert y is not None, msg
        else:
            # THE ORDER IS DEPENDENT ON THE ORDER OF EDGES!
            y_hat = signal[0]
            y = signal[1]
        self.inputs.append({"y": y, "y_hat": y_hat})
        return np.mean((y - y_hat)**2)

    def backward(self, gradient):
        r"""Calculate MSE gradient with respect to :math:`\hat{y}`.

        This will take a single gradient value, and return the average gradient
        with respect to :math:`\hat{y}`. If :math:`\hat{y}` is more than 1 dim
        it will return a multidimensional array of values which are the
        average gradients in those dims.

        :math:`\frac{d}{d\hat{y}}(\text{MSE})=\sum_{i=0}^{N-1} -2(y-\hat{y})`
        """
        inp = self.inputs.pop()
        y = inp["y"]
        y_hat = inp["y_hat"]
        local_grads = np.sum((-2 * (y - y_hat)), axis=0)
        # check if we need to give a multidimensional output if each y item
        # is itself another dimension
        if len(y_hat.shape) > 1:
            return np.mean(local_grads, axis=0) * gradient
        return np.mean(local_grads) * gradient

    def update(self):
        """Do nothing as there are no parameters to update."""
        return NotImplemented

    def updates(self):
        """Do nothing as there are no parameters to update."""
        return NotImplemented

    @property
    def cost(self):
        """Get computational cost of the forward function."""
        return 2
