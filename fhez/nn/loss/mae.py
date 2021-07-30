# @Author: George Onoufriou <archer>
# @Date:   2021-07-30T11:52:31+01:00
# @Last modified by:   archer
# @Last modified time: 2021-07-30T11:53:12+01:00
import numpy as np
from fhez.nn.loss.loss import Loss


class MAE(Loss):
    """Loss function to node wrapper."""

    def forward(self, y: np.ndarray, y_hat: np.ndarray):
        """Calculate the loss of the output given the ground truth."""
        self.inputs.append({"y": y, "y_hat": y_hat})
        return np.mean(np.absolute(y - y_hat))

    def backward(self, gradient):
        r"""Calculate MAE gradient with respect to :math:`\hat{y}`."""
        inp = self.inputs.pop()
        y = inp["y"]
        y_hat = inp["y_hat"]
        # if y_hat = y then we want the grad to be 0 as its exactly right
        local_grad = 0
        if y_hat > y:
            local_grad = 1
        elif y_hat < y:
            local_grad = -1
        return local_grad * gradient

    def update(self):
        """Do nothing as there are no parameters to update."""
        return NotImplemented

    def updates(self):
        """Do nothing as there are no parameters to update."""
        return NotImplemented

    def cost(self):
        """Get computational cost of the forward function."""
        return 2
