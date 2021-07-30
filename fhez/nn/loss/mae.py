# @Author: George Onoufriou <archer>
# @Date:   2021-07-30T11:52:31+01:00
# @Last modified by:   archer
# @Last modified time: 2021-07-30T17:31:15+01:00
import numpy as np
from fhez.nn.loss.loss import Loss


class MAE(Loss):
    """Loss function to node wrapper."""

    def forward(self, y: np.ndarray, y_hat: np.ndarray):
        r"""Calculate the loss of the output given the ground truth.

        This will take multiple values for both :math:`y` and :math:`\hat{y}`,
        and return a
        single value that is the mean of their absolute difference.

        :math:`\text{MAE}=\frac{\sum_{i=0}^{N-1} \left\|y-\hat{y}\right\| }{N}`
        """
        self.inputs.append({"y": y, "y_hat": y_hat})
        return np.mean(np.absolute(y - y_hat))

    def backward(self, gradient):
        r"""Calculate MAE gradient with respect to :math:`\hat{y}`.

        This will take a single gradient value, and return the average gradient
        with respect to :math:`\hat{y}`

        :math:`\dfrac{d}{d\hat{y}}(\text{MAE}) = \begin{cases} +1,\quad \hat{y}>y\\ \ \ \ 0,\quad \hat{y}=y\\-1,\quad \hat{y}<y \end{cases}`
        """
        inp = self.inputs.pop()
        y = inp["y"]
        y_hat = inp["y_hat"]
        # create an array by element wise comparison against the two inputs
        # if y==y_hat, then grad=0
        # if y_hat>y, then grad=1
        # if y_hat<y, then grad=-1
        local_grads = (1 * (y_hat > y)) + (-1 * (y_hat < y))
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

    def cost(self):
        """Get computational cost of the forward function."""
        return 2
