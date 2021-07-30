"""Neural Network Loss Functions."""

# @Author: George Onoufriou <archer>
# @Date:   2021-07-28T21:37:24+01:00
# @Last modified by:   archer
# @Last modified time: 2021-07-30T12:10:15+01:00

from collections import deque
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


def rmse(y: np.array, y_hat: np.array):
    r"""Calculate the Mean of the Squared Error (MSE).

    :math:`\text{RMSE}=\sqrt{\frac{\sum_{i=0}^{N-1} (y-\hat{y})^2 }{N}}`
    """
    return np.sqrt(np.mean(np.square(y - y_hat)))


class Loss():
    """Abstract loss class to unify loss function format."""

    @property
    def cache(self):
        """Get caching dictionary of auxilary data."""
        # initialise empty cache if it does not exist already
        if self.__dict__.get("_cache") is None:
            self._cache = {}
        return self._cache

    @cache.setter
    def cache(self, cache):
        self._cache = cache

    @property
    def is_cache_enabled(self):
        """Get status of whether or not caching is enabled."""
        if self.__dict__.get("_is_cache_enabled") is None:
            # cache enabled by default
            self._is_cache_enabled = True
        return self._is_cache_enabled

    @is_cache_enabled.setter
    def is_cache_enabled(self, state: bool):
        """Set the state of the cache."""
        self._is_cache_enabled = state

    def enable_cache(self):
        """Enable caching."""
        self.is_cache_enabled = True

    def disable_cache(self):
        """Disable caching."""
        self.is_cache_enabled = False

    @property
    def inputs(self):
        """Get cached input stack.

        Neural networks backpropogation requires cached inputs to calculate
        the gradient with respect to x and the weights. This is a utility
        method that initialises a stack and allows you to easily append
        or pop off of it so that the computation can occur in FILO.
        """
        if self.cache.get("_inputs") is None:
            self.cache["_inputs"] = deque()
        if self.is_cache_enabled:
            # if cache enabled return real stack
            return self.cache["_inputs"]
        # if cache disabled return dud que
        return deque()
