"""Neural Network Loss Functions."""

# @Author: George Onoufriou <archer>
# @Date:   2021-07-28T21:37:24+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-03T11:15:14+01:00

import abc
from collections import deque
import numpy as np


class Loss(abc.ABC):
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

    # ABSTRACT METHODS

    @abc.abstractmethod
    def forward(self, y: np.ndarray, y_hat: np.ndarray):
        """Calculate loss(es) given one or more truths."""

    @abc.abstractmethod
    def backward(self, gradient: np.ndarray):
        r"""Calculate gradient of loss with respect to :math:`\hat{y}`."""
