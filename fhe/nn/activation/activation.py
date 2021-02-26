# @Author: GeorgeRaven <archer>
# @Date:   2021-02-22T11:46:18+00:00
# @Last modified by:   archer
# @Last modified time: 2021-02-24T16:13:22+00:00
# @License: please see LICENSE file in project root
import numpy as np
from fhe.rearray import ReArray
from fhe.reseal import ReSeal


class Activation():

    @property
    def cache(self):
        if self.__dict__.get("_cache") is None:
            self._cache = {}
        return self._cache

    @cache.setter
    def cache(self, cache):
        self._cache = cache

    @property
    def x(self):
        """Plaintext x for backward pass"""
        if self.cache.get("x") is None:
            self.cache["x"] = []
        return self.cache["x"]

    @x.setter
    def x(self, x):
        self.cache["x"] = x

    @property
    def gradient(self):
        return self.cache.get("gradient")

    @gradient.setter
    def gradient(self, gradient):
        self.cache["gradient"] = gradient

    def to_plaintext(self, x):
        if isinstance(x, ReArray):
            return np.array(x)
        elif isinstance(x, ReSeal):
            return np.array(x.plaintext)
        else:
            return np.array(x)
