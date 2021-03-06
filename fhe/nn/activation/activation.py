# @Author: GeorgeRaven <archer>
# @Date:   2021-02-22T11:46:18+00:00
# @Last modified by:   archer
# @Last modified time: 2021-03-06T13:10:34+00:00
# @License: please see LICENSE file in project root
import numpy as np
from fhe.rearray import ReArray
from fhe.reseal import ReSeal
import logging as logger

from tqdm import tqdm


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

    def probe_shape(self, lst: list):
        """Get the shape of a list, assuming each sublist is the same length.

        This function is recursive, sending the sublists down and terminating
        once a type error is thrown by the final point being a non-list
        """
        if isinstance(lst, list):
            # try appending current length with recurse of sublist
            try:
                return (len(lst),) + self.probe_shape(lst[0])
            # once we bottom out and get some non-list type abort and pull up
            except (AttributeError, IndexError):
                return (len(lst),)
        else:
            return lst.shape

    def fwd(func):
        """Forward decorator, unpacking + stashing x to use in backward."""

        def inner(self, x):
            self.x.append(x)
            logger.debug("{}.{} x.shape={}".format(
                self.__class__.__name__,
                func.__name__,
                self.probe_shape(x)))
            temp = func(self, x)
            logger.debug("{}.{} return.shape={}".format(
                self.__class__.__name__,
                func.__name__,
                self.probe_shape(temp)))
            return temp
        return inner

    def bwd(func):
        """Backward decorator to use decrypted or decrypt stashed x."""

        def inner(self, gradient=1):
            if isinstance(gradient, int):
                gradient = np.array([gradient])
            logger.debug("{}.{} gradient.shape={}".format(
                self.__class__.__name__,
                func.__name__,
                gradient.shape))
            accumulator = []
            for i in tqdm(range(len(self.x)), desc="{}.{}".format(
                    self.__class__.__name__, func.__name__),
                    position=0, leave=False, ncols=80, colour="blue"
            ):
                # broadcast shape of gradient to match number of batches by
                # adding tuple of num batches to gradient tuple of shape
                gradient_broadcast = np.broadcast_to(
                    gradient,
                    # add the two tuples of batch size + gradient shape
                    (len(self.x[0]),) + gradient.shape)
                # start decrypting and popping x to reduce consume cache/ size
                x = self.to_plaintext(self.x.pop(0))
                accumulator.append(func(self, gradient_broadcast, x))
            df_dx = np.array(accumulator)
            logger.debug("{}.{} gradient.shape={}".format(
                self.__class__.__name__,
                func.__name__,
                df_dx.shape))
            return df_dx
        return inner
