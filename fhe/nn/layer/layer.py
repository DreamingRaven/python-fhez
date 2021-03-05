#!/usr/bin/env python3

# @Author: GeorgeRaven <archer>
# @Date:   2020-09-16T11:33:51+01:00
# @Last modified by:   archer
# @Last modified time: 2021-03-05T14:35:36+00:00
# @License: please see LICENSE file in project root

from tqdm import tqdm
import numpy as np
from fhe.nn.activation.sigmoid import Sigmoid_Approximation
import logging as logger


class Layer():

    def __init__(self, weights, bias, stride=None, activation=None):
        self.weights = weights
        self.bias = bias
        if activation:
            self.activation_function = activation
        if stride:
            self.stride = stride

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        # initialise weights from tuple dimensions
        # TODO: properly implement xavier weight initialisation over np.rand
        if isinstance(weights, tuple):
            # https://www.coursera.org/specializations/deep-learning
            # https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78
            self._weights = np.random.rand(*weights)
        else:
            self._weights = weights

    @property
    def bias(self):
        if self.__dict__.get("_bias") is not None:
            return self._bias
        else:
            self.bias = 0
            return self.bias

    @bias.setter
    def bias(self, bias):
        self._bias = bias

    @property
    def stride(self):
        if self.__dict__.get("_stride") is not None:
            return self._stride
        else:
            self.stride = 1
            return self.stride

    @stride.setter
    def stride(self, stride):
        self._stride = stride

    @property
    def activation_function(self):
        if self.__dict__.get("_activation_function") is not None:
            return self._activation_function
        else:
            self.activation_function = Sigmoid_Approximation()
            return self.activation_function

    @activation_function.setter
    def activation_function(self, activation_function):
        self._activation_function = activation_function

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
        if self.cache.get("gradient") is None:
            self.cache["gradient"] = {}
        return self.cache["gradient"]

    @gradient.setter
    def gradient(self, gradient):
        self.cache["gradient"] = gradient

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
            accumulator = []
            for i in tqdm(range(len(self.x)), desc="{}.{}".format(
                    self.__class__.__name__, func.__name__),
                    position=0, leave=False, ncols=80, colour="blue"
            ):
                accumulator.append(func(self, gradient))
            return np.array(accumulator)
        return inner
