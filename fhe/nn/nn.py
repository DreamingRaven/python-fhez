#!/usr/bin/env python3

# @Author: George Onoufriou <archer>
# @Date:   2021-07-11T14:35:36+01:00
# @Last modified by:   archer
# @Last modified time: 2021-07-11T20:29:37+01:00

import os
import time
import unittest
import logging as logger

import abc
import numpy as np


# graphing libs
from networkx import nx
# import igraph


class ComputationalNode(abc.ABC):
    """Abstract class for neural network nodes for traversal/ computation."""

    # # # Caching
    # This section deals with caching/ enabling/ disabling caching we want to
    # be able to easily prevent caching without breaking any code by silently
    # ignoring implied cache assignment. E.G:
    # self.cache["x"] = 10, will not fail but will not be assigned silently
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
    def cache(self):
        """Get caching dictionary of auxilary data."""
        # initialise empty cache if it does not exist already
        if self.__dict__.get("_cache") is None:
            self._cache = {}
        if self.is_cache_enabled:
            return self._cache
        # returning a blank dict as cache disabled and we dont want
        # anything back again, if they do modify dict it will be ephemeral
        return {}

    @cache.setter
    def cache(self, cache):
        self._cache = cache

    # # # Abstract Methods
    # These abstract methods are intended to notify node implementers of any
    # required functions since they will be extensiveley used in the
    # computational graph, and will error if un-populated from subclasses
    @abc.abstractmethod
    def forward(self, x):
        """Calculate forward pass for singular example."""

    @abc.abstractmethod
    def backward(self, gradient):
        """Calculate backward pass for singular example."""

    @abc.abstractmethod
    def forwards(self, xs):
        """Calculate forward pass for multiple examples simultaneously."""

    @abc.abstractmethod
    def backwards(self, gradients):
        """Calculate backward pass for multiple examples simultaneously."""

    @abc.abstractmethod
    def update(self):
        """Update node state/ weights for a single example."""

    @abc.abstractmethod
    def updates(self):
        """Update node state/ weights for multiple examples simultaneously."""


class RELU(ComputationalNode):
    """Rectified Liniar Unit (ReLU) computational graph node."""

    @property
    def q(self):
        """Get the current ReLU approximation range."""
        if self.__dict__.get("q") is None:
            self._q = 1
        return self._q

    @q.setter
    def q(self, q):
        """Set the current ReLU approximation range."""
        self._q = q

    def forward(self, x):
        """Calculate forward pass for singular example."""
        # https://www.researchgate.net/publication/345756894_On_Polynomial_Approximations_for_Privacy-Preserving_and_Verifiable_ReLU_Networks
        # \frac{4}{3 \pi q}x^2 + \frac{1}{2}x + \frac{q}{3 \pi}
        # we have divided the constant bias function so when broadcast it does
        # not explode since it could be given a "commuted sum" array
        t = (4/(3*np.pi*self.q)) * (x*x)
        bias_function = ((0.5*x)+((self.q/(3*np.pi))/(x.size/len(x))))
        t = t + bias_function
        return t

    def backward(self, gradient):
        """Calculate backward pass for singular example."""

    def forwards(self, xs):
        """Calculate forward pass for multiple examples simultaneously."""

    def backwards(self, gradients):
        """Calculate backward pass for multiple examples simultaneously."""

    def update(self):
        """Update node state/ weights for a single example."""

    def updates(self):
        """Update node state/ weights for multiple examples simultaneously."""


class NeuralNetwork():
    """Multi-Directed Neural Network Graph Handler.

    This class handles traversing computational graphs toward some end-state,
    while computing forward(s), backward(s), and update(s) of the respective
    components described within.
    """

    def __init__(self, graph=None):
        """Instantiate a neural network using an existing graph object."""
        self.g = graph if graph is not None else nx.MultiDiGraph()

    @property
    def g(self):
        """Get computational graph."""
        return self.__dict__.get("_graph")

    @g.setter
    def g(self, graph):
        """Set computational graph."""
        self._graph = graph

    def forward(self, x):
        pass

    def backward(self, l):
        pass

    def forwards(self, xs):
        pass

    def backwards(self, ls):
        pass


# Shorthand / Alias for Neural Network
NN = NeuralNetwork


class NNTest(unittest.TestCase):

    def setUp(self):
        self.nn = NN()
        self.relu = RELU()
        pass

    def tearDown(self):
        pass


if __name__ == "__main__":
    logger.basicConfig(  # filename="{}.log".format(__file__),
        level=logger.INFO,
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S")
    # run all the unit-tests
    print("now testing:", __file__, "...")
    unittest.main()
