#!/usr/bin/env python3

# @Author: George Onoufriou <archer>
# @Date:   2021-07-11T14:35:36+01:00
# @Last modified by:   archer
# @Last modified time: 2021-07-12T21:12:02+01:00

import os
import time
import unittest
import logging as logger

import abc
import numpy as np
from collections import deque


# graphing libs
from networkx import nx
# import igraph


class ComputationalNode(abc.ABC):
    """Abstract class for neural network nodes for traversal/ computation."""

    # # # Caching
    # This section deals with caching/ enabling/ disabling caching
    # it is the responsibility of subclassers to respect this flag but we help
    # with some properties such as "inputs" being cache aware
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
        return self._cache

    @cache.setter
    def cache(self, cache):
        self._cache = cache

    # # # Utility Methods
    # these methods help implementers respect flags such as enabled cache,
    # while also alleviating some of the repeate code needing to be implemented

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

    @property
    def gradients(self):
        """Get cached input stack.

        For neural networks to calculate any given weight update, it needs to
        remember atleast the last gradient in the case of stocastic descent,
        or multiple gradients if implementing batch normalised gradient
        descent. This is a helper method that initialises a stack so that
        implementation can be offloaded and made-uniform between all subclasses
        """
        if self.cache.get("_gradients") is None:
            self.cache["_gradients"] = deque()
        if self.is_cache_enabled:
            # if cache enabled return real stack
            return self.cache["_gradients"]
        # if cache disabled return dud que
        return deque()

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

    def forwards(self, xs):
        """Calculate forward pass for multiple examples simultaneously."""
        accumulator = []
        for i in xs:
            accumulator.append(self.forward(x=i))
        return accumulator

    def backwards(self, gradients):
        """Calculate backward pass for multiple examples simultaneously."""
        accumulator = []
        for i in gradients:
            accumulator.append(self.backward(gradient=i))
        return accumulator

    @abc.abstractmethod
    def update(self):
        """Update node state/ weights for a single example."""

    @abc.abstractmethod
    def updates(self):
        """Update node state/ weights for multiple examples simultaneously."""


class RELU(ComputationalNode):
    """Rectified Liniar Unit (ReLU) computational graph node."""

    def __init__(self, q=None):
        """Create a RELU approximation object."""
        self.q = q  # this is the approximation range of this ReLU approximator

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
        # storing inputs (ignored if caching is disabled)
        self.inputs.append(x)
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
        # make sure x is decrypted into a numpy array (implicitly), and summed
        # in case it is a commuted sum, but this wont make a difference if not
        x = np.array(self.inputs.pop()).sum()
        # df/dx
        dfdx = (8/(3*np.pi*self.q)) * x + 0.5
        # df/dq
        dfdq = ((4/(3*np.pi))*(x*x)) + (1/(3*np.pi))
        # this function was called using a FILO popped queue
        # so we maintain the order of inputs by flipping again using a FILO que
        # again
        # x = [1, 2, 3, 4, 5] # iterate in formward order -> (matters)
        # df = [1, 2, 3, 4, 5] # working backwards for "backward" <- (matters)
        # update = [5, 4, 3, 2, 1] # update in forward order <- (arbitrary)
        self.gradients.append({"dfdq": dfdq, "dfdx": dfdx})
        return dfdx

    def update(self):
        """Update node state/ weights for a single example."""
        dfd_ = self.gradients.pop()
        print(dfd_)

    def updates(self):
        """Update node state/ weights for multiple examples simultaneously."""
        for _ in len(self.gradients):
            dfd_ = self.gradients.pop()
            print(dfd_)


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
        graph = nx.MultiDiGraph()
        graph.add_node("input")
        graph.add_node("ReLU", layer=RELU())
        graph.add_node("output")
        graph.add_edge("input", "ReLU")
        graph.add_edge("ReLU", "output")
        self.nn = NN(graph=graph)

    def tearDown(self):
        pass

    def test_init(self):
        pass


if __name__ == "__main__":
    logger.basicConfig(  # filename="{}.log".format(__file__),
        level=logger.INFO,
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S")
    # run all the unit-tests
    print("now testing:", __file__, "...")
    unittest.main()
