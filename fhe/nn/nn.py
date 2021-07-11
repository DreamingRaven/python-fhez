#!/usr/bin/env python3

# @Author: George Onoufriou <archer>
# @Date:   2021-07-11T14:35:36+01:00
# @Last modified by:   archer
# @Last modified time: 2021-07-11T16:01:02+01:00

import os
import time
import unittest
import logging as logger

# graphing libs
from networkx import nx
# import igraph

import abc


class Node(abc.ABC):
    """Abstract class for neural network nodes for traversal/ computation."""

    @property
    def cache(self):
        if self.__dict__.get("_cache") is None:
            self._cache = {}
        return self._cache

    @cache.setter
    def cache(self, cache):
        self._cache = cache

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


class RELU(Node):
    """Rectified Liniar Unit (ReLU) computational graph node."""

    def forward(self, x):
        """Calculate forward pass for singular example."""
        t = 4/3
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
