#!/usr/bin/env python3

# @Author: GeorgeRaven <archer>
# @Date:   2021-04-15T14:24:29+01:00
# @Last modified by:   archer
# @Last modified time: 2021-04-19T21:44:08+01:00
# @License: please see LICENSE file in project root

import unittest
from fhe.nn.graph.node import Node


class Net(object):
    """Graph representing neural network computations as a network of nodes."""

    def __init__(self):
        """Initialise a new network/ graph."""
        pass

    @property
    def head(self):
        """Get the single origin node of the computational graph."""
        return self.__dict__.get("_head")

    @head.setter
    def head(self, node: Node):
        self._head = node

    @property
    def tail(self):
        """Get the single final node of the computational graph."""
        return self.__dict__.get("_tail")

    @tail.setter
    def tail(self, node: Node):
        self._tail = node

    def train(self, x, y):
        """Train graphed neural network using some input data."""
        pass

    def test(self, x):
        """Test / infer/ predict based on some input data."""
        pass

    def traverse(self):
        """Traverse and yield nodes on the graph, depth first."""
        pass


class net_tests(unittest.TestCase):
    """Testing net class."""
