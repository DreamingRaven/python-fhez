#!/usr/bin/env python3

# @Author: GeorgeRaven <archer>
# @Date:   2021-04-15T14:24:29+01:00
# @Last modified by:   archer
# @Last modified time: 2021-04-19T22:14:25+01:00
# @License: please see LICENSE file in project root

import unittest
from fhe.nn.graph.node import Node


class Net(object):
    """Graph representing neural network computations as a network of nodes."""

    def __init__(self):
        """Initialise a new network/ graph."""
        pass

    @property
    def graph(self):
        """Get computational graph describing neural network."""
        return self.__dict__.get("_graph")

    @graph.setter
    def graph(self, graph):
        self._graph = graph

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

    def setUp(self):
        """Init graph."""
        import networkx as nx
        graph = nx.Graph()
        self.net = Net()
        self.net.graph = graph

    def test_graph_run(self):
        """Test running graph."""
        pass

    def tearDown(self):
        """Delete graph."""
        pass
