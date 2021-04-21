#!/usr/bin/env python3

# @Author: GeorgeRaven <archer>
# @Date:   2021-04-15T14:24:29+01:00
# @Last modified by:   archer
# @Last modified time: 2021-04-20T00:38:31+01:00
# @License: please see LICENSE file in project root

import unittest
import logging as logger

# graphing libs
from networkx import nx
import igraph


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

    @property
    def g(self):
        """Get computational graph describing neural network."""
        return self.__dict__.get("_graph")

    @g.setter
    def g(self, graph):
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

    def basic_networkx(self):
        """Init graph."""
        from fhe.nn.layer.cnn import Layer_CNN
        from fhe.nn.layer.ann import Layer_ANN
        # init basic graph
        graph = nx.DiGraph()
        net = Net()
        # populate basic graph
        graph.add_node("cnn-0",
                       type="neuron",
                       nn=Layer_CNN(weights=(1, 3, 3, 3),
                                    stride=[1, 3, 3, 3],
                                    bias=0))
        graph.add_node("ann-0",
                       type="neuron",
                       nn=Layer_ANN(weights=(5,), bias=0))
        graph.add_edge("cnn-0", "ann-0")
        # return basic graph in net
        net.graph = graph
        return net

    def basic_igraph(self):
        return None

    def test_basic_networkx(self):
        """Test running graph."""
        net = self.basic_igraph()
        print(net)

    def test_basic_igraph(self):
        """Test running graph."""
        net = self.basic_networkx()
        print(net)


if __name__ == "__main__":
    logger.basicConfig(  # filename="{}.log".format(__file__),
        level=logger.INFO,
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S")
    # run all the unit-tests
    print("now testing:", __file__, "...")
    unittest.main()
