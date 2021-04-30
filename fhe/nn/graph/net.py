#!/usr/bin/env python3

# @Author: GeorgeRaven <archer>
# @Date:   2021-04-15T14:24:29+01:00
# @Last modified by:   archer
# @Last modified time: 2021-04-30T11:07:56+01:00
# @License: please see LICENSE file in project root

import os
import time
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
        assert isinstance(graph, (nx.DiGraph, igraph.Graph))
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

    def basic_plot(self, filepath: str):
        """Generate a very basic plot of the directed graph network."""
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        # set basic values
        G = self.g
        pos = nx.layout.spring_layout(G)
        node_sizes = [3 + 10 * i for i in range(len(G))]
        M = G.number_of_edges()
        edge_colors = range(2, M + 2)
        edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
        fig = plt.figure()

        # create graph based on values
        nodes = nx.draw_networkx_nodes(G,
                                       pos,
                                       node_size=node_sizes,
                                       node_color="blue")
        edges = nx.draw_networkx_edges(
            G,
            pos,
            node_size=node_sizes,
            arrowstyle="->",
            arrowsize=10,
            edge_color=edge_colors,
            edge_cmap=plt.cm.Blues,
            width=2,
        )
        # set alpha value for each edge
        for i in range(M):
            edges[i].set_alpha(edge_alphas[i])

        pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.Blues)
        pc.set_array(edge_colors)
        plt.colorbar(pc)

        ax = plt.gca()
        ax.set_axis_off()
        plt.show()
        fig.savefig(self._create_path(filepath))
        # fig.savefig(filepath)

    def _create_path(self, path: str):
        """Util functon to create a path properly for us for plotting."""
        path = os.path.abspath(path)
        try:
            dirpath = os.path.split(path)[0]
            os.mkdir(dirpath)
            logger.info("Created path: {}".format(dirpath))
        except FileExistsError:
            logger.info("path: {}, exists ignoring".format(dirpath))
        return path


class net_tests(unittest.TestCase):
    """Testing net class."""

    def setUp(self):
        """Connect to db and set up timer."""
        self.startTime = time.time()

    def tearDown(self):
        """Consume time and display."""
        t = time.time() - self.startTime
        logger.info("{}: {}".format(
            self.id(), t))

    def basic_networkx(self):
        """Init graph."""
        from fhe.nn.layer.cnn import Layer_CNN
        from fhe.nn.layer.ann import Layer_ANN
        # init basic graph
        graph = nx.DiGraph()
        net = Net()
        # populate basic graph
        graph.add_node("x")
        graph.add_node("cnn-0",
                       type="neuron",
                       nn=Layer_CNN(weights=(1, 3, 3, 3),
                                    stride=[1, 3, 3, 3],
                                    bias=0))
        graph.add_edge("x", "cnn-0")
        graph.add_node("ann-0",
                       type="neuron",
                       nn=Layer_ANN(weights=(5,), bias=0))
        graph.add_edge("cnn-0", "ann-0")
        graph.add_node("sum")
        graph.add_edge("ann-0", "sum")
        graph.add_node("loss")
        graph.add_edge("sum", "loss")
        graph.add_node("y")
        graph.add_edge("y", "loss")

        # return basic graph in net
        net.graph = graph
        return net

    def test_basic_networkx(self):
        """Test running graph."""
        net = self.basic_networkx()
        print(type(net.graph))
        print(net)

    def test_basic_networkx_plot(self):
        """Test running graph."""
        net = self.basic_networkx()
        net.basic_plot("./plots/basic_plot.png")
        print(net)

    def basic_igraph(self):
        """Generate a basic igraph example to operate on."""
        pass

    def test_basic_igraph(self):
        """Testing initialising the basic igraph."""
        self.basic_igraph()


if __name__ == "__main__":
    logger.basicConfig(  # filename="{}.log".format(__file__),
        level=logger.INFO,
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S")
    # run all the unit-tests
    print("now testing:", __file__, "...")
    unittest.main()
