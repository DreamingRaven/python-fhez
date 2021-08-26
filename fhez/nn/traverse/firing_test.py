# @Author: George Onoufriou <archer>
# @Date:   2021-08-23T17:19:31+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-26T12:57:17+01:00

import time
import unittest
import numpy as np
import networkx as nx

from fhez.nn.graph.io import IO
from fhez.nn.graph.prefab import cnn_classifier
from fhez.nn.traverse.firing import Firing


class FiringTest(unittest.TestCase):
    """Test linear activation function."""

    def setUp(self):
        """Start timer and init variables."""
        self.start_time = time.time()

    def tearDown(self):
        """Calculate and print time delta."""
        t = time.time() - self.start_time
        print('%s: %.3f' % (self.id(), t))

    @property
    def data_shape(self):
        """Define desired data shape."""
        return (28, 28)

    @property
    def data(self):
        """Get some generated data."""
        array = np.random.rand(*self.data_shape)
        return array

    @property
    def reseal_args(self):
        """Get some reseal arguments for encryption."""
        return {
            "scheme": 2,  # seal.scheme_type.CKK,
            "poly_modulus_degree": 8192*2,  # 438
            # "coefficient_modulus": [60, 40, 40, 60],
            "coefficient_modulus":
                [45, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 45],
            "scale": pow(2.0, 30),
            "cache": True,
        }

    @property
    def graph(self):
        """Get neuron/ computational graph to test against."""
        return cnn_classifier(10)

    def test_stimulate_forward(self):
        """Check neuronal firing algorithm forward stimulation of graph."""
        graph = self.graph
        data = self.data
        f = Firing(graph=graph)
        f.stimulate(neurons=["x", "y"], signals=[data, 1])

    def test_get_signal_many(self):
        """Check get multi signal is working as expected.

        This includes considering parallel edges, and standard multi edges.
        """
        # create a graph with 3 parallel edges from one node and one other
        some_signal = np.array([1, 2, 3])
        graph = nx.MultiDiGraph()
        graph.add_node("x", node=IO())
        graph.add_node("a", node=IO())
        graph.add_node("y", node=IO())
        graph.add_edge("x", "y", fwd=some_signal)
        graph.add_edge("x", "y", fwd=some_signal)
        graph.add_edge("x", "y", fwd=some_signal)
        graph.add_edge("a", "y", fwd=some_signal)
        f = Firing()
        signal = f._get_signal(graph=graph, node_name="y", signal_name="fwd")
        truth = np.broadcast_to(some_signal, shape=(4, 3))
        np.testing.assert_array_almost_equal(signal, truth,
                                             decimal=1,
                                             verbose=True)

    def test_get_signal_one(self):
        """Check get single signal is working as expected.

        This includes single edges which should not be in a meta container.
        """
        some_signal = np.array([1, 2, 3])
        graph = nx.MultiDiGraph()
        graph.add_node("x", node=IO())
        graph.add_node("y", node=IO())
        graph.add_edge("x", "y", fwd=some_signal)
        f = Firing()
        signal = f._get_signal(graph=graph, node_name="y", signal_name="fwd")
        np.testing.assert_array_almost_equal(signal, some_signal,
                                             decimal=1,
                                             verbose=True)
