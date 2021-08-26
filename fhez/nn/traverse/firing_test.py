# @Author: George Onoufriou <archer>
# @Date:   2021-08-23T17:19:31+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-26T13:28:54+01:00

import time
import unittest
import numpy as np
import networkx as nx

from fhez.nn.graph.io import IO
from fhez.nn.activation.relu import RELU
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
        graph.add_node("y", node=IO())  # from this node
        graph.add_edge("x", "y", fwd=some_signal)
        graph.add_edge("x", "y", fwd=some_signal)
        graph.add_edge("x", "y", fwd=some_signal)
        graph.add_edge("a", "y", fwd=some_signal)
        f = Firing()
        signal = f._get_signal(graph=graph, node_name="y", signal_name="fwd")
        truth = np.broadcast_to(some_signal, shape=(4, 3))
        # should return all inputs in meta container of shape (4,3) here
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
        graph.add_node("y", node=IO())  # from this node
        graph.add_edge("x", "y", fwd=some_signal)
        f = Firing()
        signal = f._get_signal(graph=graph, node_name="y", signal_name="fwd")
        # should return just the input without meta container
        np.testing.assert_array_almost_equal(signal, some_signal,
                                             decimal=1,
                                             verbose=True)

    def test_get_signal_none(self):
        """Check get non-existant signal is working as expected.

        If the node does not have all of its inputs then it is not ready.
        So getsignal should return None as the signal is incomplete.
        """
        # create a graph with 3 parallel edges from one node and one other
        some_signal = np.array([1, 2, 3])
        graph = nx.MultiDiGraph()
        graph.add_node("x", node=IO())
        graph.add_node("a", node=IO())
        graph.add_node("y", node=IO())  # from this node
        graph.add_edge("x", "y", fwd=some_signal)
        graph.add_edge("x", "y")  # some missing signal
        graph.add_edge("x", "y", fwd=some_signal)
        graph.add_edge("a", "y", fwd=some_signal)
        f = Firing()
        signal = f._get_signal(graph=graph, node_name="y", signal_name="fwd")
        self.assertEqual(signal, None)

    def test_use_signal(self):
        """Check that func is modifying the state of the node properly."""
        some_signal = np.array([1, 0.5, 0])
        some_signal = np.broadcast_to(some_signal, shape=(4, 3))
        graph = nx.MultiDiGraph()
        graph.add_node("y", node=RELU())
        f = Firing()
        activation = f._use_signal(graph=graph, node_name="y",
                                   receptor_name="forward", signal=some_signal)
        activation_truth = RELU().forward(some_signal)
        np.testing.assert_array_almost_equal(activation, activation_truth,
                                             decimal=1,
                                             verbose=True)

    def test_propogate_signal(self):
        """Check that signal propogation occurs properly for all edges.

        This is standard iterable return NOT YIELD, applied to each edge.
        """
        some_signal = np.array([1, 2, 3])
        graph = nx.MultiDiGraph()
        graph.add_node("x", node=IO())  # from this node
        graph.add_node("a", node=IO())
        graph.add_node("y", node=IO())
        graph.add_edge("x", "y")  # no signals yet
        graph.add_edge("x", "y")
        graph.add_edge("x", "y")
        graph.add_edge("a", "y")  # should never recieve a signal
        f = Firing()
        f._propogate_signal(graph=graph, node_name="x", signal_name="fwd",
                            signal=some_signal)
        # check signal has been applied to each individually
        for edge in graph.edges("x", data=True):
            np.testing.assert_array_almost_equal(edge[2]["fwd"],
                                                 some_signal,
                                                 decimal=1,
                                                 verbose=True)
        # check not applied to seperate node
        for edge in graph.edges("a", data=True):
            self.assertEqual(edge[2].get("fwd"), None)
