"""Test module for Neural network graph."""
# @Author: George Onoufriou <archer>
# @Date:   2021-07-24T15:01:00+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-23T10:01:41+01:00

import time
import unittest
import numpy as np
import networkx as nx
from fhez.nn.nn import NN

from fhez.nn.graph.node import IO
from fhez.nn.activation.relu import RELU  # Rectified Linear Unit (approx)


class NNTest(unittest.TestCase):
    """Test neural network graph."""

    def setUp(self):
        """Set up basic variables and start timer."""
        self.weights = (1, 3, 3, 3)  # tuple allows cnn to initialise itself
        self.stride = [1, 3, 3, 3]  # stride list per-dimension
        self.bias = 0  # assume no bias at first
        self.start_time = time.time()

        graph = nx.MultiDiGraph()
        graph.add_node("input", node=IO())
        graph.add_node("ReLU", node=RELU())
        graph.add_node("output", node=IO())
        graph.add_edge("input", "ReLU",
                       cost=4)  # graph.nodes["ReLU"]["node"].cost)
        graph.add_edge("ReLU", "output",
                       cost=graph.nodes["output"]["node"].cost)
        self.nn = NN(graph=graph)

    def tearDown(self):
        """Calculate time difference from start."""
        t = time.time() - self.start_time
        print('%s: %.3f' % (self.id(), t))

    @property
    def data(self):
        """Get random input data example."""
        array = np.random.rand(32, 32, 3)
        return array

    @property
    def datas(self):
        """Get random input data batch."""
        array = np.random.rand(2, 32, 32, 3)
        return array

    def test_init(self):
        """Test that object is initialised properly."""
        self.assertIsInstance(self.nn, NN)

    def test_cost(self):
        """Test that we can get the cost of an edge properly."""
        from fhez.nn.activation.relu import RELU

        self.assertEqual(self.nn.g.edges["input", "ReLU", 0]["cost"],
                         RELU().cost)

    def test_forward(self):
        """Testing single input/ example forward pass."""
        a = self.nn.forward(x=self.data, current_node="input",
                            end_node="output")
        print(self.nn.g.edges["input", "ReLU", 0])

    def test_forwards(self):
        """Testing multi-input/ examples forward pass."""
        a = self.nn.forwards(xs=self.datas,
                             current_node="input",
                             end_node="output")

    def test_backward(self):
        """Testing single input/ example backward pass."""
        a = self.nn.forward(x=self.data, current_node="input",
                            end_node="output")
        self.nn.backward(gradient=1, current_node="output",
                         end_node="input")

    def test_backwards(self):
        """Testing multi-input/ examples backward pass."""
        a = self.nn.forwards(xs=self.datas,
                             current_node="input",
                             end_node="output")
        self.nn.backwards(gradients=[1, 1], current_node="output",
                          end_node="input")

    def test_update(self):
        """Testing multi-input/ example updating."""
        a = self.nn.forward(x=self.data, current_node="input",
                            end_node="output")
        self.nn.backward(gradient=1, current_node="output",
                         end_node="input")
        self.nn.update(current_node="input", end_node="output")

    def test_updates(self):
        """Testing multi-input/ examples updating."""
        a = self.nn.forwards(xs=self.datas,
                             current_node="input",
                             end_node="output")
        self.nn.backwards(gradients=[1, 1], current_node="output",
                          end_node="input")
        self.nn.updates(current_node="input", end_node="output")

    def test_mnist(self):
        """Testing against MNIST."""
