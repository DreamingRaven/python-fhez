#!/usr/bin/env python3

# @Author: George Onoufriou <archer>
# @Date:   2021-07-11T14:35:36+01:00
# @Last modified by:   archer
# @Last modified time: 2021-07-16T01:01:59+01:00

import time
import unittest
import logging as logger

import numpy as np


# graphing libs
from networkx import nx
# import igraph


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

    def forward(self, x, current_node, end_node):
        """Traverse and activate nodes until all nodes processed."""
        node = self.g.nodes[current_node]
        logger.debug("processing node: `{}`, input_shape({})".format(
            current_node,
            self.probe_shape(x)))
        # process current node
        output = node["node"].forward(x)
        # process next nodes recursiveley
        next_nodes = self.g.successors(current_node)
        for i in next_nodes:
            self.forward(x=output,
                         current_node=i,
                         end_node=end_node)

    def backward(self, gradient, current_node, end_node):
        """Traverse backwards until all nodes processed."""
        node = self.g.nodes[current_node]
        logger.debug("processing node: `{}`, gradient({})".format(
            current_node,
            gradient))
        # process current nodes gradients
        local_gradient = node["node"].backward(gradient)
        # process previous nodes recursiveley
        previous_nodes = self.g.predecessors(current_node)
        for i in previous_nodes:
            self.backward(gradient=local_gradient,
                          current_node=i,
                          end_node=end_node)

    def forwards(self, xs, current_node, end_node):
        """Calculate forward pass for multiple examples simultaneously."""
        accumulator = []
        for i in xs:
            accumulator.append(
                self.forward(
                    x=i,
                    current_node=current_node,
                    end_node=end_node))
        return accumulator

    def backwards(self, gradients, current_node, end_node):
        """Calculate backward pass for multiple examples simultaneously."""
        accumulator = []
        for i in gradients:
            accumulator.append(
                self.backward(
                    gradient=i,
                    current_node=current_node,
                    end_node=end_node))
        return accumulator

    def update(self, current_node, end_node):
        """Update weights of all nodes using oldest single example gradient."""
        node = self.g.nodes[current_node]
        logger.debug("updating node: `{}`".format(current_node))
        # update current node
        node["node"].update()
        # process next nodes recursiveley
        next_nodes = self.g.successors(current_node)
        for i in next_nodes:
            # update successors recursiveley
            self.update(current_node=i, end_node=end_node)

    def updates(self, current_node, end_node):
        """Update the weights of all nodes by taking the average gradient."""
        node = self.g.nodes[current_node]
        logger.debug("updating node: `{}`".format(current_node))
        # update current node
        node["node"].updates()
        # process next nodes recursiveley
        next_nodes = self.g.successors(current_node)
        for i in next_nodes:
            # update successors recursiveley
            self.updates(current_node=i, end_node=end_node)

    def probe_shape(self, lst: list):
        """Get the shape of a list, assuming each sublist is the same length.

        This function is recursive, sending the sublists down and terminating
        once a type error is thrown by the final point being a non-list
        """
        if isinstance(lst, list):
            # try appending current length with recurse of sublist
            try:
                return (len(lst),) + self.probe_shape(lst[0])
            # once we bottom out and get some non-list type abort and pull up
            except (AttributeError, IndexError):
                return (len(lst),)
        elif isinstance(lst, (int, float)):
            return (1,)
        else:
            return lst.shape


# Shorthand / Alias for Neural Network
NN = NeuralNetwork


class NNTest(unittest.TestCase):

    def setUp(self):
        """Set up basic variables and start timer."""

        from fhez.nn.graph.node import IO
        from fhez.nn.activation.relu import RELU

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


if __name__ == "__main__":
    logger.basicConfig(  # filename="{}.log".format(__file__),
        level=logger.INFO,
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S")
    # run all the unit-tests
    print("now testing:", __file__, "...")
    unittest.main()
