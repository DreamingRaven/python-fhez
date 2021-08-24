#!/usr/bin/env python3

# @Author: George Onoufriou <archer>
# @Date:   2021-07-11T14:35:36+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-24T05:47:24+01:00

import logging as logger


# graphing libs
# from networkx import nx
import networkx as nx
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
