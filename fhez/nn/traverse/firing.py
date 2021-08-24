"""Neural network generic firing abstraction."""
# @Author: George Onoufriou <archer>
# @Date:   2021-08-23T17:10:35+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-24T10:53:39+01:00

import types
import itertools
import numpy as np
from fhez.nn.traverse.traverser import Traverser


class Firing(Traverser):
    """Simple exhaustive neuronal firing calculation."""

    def __init__(self, graph=None):
        """Initialise a neuronal firing object, pre-populated."""
        self.graph = graph

    @property
    def graph(self):
        """Get neuron graph to fire."""
        return self.__dict__.get("_graph")

    @graph.setter
    def graph(self, graph):
        self._graph = graph

    def stimulate(self, neurons: np.ndarray, signals: np.ndarray,
                  is_forward_receptor: bool = True):
        """Stimulate a set of receptors with a set of signals for response.

        Breadth first stimulation of neurons/ nodes.
        Note that this is a single simultaneous stimulation and subsequent
        response. If a neuron does not fire I.E produces no (None) result
        then that neuron will not be followed until it does produce a result.

        :arg receptors: list of node names to recieve stimulus
        :type receptors: list(str)
        :arg signals: positional list of signals for the equally positioned
            receptor
        :type signals: np.ndarray or compatible
        """
        assert len(neurons) == len(signals), \
            "Signals and receptors length (axis=0) should match"
        # could use zip longest but zip will ensure atleast some can be
        # processed since it stops at the shortest of the two lists
        for (neuron, signal) in zip(neurons, signals):
            self._carry_signal(
                node_name=neuron, is_forward_receptor=is_forward_receptor,
                bootstrap=signal)

    def _carry_signal(self, node_name, is_forward_receptor: bool = None,
                      bootstrap: np.ndarray = None):
        """Bootstrap and recursiveley carry signal through successor nodes."""
        signal_name = "fwd-signal" if is_forward_receptor is True else \
            "bwd-signal"
        graph = self.graph
        # Get current nodes signal or bootstrap signal
        if bootstrap is None:
            signal = []
            # how we can access predecessor edges
            for prev_node, adjacency in graph.pred[node_name].items():
                for edge in adjacency.items():
                    # tuple(index, dict(attributes))
                    signal.append(edge[1][signal_name])
                    del edge[1][signal_name]  # clean up after ourselves
            # for edge in self.graph.in_edges(node_name):
            #     signal.append(edge[signal_name])
            #     del edge[signal_name]
        else:
            signal = bootstrap

        node = self.graph.nodes[node_name]

        # apply signal to current node
        if is_forward_receptor:
            activation = node["node"].forward(signal)
        else:
            activation = node["node"].backward(signal)

        # distribute output-signal to outbound edges if any
        if activation is None:
            return None  # early exit no signal to propogate
        for next_node, adjacency in graph[node_name].items():
            for edge in adjacency.items():
                if isinstance(activation, types.GeneratorType):
                    # tuple(index, dict(attributes))
                    edge[1][signal_name] = next(activation)
                else:
                    # tuple(index, dict(attributes))
                    edge[1][signal_name] = activation

        # recurse to all successors
        for next_node in self.graph.successors(node_name):
            print("NEXT", next_node)
            # self._carry_signal(
            #     node_name=next_node, is_forward_receptor=is_forward_receptor,
            #     bootstrap=None)

    def harvest(self, probes):
        """Harvest forward response from neuronal firing, using probes."""

    def correction(self, signals, receptors):
        """Calculate/ learn correction necessary to become closer to our goal.

        :arg signals: signal to be induced in corresponding receptor
        :arg receptors: receptor to be signaled
        """

    def adaptation(self):
        """Correct nodes based on learnt gradient."""


NeuronalFiring = Firing
