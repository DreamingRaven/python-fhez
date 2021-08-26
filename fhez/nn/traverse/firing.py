"""Neural network generic firing abstraction."""
# @Author: George Onoufriou <archer>
# @Date:   2021-08-23T17:10:35+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-26T12:57:38+01:00

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
        node = graph.nodes[node_name]
        # print("Node: ", node_name, node["node"])
        # get signal from edges behind us
        signal = self._get_signal(graph=graph, node_name=node_name,
                                  signal_name=signal_name, bootstrap=bootstrap)
        # if node is not ready I.E not all predecessors are processed skip
        if signal is None:
            return None

        # get activation on application of signal to current node
        activation = self._use_signal(node=node, signal=signal,
                                      is_forward_receptor=is_forward_receptor)
        # distibute activation to edges ahead of us
        self._propogate_signal(graph=graph, node_name=node_name,
                               signal_name=signal_name, activation=activation)
        # recurse to all successors
        for next_node_name in self.graph.successors(node_name):
            self._carry_signal(
                node_name=next_node_name,
                is_forward_receptor=is_forward_receptor,
                bootstrap=None)
        return None

    def _get_signal(self, graph, node_name, signal_name, bootstrap=None):
        # Get current nodes signal or bootstrap signal
        if bootstrap is None:
            signal = []
            for edge in graph.in_edges(node_name, data=True):
                try:
                    # edge = tuple("source_node", "dest_node", attributes)
                    signal.append(edge[2][signal_name])
                except KeyError:
                    return None
            if len(signal) == 1:
                signal = signal[0]
        else:
            signal = bootstrap
        return signal

    def _use_signal(self, node, signal, is_forward_receptor=True):
        # apply signal to current node
        # print("signal-shape", self.probe_shape(signal))
        if is_forward_receptor:
            activation = node["node"].forward(signal)
        else:
            activation = node["node"].backward(signal)
        return activation

    def _propogate_signal(self, graph, node_name, signal_name, activation):
        # distribute output-signal to outbound edges if any
        if activation is None:
            return None  # early exit no signal to propogate
            # how we access successor edges
        for next_node, adjacency in graph[node_name].items():
            for edge in adjacency.items():
                if isinstance(activation, types.GeneratorType):
                    # tuple(index, dict(attributes))
                    edge[1][signal_name] = next(activation)
                else:
                    # tuple(index, dict(attributes))
                    edge[1][signal_name] = activation

    def harvest(self, probes):
        """Harvest forward response from neuronal firing, using probes."""

    def correction(self, signals, receptors):
        """Calculate/ learn correction necessary to become closer to our goal.

        :arg signals: signal to be induced in corresponding receptor
        :arg receptors: receptor to be signaled
        """

    def adaptation(self):
        """Correct nodes based on learnt gradient."""

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


NeuronalFiring = Firing
