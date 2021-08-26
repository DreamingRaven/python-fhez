"""Neural network generic firing abstraction."""
# @Author: George Onoufriou <archer>
# @Date:   2021-08-23T17:10:35+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-26T16:01:34+01:00

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

    @property
    def forward_name(self):
        return "forward"

    @property
    def backward_name(self):
        return "backward"

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
        signal_name = "forward" if is_forward_receptor is True else \
            "backward"
        graph = self.graph
        # get signal from edges behind us
        signal = self._get_signal(graph=graph, node_name=node_name,
                                  signal_name=signal_name, bootstrap=bootstrap)
        # if node is not ready I.E not all predecessors are processed skip
        if signal is None:
            return None

        # if node is the last one dont run it as we have no output edges
        # with which to store it in
        if len(list(self.graph.successors(node_name))) == 0:
            return None

        # get activation on application of signal to current node
        activation = self._use_signal(graph=graph,
                                      node_name=node_name, signal=signal,
                                      receptor_name=signal_name)

        # if the node has not activated then there is no need to compute
        if activation is None:
            return None

        # distibute activation to edges ahead of us
        self._propogate_signal(graph=graph, node_name=node_name,
                               signal_name=signal_name, signal=activation)
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

    def _use_signal(self, graph, node_name, receptor_name, signal):
        """Apply signal to given node in graph, and receptor."""
        # apply signal to current node
        method = getattr(
            graph.nodes(data=True)[node_name]["node"],
            receptor_name)
        activation = method(signal)
        return activation

    def _propogate_signal(self, graph, node_name, signal_name, signal):
        # distribute output-signal to outbound edges if any
        if signal is None:
            return None  # early exit no signal to propogate
        for (_, _, edge) in graph.edges(node_name, data=True):
            # if of type YIELD/ generator use next to iterate
            if isinstance(signal, types.GeneratorType):
                edge[signal_name] = next(signal)
            else:
                edge[signal_name] = signal

    def harvest(self, node_names: list):
        """Harvest forward response from neuronal firing, using probes.

        This will replay the last node to calculate its output.
        """
        accumulator = []
        for node_name in node_names:
            signal = self._get_signal(graph=self.graph, node_name=node_name,
                                      signal_name=self.forward_name)
            if signal is None:
                accumulator.append((node_name, None))
            else:
                activation = self._use_signal(graph=self.graph,
                                              node_name=node_name,
                                              receptor_name=self.forward_name,
                                              signal=signal)
                accumulator.append((node_name, activation))
        return accumulator

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
