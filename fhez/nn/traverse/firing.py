"""Neural network generic firing abstraction."""
# @Author: George Onoufriou <archer>
# @Date:   2021-08-23T17:10:35+01:00
# @Last modified by:   archer
# @Last modified time: 2021-09-21T16:48:39+01:00

import logging as logger
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
                  receptor="forward", debug=False):
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
        :arg receptor: Name of function/ sequence of functions to call of nodes
        :type receptor: str
        """
        assert len(neurons) == len(signals), \
            "Signals and receptors length (axis=0) should match"

        receptor = receptor if receptor is not None else "forward"
        # CLEAR GRAPH OF SPECIFIC RECEPTOR CACHE SO we dont use the existing
        # partial calculations this also reduces the need for catching
        # non existant key
        edges = self.graph.edges(data=True)
        for e in edges:
            e[2][receptor] = None

        # could use zip longest but zip will ensure atleast some can be
        # processed since it stops at the shortest of the two lists
        outputs = {}
        for (neuron, signal) in zip(neurons, signals):
            out = self._carry_signal(
                node_name=neuron, receptor=receptor,
                bootstrap=signal, debug=debug)
            outputs.update(out)
        return outputs

    def _carry_signal(self, node_name, receptor: str,
                      bootstrap: np.ndarray = None, outputs=None, debug=None):
        """Bootstrap and recursiveley carry signal through successor nodes."""
        graph = self.graph
        outputs = outputs if outputs is not None else {}
        debug = debug if debug is not None else False
        # get signal from edges behind us
        signal = self._get_signal(graph=graph, node_name=node_name,
                                  signal_name=receptor, bootstrap=bootstrap)
        # if node is not ready I.E not all predecessors are processed skip
        if signal is None:
            return None

        # some contextual logging
        msg = "{}:".format(node_name)
        if debug is True:
            print(msg)
        else:
            logger.debug(msg)

        # get activation on application of signal to current node
        activation = self._use_signal(graph=graph,
                                      node_name=node_name, signal=signal,
                                      receptor_name=receptor)

        # some incredibly important logging
        msg = "\trtype: {}, rshape: {}".format(
            type(activation),
            self.probe_shape(activation) if isinstance(
                activation, (types.GeneratorType, type(None))
            ) is not True else "?")
        if debug is True:
            print(msg)
        else:
            logger.debug(msg)

        # if the node has not activated then there is no need to compute
        if activation is None:
            return None

        # distibute activation to edges ahead of us
        self._propogate_signal(graph=graph, node_name=node_name,
                               signal_name=receptor, signal=activation)

        if len(graph.edges(node_name, data=False)) == 0:
            # this is a terminating node so record output
            msg = "output from this node: {} already exists somehow".format(
                node_name
            )
            assert outputs.get(node_name) is None, msg
            assert activation is not None, "activation of node cannot be none"
            outputs[node_name] = activation  # modify reference dictionary
        else:
            # recurse to all successors
            for next_node_name in self.graph.successors(node_name):
                out = self._carry_signal(
                    node_name=next_node_name,
                    receptor=receptor,
                    bootstrap=None,
                    outputs=None,
                    debug=debug)
                outputs.update(out if out is not None else {})
        return outputs

    def _get_signal(self, graph, node_name, signal_name, bootstrap=None):
        # Get current nodes signal or bootstrap signal
        if bootstrap is None:
            signal = []
            for edge in graph.in_edges(node_name, data=True):
                # edge = tuple("source_node", "dest_node", {attributes})
                edge_signal = edge[2].get(signal_name)
                if edge_signal is None:
                    return None  # early exit no.1 if signal is nothing avoid
                signal.append(edge_signal)
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
