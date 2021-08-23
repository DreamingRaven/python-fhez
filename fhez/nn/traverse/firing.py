"""Neural network generic firing abstraction."""
# @Author: George Onoufriou <archer>
# @Date:   2021-08-23T17:10:35+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-23T20:27:10+01:00

from fhez.nn.traverse.traverser import Traverser


class Firing(Traverser):
    """Simple exhaustive neuronal firing calculation."""

    @property
    def graph(self):
        """Get neuron graph to fire."""
        return self.__dict__.get("_graph")

    @graph.setter
    def graph(self, graph):
        self._graph = graph

    def stimulate(self, receptors, signals):
        """Stimulate a set of receptors with a set of signals for response.

        Breadth first stimulation of neurons/ nodes.
        Note that this is a single simultaneous stimulation and subsequent
        response.

        :arg receptors: list of node names to recieve stimulus
        :type receptors: list(str)
        :arg signals: positional list of signals for the equally positioned
            receptor
        :type signals: np.ndarray or compatible
        """

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
