"""Automatic FHE/ CKKS parameterisation utilities."""
# @Author: George Onoufriou <archer>
# @Date:   2021-09-14T10:34:17+01:00
# @Last modified by:   archer
# @Last modified time: 2021-09-16T14:47:52+01:00

from fhez.nn.graph.utils import assign_edge_costing
from fhez.nn.operations.encrypt import Encrypt


def autoHE(graph, node: str, cost: int = None,
           cost_edges: bool = None, concern: list = None):
    """Adjust and generate parameters along full forward path of input nodes.

    A graph may have multiple input nodes, and each one will have slightly
    different paths which may need different parameters. We dont worry about
    output nodes since the full forward graph needs activating which means we
    can automatically find the end nodes.

    This will modify the input graph, but only needs to be done once to set
    the optimal parms!

    :arg graph: A neural network graph to automatically parameterise.
    :type graph: networkx.MultiDiGraph
    :arg node: Input node name to parameterise from
    :type input_nodes: str
    :arg cost: current cost up till this node from previous key-rotation
    :arg concern: types list which is used to consume cost
    :rtype: networkx.MultiDiGraph
    :return: modified networks graph
    """
    cost = cost if cost is not None else 0
    cost_edges = cost_edges if cost_edges is not None else True
    concern = tuple(concern if concern is not None else [Encrypt])
    assert graph is not None, "graph should exist, cannot operate on none!"

    # ensures all edges are labeled with weigts that correspond to
    # computational cost of traversing to the associated node
    if cost_edges is True:  # should only be run once as unecessary twice
        assign_edge_costing(graph)

    # if this node is an encryptor or other concerned type
    node_object = graph.nodes(data=True)[node]["node"]
    if isinstance(node_object, concern):
        # check if node already has a higher forward cost
        # consume cost either into node or skip if already higher cost
        pass

    # for each forward edge recurse with their respective costs + our current
    for edge in graph.edges(node, data=True):
        print(edge)
