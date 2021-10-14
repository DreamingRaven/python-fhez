"""Automatic FHE/ CKKS parameterisation utilities."""
# @Author: George Onoufriou <archer>
# @Date:   2021-09-14T10:34:17+01:00
# @Last modified by:   archer
# @Last modified time: 2021-10-14T13:21:20+01:00

from fhez.nn.graph.utils import assign_edge_costing
from fhez.nn.operations.encrypt import Encrypt
from fhez.nn.operations.decrypt import Decrypt
from fhez.nn.operations.rotate import Rotate


def source_prop(graph,
                source: str,
                node: str,
                concern: list = None,
                cost: int = None):
    cost = cost if cost is not None else 0
    concern = tuple(concern if concern is not None else [Encrypt,
                                                         Decrypt,
                                                         Rotate])
    # gather facts
    node_object = graph.nodes(data=True)[node]["node"]
    node_data = graph.nodes(data=True)[node]

    # on all nodes
    if node_data.get("sources") is None:
        node_data["sources"] = {}
    if source != node:
        if node_data["sources"].get(source) is None:
            node_data["sources"][source] = cost
        elif node_data["sources"].get(source) < cost:
            node_data["sources"][source] = cost

    print(node, node_data["sources"])

    # on concerned nodes which is not our initial one
    if isinstance(node_object, concern) and source != node:
        source_prop(graph, source=node, node=node, concern=concern, cost=0)
    else:
        # on every other node
        for i in graph.successors(node):
            next_cost = graph.nodes(data=True)[i]["node"].cost
            source_prop(graph, source=source, node=i, concern=concern,
                        cost=cost+next_cost)


def autoHE(graph, nodes, concern=None, cost_edges=None):
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

    .. warning::

        This function is still a work in progress, and is subject to change!
    """
    cost_edges = cost_edges if cost_edges is not None else True
    concern = tuple(concern if concern is not None else [Encrypt,
                                                         Decrypt,
                                                         Rotate])
    assert graph is not None, "graph should exist, cannot operate on none!"

    # ensures all edges are labeled with weights that correspond to
    # computational cost of traversing to the associated node
    if cost_edges is True:  # should only be run once as unecessary twice
        assign_edge_costing(graph)
    for i in nodes:
        source_prop(graph=graph, source=i, node=i, concern=concern)
