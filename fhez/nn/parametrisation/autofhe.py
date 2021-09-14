"""Automatic FHE/ CKKS parameterisation utilities."""
# @Author: George Onoufriou <archer>
# @Date:   2021-09-14T10:34:17+01:00
# @Last modified by:   archer
# @Last modified time: 2021-09-14T11:51:13+01:00


def autoHE(graph, node, cost=0):
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
    :rtype: networkx.MultiDiGraph
    :return: modified networks graph
    """
