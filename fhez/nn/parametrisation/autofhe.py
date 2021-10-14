"""Automatic FHE/ CKKS parameterisation utilities."""
# @Author: George Onoufriou <archer>
# @Date:   2021-09-14T10:34:17+01:00
# @Last modified by:   archer
# @Last modified time: 2021-10-14T16:22:42+01:00

import numpy as np
from fhez.nn.graph.utils import assign_edge_costing
from fhez.nn.operations.encrypt import Encrypt
from fhez.nn.operations.decrypt import Decrypt
from fhez.nn.operations.rotate import Rotate


def source_prop(graph,
                source: str,
                node: str,
                concern: list = None,
                cost: int = None):
    """Propagate some source through the network.

    .. note::

        You are porbably not interested in this function, it is here for manual
        control and as a subpart of autoHE and in future autoFHE.
    """
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

    # print(node, node_data["sources"])

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

    # use the now calculated paths to calculate groupings, and who takes
    # from whom
    # groups as a tuple:
    # - dict of (key) node names to group number (value)
    # - list of (key) group numbers, maximum group cost (value)
    groups = ({}, [])
    # for each input node
    for i in nodes:
        if groups[0].get(i) is None:
            groups[0][i] = len(groups[1])
            groups[1].append(0)
        t = graph.nodes(data=True)
        # for each node in the graph
        for j in t:
            src = j[1]["sources"]
            if i in src:
                # for each key in this nodes sources
                for key in src:
                    # drag every keys group to our group number
                    groups[0][key] = groups[0][i]
                    # increase group cost if greater than ours
                    if src[key] > groups[1][groups[0][i]]:
                        groups[1][groups[0][i]] = src[key]
    # print("GROUPS", groups)
    return groups


def temp_encryptor_generator(cost, scale_pow=30, special_mult=1.5):
    """Given some cost generate encryption parameters.

    .. note::

        This is highly subject to removal, this is here in the interip untill a
        full specification for Erray can be finalised.

    +---------------------+------------------------------+
    | poly_modulus_degree | max coeff_modulus bit-length |
    +=====================+==============================+
    | 1024                | 27                           |
    +---------------------+------------------------------+
    | 2048                | 54                           |
    +---------------------+------------------------------+
    | 4096                | 109                          |
    +---------------------+------------------------------+
    | 8192                | 218                          |
    +---------------------+------------------------------+
    | 16384               | 438                          |
    +---------------------+------------------------------+
    | 32768               | 881                          |
    +---------------------+------------------------------+

    slots (CKKS) = poly_modulus_degree/2
    """
    dummy = {
        "scheme": 2,  # seal.scheme_type.CKK,
        "poly_modulus_degree": 8192*2,  # 438
        # "coefficient_modulus": [60, 40, 40, 60],
        "coefficient_modulus":
        [45, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 45],
        "scale": pow(2.0, 30),
        "cache": True,
    }
    coef_mod = np.ones((cost+2,)) * scale_pow
    coef_mod[0] *= special_mult
    coef_mod[-1] *= special_mult
    max_coef_mod_bits = 27
    while max_coef_mod_bits < np.sum(coef_mod):
        max_coef_mod_bits *= 2

    parms = {
        "scheme": 2,
        "poly_modulus_degree": int(1024 * (max_coef_mod_bits/27)),
        "coefficient_modulus": coef_mod.astype(int).tolist(),
        "scale": pow(2.0, scale_pow),
        "cache": True,
    }
    return parms
