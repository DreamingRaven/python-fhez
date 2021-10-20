"""Automatic FHE/ CKKS parameterisation utilities."""
# @Author: George Onoufriou <archer>
# @Date:   2021-09-14T10:34:17+01:00
# @Last modified by:   archer
# @Last modified time: 2021-10-20T11:24:02+01:00

import numpy as np
import logging as logger
from fhez.rearray import ReArray
from fhez.nn.graph.utils import assign_edge_costing
from fhez.nn.operations.encrypt import Encrypt
from fhez.nn.operations.decrypt import Decrypt
from fhez.nn.operations.rotate import Rotate


def autoDiscover(graph,
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
        autoDiscover(graph, source=node, node=node, concern=concern, cost=0)
    else:
        # on every other node
        for i in graph.successors(node):
            next_cost = graph.nodes(data=True)[i]["node"].cost
            autoDiscover(graph, source=source, node=i, concern=concern,
                         cost=cost+next_cost)


def autoGroup(graph, nodes, concern=None, cost_edges=None):
    """Calculate all encryption groups in neural network, and their costs."""
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
        autoDiscover(graph=graph, source=i, node=i, concern=concern)

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
            src = j[1].get("sources")

            if src is None:
                logger.warning("{} does not have {}".format(j, "sources"))
            elif i in src:
                # for each key in this nodes sources
                for key in src:
                    # drag every keys group to our group number
                    # TODO: This should drag ALL members who are in the group
                    # existing group. Currentley it will only drag this one
                    # member if there are many.
                    groups[0][key] = groups[0][i]
                    # increase group cost if greater than ours
                    if src[key] > groups[1][groups[0][i]]:
                        groups[1][groups[0][i]] = src[key]
    # print("GROUPS", groups)
    return groups


def ckks_param_heuristic(cost, scale_pow=40, special_mult=1.5):
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


def autoHE(graph, nodes, parm_func=None, provider=None,
           concern=None, cost_edges=None,
           **kwargs):
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
    # setting sane defaults to RNS-CKKS scheme
    parm_func = parm_func if parm_func is not None else ckks_param_heuristic
    provider = provider if provider is not None else ReArray
    concern = tuple(concern if concern is not None else [Encrypt,
                                                         Decrypt,
                                                         Rotate])
    # label the graph and get all the groups, costs, etc
    groups_nodes, groups_costs = autoGroup(graph, nodes, concern, cost_edges)
    groups_encryptors = []
    # for each group (no names just intiger keys) in list of costs
    for group in range(len(groups_costs)):
        # generate the respective parameters with optional kwargs
        parms = parm_func(cost=groups_costs[group], **kwargs)
        # this is now the cyphertext generator shared between grouped nodes
        # adding np array in case provider expects default input
        encryptor = provider(np.array([1]), **parms)
        groups_encryptors.append(encryptor)
        # limit the nodes searched to just the ones related to this group
        group_nodes = {
            key: value for key, value in groups_nodes.items() if value == group
        }
        for key in group_nodes:
            graph.nodes(data=True)[key]["node"].encryptor = encryptor

    return groups_nodes, groups_costs, groups_encryptors
