"""Basic graph utility functions to help graph construction and maintenance."""
# @Author: George Onoufriou <archer>
# @Date:   2021-09-16T14:10:58+01:00
# @Last modified by:   archer
# @Last modified time: 2021-09-20T13:54:24+01:00


def assign_edge_costing(graph):
    """Modify a graph so edges represent costs of the forward node."""
    # for every node
    for node in graph.nodes(data=True):
        # assign node.cost() to every inbound edge
        for edge in graph.in_edges(node[0], data=True):
            edge[2]["weight"] = node[1]["node"].cost


def train(graph, sources, sinks, inputs, outcomes):
    """Train neural network graph through backpropagation."""
    pass


def infer(graph, sources, sinks, inputs):
    """Use neural network graph to infer some outcomes from inputs."""
    pass
