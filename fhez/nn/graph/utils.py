"""Basic graph utility functions to help graph construction and maintenance."""
# @Author: George Onoufriou <archer>
# @Date:   2021-09-16T14:10:58+01:00
# @Last modified by:   archer
# @Last modified time: 2021-09-23T11:42:26+01:00

import itertools
import numpy as np
from tqdm import tqdm
from fhez.nn.traverse.firing import Firing


def assign_edge_costing(graph):
    """Modify a graph so edges represent costs of the forward node."""
    # for every node
    for node in graph.nodes(data=True):
        # assign node.cost() to every inbound edge
        for edge in graph.in_edges(node[0], data=True):
            edge[2]["weight"] = node[1]["node"].cost


def train(graph, inputs, batch_size, debug=False):
    """Train neural network graph through backpropagation."""
    neurons = list(inputs.keys())

    # setting up our graph in both normal and reversed directions for
    # forward and backward pass
    forward = Firing(graph=graph)
    backward = Firing(graph=graph.reverse(copy=False))  # we want them linked

    train = list(inputs.values())
    # external counter as I want to rework this in future to work
    # with generators + its more efficient to use itertools than to
    # iterate over the training set manually using a counter + lookup
    # on each iteration which would start from head of list
    i = 0
    with tqdm(total=len(inputs[neurons[0]]), desc="Learn") as pbar:
        for signals in itertools.zip_longest(*train):
            # forward pass over all avaliable nodes on graph
            out = forward.stimulate(
                neurons=neurons,
                signals=list(signals),
                receptor="forward")
            # backward pass using output from forward pass to select
            # the nodes they came from to pass them back in but as
            # losses (or ignored if not a loss)
            backward.stimulate(
                neurons=list(out.keys()),
                signals=list(out.values()),
                receptor="backward")
            # if we happen to be at the end of a batch update using avg of
            # our calculated gradients in all backward passes
            # (internal state of the graph nodes so no need to do it
            # ourselves)
            if i % batch_size == 0:
                for node_meta in graph.nodes(data=True):
                    node = node_meta[1]["node"]
                    node.updates()
            # iterate counter to keep track of batch sizes
            pbar.update(1)
            i += 1
    return out


def infer(graph, inputs):
    """Use neural network graph to infer some outcomes from inputs."""
    neurons = list(inputs.keys())

    # setting up our graph in both normal and reversed directions for
    # forward and backward pass
    forward = Firing(graph=graph)

    train = list(inputs.values())
    # external counter as I want to rework this in future to work
    # with generators + its more efficient to use itertools than to
    # iterate over the training set manually using a counter + lookup
    # on each iteration which would start from head of list
    i = 0
    activations = {}
    with tqdm(total=len(inputs[neurons[0]]), desc="Infer") as pbar:
        for signals in itertools.zip_longest(*train):
            # forward pass over all avaliable nodes on graph
            out = forward.stimulate(
                neurons=neurons,
                signals=list(signals),
                receptor="forward")
            pbar.update(1)
            i += 1
            # remapping outputs into a single dictionary again just like inputs
            for key, value in out.items():
                if activations.get(key) is None:
                    activations[key] = []
                activations[key].append(value)
    return activations
