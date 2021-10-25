"""Selection of various prefabricated neural networks and generators."""
# @Author: George Onoufriou <archer>
# @Date:   2021-08-23T17:22:55+01:00
# @Last modified by:   archer
# @Last modified time: 2021-10-21T12:52:45+01:00

import numpy as np

import networkx as nx

from fhez.nn.graph.io import IO
from fhez.nn.operations.sum import Sum
from fhez.nn.operations.cc import CC  # Cross Correlation
from fhez.nn.layer.ann import ANN  # Dense/ Artificial Neural Network
from fhez.nn.activation.relu import RELU  # Rectified Linear Unit (approx)

from fhez.nn.activation.softmax import Softmax
from fhez.nn.activation.argmax import Argmax

from fhez.nn.loss.cce import CCE  # categorical cross entropy
from fhez.nn.loss.mse import MSE  # Mean of the Squared Error

from fhez.nn.operations.encrypt import Encrypt
from fhez.nn.operations.decrypt import Decrypt
from fhez.nn.operations.rotate import Rotate

from fhez.nn.operations.selector import Selector
from fhez.nn.operations.distributor import Distributor

from fhez.nn.operations.enqueue import Enqueue
from fhez.nn.operations.dequeue import Dequeue

from fhez.nn.operations.one_hot_encode import OneHotEncode
from fhez.nn.operations.one_hot_decode import OneHotDecode


def cnn_regressor(data_shape, filter_length, stride=1):
    """Get simple 1 Layer 1D-CNN for time-series regression.

    :arg data_shape: The shape of the input data in the shape format of
     (timesteps, features)
    :type data_shape: tuple
    :arg filter_length: The length of the 1D CNN filter which to CC over data
    :type filter_length: int
    :arg stride: The steps between filters (default = 1)
    :type stride: int
    :return: neural network graph
    :rtype: networkx.MultiDigGraph
    """
    graph = nx.MultiDiGraph()
    data_shape = data_shape
    filter_shape = (filter_length, data_shape[1])
    stride = [stride, data_shape[1]]
    # creating window expression so we know how many nodes we need
    windows = CC().windex(data_shape,
                          filter_shape,
                          stride)

    # INPUTS
    graph.add_node("x", group=0, node=Rotate())
    graph.add_node("y", group=0, node=Rotate())

    # 1D CNN/ CC
    graph.add_node("1D-CC", group=1,
                   node=CC(weights=filter_shape, stride=stride, bias=0))
    graph.add_edge("x", "1D-CC")
    graph.add_node("CC-dequeue", group=6, node=Dequeue())
    graph.add_edge("1D-CC", "CC-dequeue")
    graph.add_node("CC-enqueue", group=6,
                   node=Enqueue(length=len(windows)))
    for i in range(len(windows)):
        graph.add_node("CC-sop-{}".format(i), group=1, node=Sum())
        graph.add_edge("CC-dequeue", "CC-sop-{}".format(i))
        graph.add_edge("CC-sop-{}".format(i), "CC-enqueue")
    graph.add_node("CNN-acti", group=1, node=RELU())
    graph.add_edge("CC-enqueue", "CNN-acti")

    # DENSE
    graph.add_node("Dense", group=2,
                   node=ANN(weights=(len(windows),)))
    graph.add_edge("CNN-acti", "Dense", weight=ANN().cost)
    graph.add_node("Dense-acti", group=2, node=RELU())
    graph.add_edge("Dense", "Dense-acti")
    graph.add_node("Decrypt", group=5, node=Decrypt())
    graph.add_edge("Dense-acti", "Decrypt")
    graph.add_node("Selector", group=6, node=Selector(backward=[1, 0]))
    graph.add_edge("Decrypt", "Selector")

    # LOSS
    graph.add_node("MSE", group=3, node=MSE())
    graph.add_edge("Selector", "MSE", weight=MSE().cost)
    graph.add_edge("y", "MSE", weight=MSE().cost)

    # OUTPUT
    graph.add_node("y_hat", group=4, node=IO())
    graph.add_edge("Selector", "y_hat", weight=IO().cost)

    return graph


def cnn_classifier(k):
    """Get simple 1 Layer CNN, with K number of densenets -> softmax -> CCE."""
    graph = nx.MultiDiGraph()
    classes = np.arange(k)

    # add nodes to graph with names (for easy human referencing),
    # and objects for what those nodes are
    graph.add_node("x", group=0, node=Rotate())

    data_shape = (28, 28)
    cnn_weights_shape = (6, 6)
    stride = [4, 4]
    windows = CC().windex(data_shape, cnn_weights_shape, stride)

    # CONSTRUCT CNN
    # with intermediary decrypted sum to save on some complexity later
    graph.add_node("CC-products", group=1,
                   node=CC(weights=cnn_weights_shape, stride=stride, bias=0))
    graph.add_edge("x", "CC-products")
    graph.add_node("CC-dequeue", group=6, node=Dequeue(length=len(windows)))
    graph.add_edge("CC-products", "CC-dequeue")
    graph.add_node("CC-enqueue", group=6, node=Enqueue(length=len(windows)))
    for i in range(len(windows)):
        graph.add_node("Rotate-{}".format(i), group=5,
                       node=Rotate(axis=1, flatten=True))
        graph.add_edge("CC-dequeue", "Rotate-{}".format(i))
        graph.add_node("CC-sop-{}".format(i), group=1, node=Sum())
        graph.add_edge("Rotate-{}".format(i), "CC-sop-{}".format(i),
                       weight=Sum().cost)
        graph.add_edge("CC-sop-{}".format(i), "CC-enqueue")
    graph.add_node("CNN-RELU", group=1, node=RELU(q=10))
    graph.add_edge("CC-enqueue", "CNN-RELU")
    graph.add_node("CNN-distribute", group=6, node=Distributor())
    graph.add_edge("CNN-RELU", "CNN-distribute")
    # graph.add_edge("CNN-enqueue", "CNN-activation", weight=RELU().cost)

    # CONSTRUCT DENSE FOR EACH CLASS
    # we want to get the network to regress some prediction one for each class
    graph.add_node("Dense-enqueue", group=6, node=Enqueue(length=k))
    for i in classes:
        graph.add_node("Dense-{}".format(i), group=2,
                       node=ANN(weights=(len(windows),)))
        graph.add_edge("CNN-distribute", "Dense-{}".format(i))
        graph.add_node("Dense-RELU-{}".format(i), group=2, node=RELU(q=10))
        graph.add_edge("Dense-{}".format(i), "Dense-RELU-{}".format(i))
        graph.add_edge("Dense-RELU-{}".format(i), "Dense-enqueue")
    graph.add_node("Decrypt", group=5, node=Decrypt())
    graph.add_edge("Dense-enqueue", "Decrypt")

    # CONSTRUCT SELECTOR TO SELECT COMPUTATIONAL CIRCUITS
    # we need to be able to select different computational circuits depending
    # on the receptor so we can say infer, or train, and so we can also
    # selectiveley backpropagate through only one circuit and ignore the other
    graph.add_node("Selector", group=6, node=Selector(backward=[1, 0]))
    graph.add_edge("Decrypt", "Selector")

    # CONSTRUCT CLASSIFIER
    # we want to turn the dense outputs into classification probabilities
    # using softmax and how wrong / right we are using Categorical
    # Cross-Entropy(CCE) as our loss function
    graph.add_node("Softmax", group=3, node=Softmax())
    graph.add_edge("Selector", "Softmax")
    # graph.add_edge("Dense-enqueue", "Softmax", weight=Softmax().cost)
    graph.add_node("Loss-CCE", group=3, node=CCE())
    graph.add_edge("Softmax", "Loss-CCE", weight=3)
    graph.add_node("One-hot-encoder", group=0,
                   node=OneHotEncode(length=len(classes)))
    graph.add_edge("One-hot-encoder", "Loss-CCE", weight=0)
    graph.add_node("y", group=0, node=IO())
    graph.add_edge("y", "One-hot-encoder", weight=OneHotEncode().cost)

    graph.add_node("Argmax", group=4, node=Argmax())
    graph.add_edge("Selector", "Argmax")
    # graph.add_edge("Dense-enqueue", "Argmax", weight=Argmax().cost)
    graph.add_node("One-hot-decoder", group=4, node=OneHotDecode())
    graph.add_edge("Argmax", "One-hot-decoder", weight=OneHotDecode().cost)
    graph.add_node("y_hat", group=4, node=IO())
    graph.add_edge("One-hot-decoder", "y_hat", weight=0)
    return graph


def basic():
    """Get a super basic graph for purposes of testing components.

    .. note::

        This is not a useful graph outside of unit-tests and validation,
        as it does not represent any form of useful network for solving any
        particular problem.
    """
    graph = nx.MultiDiGraph()
    graph.add_node("x_0", group=0, node=Rotate())
    graph.add_node("x_1", group=0, node=Rotate())

    graph.add_node("c_0", group=1, node=ANN(weights=(1,)))
    graph.add_edge("x_0", "c_0")

    graph.add_node("c_1", group=1, node=ANN(weights=(2,)))
    graph.add_edge("c_0", "c_1")
    graph.add_edge("x_1", "c_1")

    graph.add_node("r_0", group=2, node=Rotate())
    graph.add_edge("c_1", "r_0")

    graph.add_node("y_0", group=0, node=Rotate())
    graph.add_node("c_2", group=1, node=ANN(weights=(2,)))
    graph.add_edge("r_0", "c_2")
    graph.add_edge("y_0", "c_2")

    graph.add_node("d_0", group=3, node=Decrypt())
    graph.add_edge("c_2", "d_0")

    graph.add_node("c_3", group=1, node=ANN(weights=(2,)))
    graph.add_edge("c_2", "c_3")

    graph.add_node("d_1", group=3, node=Decrypt())
    graph.add_edge("c_3", "d_1")
    return graph


def orbweaver():
    """Get prefabricated orbweaver graph."""
    return cnn_classifier(10)


def sphira():
    """Get prefabricated sphira graph."""
    return cnn_classifier(10)


def milky(**kwargs):
    """Get prefabricated milk graph."""
    g = cnn_regressor(**kwargs)
    # sideloading our additional contextual nodes
    g.add_node("Context", node=Rotate())
    g.add_node("Context-enqueue", group=6, node=Enqueue())
    g.add_edge("Context", "Context-enqueue")
    # removing existing edge which we want to interject
    g.remove_edge("CNN-acti", "Dense")
    # sewing back together the graph
    g.add_edge("Context-enqueue", "Dense")
    g.add_edge("CNN-acti", "Context-enqueue")
    return g
