"""Selection of various prefabricated neural networks and generators."""
# @Author: George Onoufriou <archer>
# @Date:   2021-08-23T17:22:55+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-24T14:26:33+01:00

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

from fhez.nn.operations.encrypt import Encrypt
from fhez.nn.operations.decrypt import Decrypt

from fhez.nn.operations.enqueue import Enqueue
from fhez.nn.operations.dequeue import Dequeue

from fhez.nn.operations.one_hot_encode import OneHotEncode
from fhez.nn.operations.one_hot_decode import OneHotDecode


def cnn_classifier(k):
    """Get simple 1 Layer CNN, with K number of densenets -> softmax -> CCE."""
    graph = nx.MultiDiGraph()
    classes = np.arange(k)

    # add nodes to graph with names (for easy human referencing), and objects for what those nodes are
    graph.add_node("x", group=0, node=IO())

    data_shape = (28, 28)
    cnn_weights_shape = (6, 6)
    stride = [4, 4]
    windows = CC().windex(data_shape, cnn_weights_shape, stride)
    print(len(windows))

    # CONSTRUCT CNN
    # with intermediary decrypted sum to save on some complexity later
    graph.add_node("CNN-products", group=1,
                   node=CC(weights=cnn_weights_shape, stride=stride, bias=0))
    graph.add_edge("x", "CNN-products", weight=CC().cost)
    graph.add_node("CNN-dequeue", group=1, node=Dequeue())
    graph.add_edge("CNN-products", "CNN-dequeue", weight=Dequeue().cost)
    # graph.add_node("CNN-enqueue", group=1, node=Enqueue())
    for i in range(len(windows)):
        graph.add_node("CNN-sop-{}".format(i), group=1, node=Sum())
        graph.add_edge("CNN-dequeue", "CNN-sop-{}".format(i), weight=Sum().cost)
        graph.add_edge("CNN-sop-{}".format(i), "CNN-acti", weight=Enqueue().cost)
    graph.add_node("CNN-acti", group=1, node=RELU())
    # graph.add_edge("CNN-enqueue", "CNN-activation", weight=RELU().cost)

    # CONSTRUCT DENSE FOR EACH CLASS
    # we want to get the network to regress some prediction one for each class
    # graph.add_node("Dense-enqueue", group=2, node=Enqueue())
    for i in classes:
        graph.add_node("Dense-{}".format(i), group=2,
                       node=ANN(weights=(len(windows),)))
        graph.add_edge("CNN-acti", "Dense-{}".format(i), weight=ANN().cost)
        graph.add_node("Dense-activation-{}".format(i), group=2, node=RELU())
        graph.add_edge("Dense-{}".format(i), "Dense-activation-{}".format(i),
                       weight=RELU().cost)
        graph.add_edge("Dense-activation-{}".format(i), "Softmax",
                       weight=Enqueue().cost)
        graph.add_edge("Dense-activation-{}".format(i), "Argmax",
                       weight=Enqueue().cost)
    #     graph.add_edge("Dense-activation-{}".format(i), "Dense-enqueue", weight=Enqueue().cost)

    # CONSTRUCT CLASSIFIER
    # we want to turn the dense outputs into classification probabilities using softmax and how wrong/ right we are using Categorical Cross-Entropy (CCE) as our loss function
    graph.add_node("Softmax", group=3, node=Softmax())
    # graph.add_edge("Dense-enqueue", "Softmax", weight=Softmax().cost)
    graph.add_node("Loss-CCE", group=3, node=CCE())
    graph.add_edge("Softmax", "Loss-CCE", weight=3)
    graph.add_node("One-hot-encoder", group=0,
                   node=OneHotEncode(length=len(classes)))
    graph.add_edge("One-hot-encoder", "Loss-CCE", weight=0)
    graph.add_node("y", group=0, node=IO())
    graph.add_edge("y", "One-hot-encoder", weight=OneHotEncode().cost)

    graph.add_node("Argmax", group=4, node=Argmax())
    # graph.add_edge("Dense-enqueue", "Argmax", weight=Argmax().cost)
    graph.add_node("One-hot-decoder", group=4, node=OneHotDecode())
    graph.add_edge("Argmax", "One-hot-decoder", weight=OneHotDecode().cost)
    graph.add_node("y_hat", group=4, node=IO())
    graph.add_edge("One-hot-decoder", "y_hat", weight=0)
    return graph
