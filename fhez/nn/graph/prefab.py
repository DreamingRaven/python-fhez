"""Selection of various prefabricated neural networks and generators."""
# @Author: George Onoufriou <archer>
# @Date:   2021-08-23T17:22:55+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-23T17:32:01+01:00

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
    classes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    graph.add_node("x", group=0, node=IO())

    # CONSTRUCT CNN
    graph.add_node("CNN-products", group=1,
                   node=CC(weights=(1, 6, 6), stride=[1, 4, 4], bias=0))
    graph.add_edge("x", "CNN-products", weight=CC().cost)
    graph.add_node("CNN-dequeue", group=1, node=Dequeue)
    graph.add_edge("CNN-products", "CNN-dequeue", weight=Dequeue().cost)
    graph.add_node("CNN-sum-of-products", group=1, node=Sum())
    graph.add_edge("CNN-dequeue", "CNN-sum-of-products", weight=Sum().cost)
    graph.add_node("CNN-enqueue", group=1, node=Enqueue())
    graph.add_edge("CNN-sum-of-products", "CNN-enqueue", weight=Enqueue().cost)
    graph.add_node("CNN-activation", group=1, node=RELU)
    graph.add_edge("CNN-enqueue", "CNN-activation", weight=RELU().cost)

    # CONSTRUCT DENSE FOR EACH CLASS
    graph.add_node("Dense-enqueue", group=2)
    for i in classes:
        graph.add_node("Dense-{}".format(i), group=2,
                       node=ANN())
        graph.add_edge("CNN-activation", "Dense-{}".format(i),
                       weight=ANN().cost)
        graph.add_node("Dense-activation-{}".format(i), group=2,
                       node=RELU())
        graph.add_edge("Dense-{}".format(i),
                       "Dense-activation-{}".format(i), weight=RELU().cost)
        graph.add_edge("Dense-activation-{}".format(i), "Dense-enqueue",
                       weight=Enqueue().cost)

    # CONSTRUCT CLASSIFIER
    graph.add_node("Softmax", group=3, node=Softmax())
    graph.add_edge("Dense-enqueue", "Softmax", weight=Softmax().cost)
    graph.add_node("Loss-CCE", group=3, node=CCE())
    graph.add_edge("Softmax", "Loss-CCE", weight=3)
    graph.add_node("One-hot-encoder", group=0, node=OneHotEncode())
    graph.add_edge("One-hot-encoder", "Loss-CCE", weight=0)
    graph.add_node("y", group=0, node=IO())
    graph.add_edge("y", "One-hot-encoder", weight=OneHotEncode().cost)

    graph.add_node("Argmax", group=4, node=Argmax())
    graph.add_edge("Dense-enqueue", "Argmax", weight=Argmax().cost)
    graph.add_node("One-hot-decoder", group=4, node=OneHotDecode())
    graph.add_edge("Argmax", "One-hot-decoder", weight=OneHotDecode().cost)
    graph.add_node("y_hat", group=4, node=IO())
    graph.add_edge("One-hot-decoder", "y_hat", weight=0)
