# @Author: George Onoufriou <archer>
# @Date:   2021-08-23T10:02:02+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-23T10:04:11+01:00
import unittest
import pandas as pd
import numpy as np

import networkx as nx
from fhez.nn.graph.io import IO
from fhez.nn.operations.cc import CC  # Cross Correlation
from fhez.nn.operations.sum import Sum
from fhez.nn.activation.relu import RELU  # Rectified Linear Unit (approximation)
from fhez.nn.layer.ann import ANN  # Dense/ Artificial Neural Network
from fhez.nn.activation.softmax import Softmax
from fhez.nn.loss.cce import CCE  # categorical cross entropy

from fhez.nn.operations.encrypt import Encrypt
from fhez.nn.operations.decrypt import Decrypt
from fhez.nn.operations.enqueue import Enqueue
from fhez.nn.operations.dequeue import Dequeue


class NNTest(unittest.TestCase):
    """Test against MNIST."""

    def graph(self):
        graph = nx.MultiDiGraph()
        classes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        graph.add_node("x", group=0)

        # CONSTRUCT CNN
        graph.add_node("CNN-products", group=1)
        graph.add_edge("x", "CNN-products", weight=CC().cost)
        graph.add_node("CNN-dequeue", group=1)
        graph.add_edge("CNN-products", "CNN-dequeue", weight=Dequeue().cost)
        graph.add_node("CNN-sum-of-products", group=1)
        graph.add_edge("CNN-dequeue", "CNN-sum-of-products", weight=Sum().cost)
        graph.add_node("CNN-enqueue", group=1)
        graph.add_edge("CNN-sum-of-products", "CNN-enqueue", weight=Enqueue().cost)
        graph.add_node("CNN-activation", group=1)
        graph.add_edge("CNN-enqueue", "CNN-activation", weight=RELU().cost)

        # CONSTRUCT DENSE FOR EACH CLASS
        graph.add_node("Dense-enqueue", group=2)
        for i in classes:
            graph.add_node("Dense-{}".format(i), group=2)
            graph.add_edge("CNN-activation", "Dense-{}".format(i), weight=ANN().cost)
            graph.add_node("Dense-activation-{}".format(i), group=2)
            graph.add_edge("Dense-{}".format(i),
                           "Dense-activation-{}".format(i), weight=RELU().cost)
            graph.add_edge("Dense-activation-{}".format(i), "Dense-enqueue", weight=Enqueue().cost)

        # CONSTRUCT CLASSIFIER
        graph.add_node("Softmax", group=3)
        graph.add_edge("Dense-enqueue", "Softmax", weight=Softmax().cost)
        graph.add_node("Loss-CCE", group=3)
        graph.add_edge("Softmax", "Loss-CCE", weight=3)
        graph.add_node("One-hot-encoder", group=0)
        graph.add_edge("One-hot-encoder", "Loss-CCE", weight=0)
        graph.add_node("y", group=0)
        graph.add_edge("y", "One-hot-encoder", weight=0)

        graph.add_node("Argmax", group=4)
        graph.add_edge("Dense-enqueue", "Argmax", weight=2)
        graph.add_node("One-hot-decoder", group=4)
        graph.add_edge("Argmax", "One-hot-decoder", weight=0)
        graph.add_node("y_hat", group=4)
        graph.add_edge("One-hot-decoder", "y_hat", weight=0)
