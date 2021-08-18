"""Enqueue as computational node.

Enqueueing is the process of taking two or more arrays and stacking them
together using some meta container that encapsulates both arrays.

While the queue is still being enqueued this node will block.
Once the queue has reached the desired length, it will return the queue as a
list. This will map gradients.

Akin to: `numpy stacking
 <https://numpy.org/doc/stable/reference/generated/numpy.stack.html>`_
"""
# @Author: George Onoufriou <archer>
# @Date:   2021-08-17T13:01:54+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-18T01:39:38+01:00

from collections import deque
import numpy as np
from fhez.nn.graph.node import Node


class Enqueue(Node):
    """Stack multiple arrays together until a given shape is achieved."""

    def __init__(self, length=None):
        """Configure enqueue initialised state."""
        if length is not None:
            self.length = length

    @property
    def length(self):
        """Get the desired length of the final enqueue."""
        return self.__dict__.get("_desired_length")

    @length.setter
    def length(self, length: int):
        """Set the desired/ target length of the queue."""
        self._desired_length = length

    @property
    def queue(self):
        """Get the current queue."""
        if self.__dict__.get("_enqueue") is None:
            self._queue = deque()
        return self._queue

    @queue.setter
    def queue(self, queue):
        self._queue = queue

    @property
    def cost(self):
        """Get **0** cost for enqueueing arrays."""
        return 0

    def forward(self, x):
        """Accumulate inputs into a single queue, then return when full."""
        self.queue.append(x)
        if len(self.queue) == self.length:
            out = list(self.queue)
            self.queue = None
            print("DO I MAKE IT?")
            return out
        return None

    def backward(self, gradient):
        pass

    def update(self):
        """Update nothing as enqueueing is not parameterisable."""
        return NotImplemented

    def updates(self):
        """Update nothing as enqueueing is not parameterisable."""
        return NotImplemented
