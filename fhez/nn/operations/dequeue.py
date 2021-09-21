"""Dequeue as computational node.

Dequeue as a node, forward is dequeue, backward is dequeue, the exact inverse
of the dequeue node as we name it for the forward pass.

See: `Comp-sci queues <https://computersciencewiki.org/index.php/Queue>`_
"""
# @Author: George Onoufriou <archer>
# @Date:   2021-08-17T13:01:54+01:00
# @Last modified by:   archer
# @Last modified time: 2021-09-21T16:31:51+01:00

from collections import deque
import numpy as np
from fhez.nn.graph.node import Node


class Dequeue(Node):
    """Stack multiple arrays together until a given shape is achieved."""

    def __init__(self, length=None):
        """Configure dequeue initialised state."""
        if length is not None:
            self.length = length

    @property
    def length(self):
        """Get the desired length of the final dequeue."""
        if self.__dict__.get("_desired_length") is not None:
            return self.__dict__.get("_desired_length")
        else:
            raise ValueError(
                "You FOOL you have not provided me `{}`, with a length".format(
                    self.__class__.__name__))

    @length.setter
    def length(self, length: int):
        """Set the desired/ target length of the queue."""
        self._desired_length = length

    @property
    def queue(self):
        """Get the current queue."""
        if self.__dict__.get("_queue") is None:
            self._queue = deque()
        return self._queue

    @queue.setter
    def queue(self, queue):
        self._queue = queue

    @property
    def cost(self):
        """Get **0** cost for dequeueing arrays."""
        return 0

    def forward(self, x):
        """Distribute input to respective outputs in order via yield.

        Effectiveley backward is a dequeue but for gradients.

        .. warning::

            This **YIELDS** gradients unlike most nodes, requiring special
            logic by a network traverser, only getting one input but
            results in many outputs.
        """
        # assert self.length is not None, "Missing length of dequeue"
        # assert len(x) == self.length
        queue = deque(x)
        # I dont want to traverse queue as iterator so will use slightly faster
        # length of queue instead so we can rely on queues heavy internal
        # optimisation.
        for _ in range(len(queue)):
            yield queue.popleft()  # yield dequeued gradient FIFO

    def backward(self, gradient: np.ndarray):
        """Accumulate inputs into a single queue, then return when full."""
        self.queue.append(gradient)
        if len(self.queue) == self.length:
            out = list(self.queue)
            self.queue = None
            return out
        return None

    def update(self):
        """Update nothing as dequeueing is not parameterisable."""
        return NotImplemented

    def updates(self):
        """Update nothing as dequeueing is not parameterisable."""
        return NotImplemented
