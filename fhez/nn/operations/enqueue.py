"""Enqueue as computational node.

Enqueueing is the process of taking two or more arrays and stacking them
together using some meta container that encapsulates both arrays, and
enqueues->dequeue in first in first out manner (FIFO).

While the queue is still being enqueued this node will return nothing.
Once the queue has reached the desired length, it will return the queue as a
list. This will map gradients again in FIFO manner using a dequeue.

Enqueue as a node, forward is enqueue, backward is dequeue, the exact inverse
of the Dequeue node as we name it for the forward pass.

See: `Comp-sci queues <https://computersciencewiki.org/index.php/Queue>`_
"""
# @Author: George Onoufriou <archer>
# @Date:   2021-08-17T13:01:54+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-18T13:15:23+01:00

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
        if self.__dict__.get("_queue") is None:
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
            return out
        return None

    def backward(self, gradient):
        """Distribute gradient to respective inputs in order via yield.

        Effectiveley backward is a dequeue but for gradients.

        .. warning::

            This **YIELDS** gradients unlike most nodes, requiring special
            logic by a network traverser, only getting one input but
            results in many outputs.
        """
        assert len(gradient) == self.length
        queue = deque(gradient)
        # I dont want to traverse queue as iterator so will use slightly faster
        # length of queue instead so we can rely on queues heavy internal
        # optimisation.
        for _ in range(len(queue)):
            yield queue.popleft()  # yield dequeued gradient FIFO

    def update(self):
        """Update nothing as enqueueing is not parameterisable."""
        return NotImplemented

    def updates(self):
        """Update nothing as enqueueing is not parameterisable."""
        return NotImplemented
