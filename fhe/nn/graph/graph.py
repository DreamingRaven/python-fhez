#!/usr/bin/env python3

# @Author: GeorgeRaven <archer>
# @Date:   2021-04-15T14:24:29+01:00
# @Last modified by:   archer
# @Last modified time: 2021-04-15T15:30:16+01:00
# @License: please see LICENSE file in project root


class Graph(object):
    """Graph representing neural network computations as a network of nodes."""

    def __init__(self):
        """Initialise a new network/ graph."""
        pass

    @property
    def head(self):
        """Get the head/ first node."""
        return self.__dict__.get("_head")

    @head.setter
    def head(self, node):
        self._head = node

    @property
    def tail(self):
        """Get the tail/last node."""
        return self.__dict__.get("_tail")

    @tail.setter
    def tail(self, node):
        self._tail = node

    @property
    def count(self):
        """Count the nodes in the doubly linked list."""
        return self.__dict__.get("_count")

    @count.setter
    def count(self, node):
        self._count = node
