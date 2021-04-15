#!/usr/bin/env python3

# @Author: GeorgeRaven <archer>
# @Date:   2021-04-15T14:24:29+01:00
# @Last modified by:   archer
# @Last modified time: 2021-04-15T14:54:48+01:00
# @License: please see LICENSE file in project root


class Node(object):
    """Computational Graph Node.

    CGs are doubly linked lists for forward and backward prop. Thus this class
    represents the nodes of this computational graph.
    """

    def __init__(self, data=None, next=None, previous=None):
        """Initialise a computational graph node, with values."""
        self.data = data
        self.next = next
        self.previous = previous

    @property
    def next(self):
        """Get the next node in the doubly linked list."""
        return self.__dict__.get("_next")

    @next.setter
    def next(self, node):
        self._next = node

    @property
    def previous(self):
        """Get the previous node in the doubly linked list."""
        return self.__dict__.get("_previous")

    @previous.setter
    def previous(self, node):
        self._previous = node

    @property
    def data(self):
        """Get the this current nodes data."""
        return self.__dict__.get("_data")

    @data.setter
    def data(self, data):
        self._data = data
