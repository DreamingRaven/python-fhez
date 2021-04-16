#!/usr/bin/env python3

# @Author: GeorgeRaven <archer>
# @Date:   2021-04-15T14:24:29+01:00
# @Last modified by:   archer
# @Last modified time: 2021-04-16T12:48:12+01:00
# @License: please see LICENSE file in project root

import unittest
import logging as logger
from fhe.nn.graph.node import Node


class DLL(object):
    """Doubly Linked List."""

    def __init__(self):
        """Initialise a new doubly linked list."""
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
        c = self.__dict__.get("_count")
        if c is not None:
            return c
        else:
            self.count = 0
            return self.count

    @count.setter
    def count(self, count: int):
        self._count = count

    @property
    def size(self):
        """Get size of dll."""
        return self.count

    def append(self, data):
        """Append given data to end of dll as a new node."""
        if self.head is None:
            self.head = Node(data)
            self.tail = self.head
        else:
            self.tail.next = Node(data)
            self.tail.next.previous = self.tail
            self.tail = self.tail.next
        self.count += 1

    def _index_check(self, index):
        if not isinstance(index, int):
            raise TypeError("Index type: {}, expected int".format(type(index)))
        if (self.count < index) | (index < 0):
            raise ValueError("Index: {} out of range: {}, dll len.".format(
                index, self.count
            ))

    def insert(self, data, index: int):
        """Insert a new node between index and index-1 nodes."""
        self._index_check(index)

        if index == self.count:
            return self.append(data)
        elif index == 0:
            self.head.previous = Node(data)
            self.head.previous.next = self.head
            self.head = self.head.previous
        else:
            node = self.head
            # traverse to indexed node as current "node"
            for _ in range(index):
                node = node.next
            # link new node to previous node
            node.previous.next = Node(data)
            node.previous.next.previous = node.previous
            # link new node to indexed node
            node.previous.next.next = node
            node.previous = node.previous.next
        self.count += 1

    def remove(self, index):
        """Remove indexed node and attach index-1 and index+1 nodes."""
        self._index_check(index)
        # handling the special off by one error for removal
        if self.count == index:
            raise ValueError("{}: {}, as zero indexed/ range 0-{}".format(
                "Off by one cannot remove", index, self.count-1))

        if index == 0:
            self.head = self.head.next
            self.head.previous = None
        elif index == (self.count - 1):  # 0 indexed last element
            self.tail = self.tail.previous
            self.tail.next = None
        else:
            node = self.head
            # traverse to indexed node as current "node"
            for i in range(index):
                node = node.next
            # sew the two nodes either side of us together
            node.previous.next, node.next.previous = node.next, node.previous
        self.count -= 1

    def search(self, data):
        """Search dll for the index of the first node with matching data."""
        node = self.head
        for i in range(self.count):
            if node.data == data:
                return i
            node = node.next

    def __len__(self):
        """Get length of dll."""
        return self.count


class DLL_tests(unittest.TestCase):
    def setUp(self):
        self.dll = DLL()

    def tearDown(self):
        del self.dll


if __name__ == "__main__":
    logger.basicConfig(  # filename="{}.log".format(__file__),
        level=logger.INFO,
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S")
    # run all the unit-tests
    print("now testing:", __file__, "...")
    unittest.main()
