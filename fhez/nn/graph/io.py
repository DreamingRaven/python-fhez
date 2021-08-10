# @Author: George Onoufriou <archer>
# @Date:   2021-07-15T15:50:42+01:00
# @Last modified by:   archer
# @Last modified time: 2021-07-15T15:50:56+01:00

from fhez.nn.graph.node import Node


class IO(Node):
    """An input output node that is primarily used to link and join nodes."""

    def forward(self, x):
        """Pass input directly to output."""
        return x

    def backward(self, gradient):
        """Pass gradient directly to output."""
        return gradient

    def update(self):
        """Do nothing."""

    def updates(self):
        """Do nothing."""
