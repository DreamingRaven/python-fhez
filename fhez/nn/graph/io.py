# @Author: George Onoufriou <archer>
# @Date:   2021-07-15T15:50:42+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-23T10:12:36+01:00

import marshmallow as mar
from fhez.nn.graph.node import Node
from fhez.nn.graph.serialise import Serialise


class IO(Node, Serialise):
    """An input output node that is primarily used to link and join nodes."""

    @property
    def schema(self):
        """Get Marshmallow schema representation of this class."""
        schema_dict = {
        }
        return mar.Schema.from_dict(schema_dict)

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

    @property
    def cost(self):
        """Get no-cost of this node."""
        return 0
