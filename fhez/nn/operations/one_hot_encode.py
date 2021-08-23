# @Author: George Onoufriou <archer>
# @Date:   2021-08-23T10:27:37+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-23T11:59:52+01:00

import numpy as np
import marshmallow as mar
from fhez.nn.graph.serialise import Serialise
from fhez.nn.graph.node import Node


class OneHotEncode(Node, Serialise):
    """Encode value in one-hot encoded sparse array."""

    def __init__(self, length=None):
        """Configure the encoder initially."""
        self.length = length

    @property
    def schema(self):
        """Get marshmallow serialisation schema."""
        schema_dict = {
            "_length": mar.fields.Int(),
        }
        return mar.Schema.from_dict(schema_dict)

    @property
    def length(self):
        """Get length of sparse matrix to be generated."""
        return self.__dict__.get("_length")

    @length.setter
    def length(self, length):
        self._length = length

    @property
    def cost(self):
        """Get non-cost of this node."""
        return 0

    def forward(self, x: np.ndarray):
        """Encode input to sparse matrix."""
        if isinstance(x, int):
            x = np.array(x)
        self.inputs.append(x)
        targets = x.reshape(-1)
        one_hot = np.eye(self.length)[targets]
        # while this does work for multidims this is the forward func
        # which should only output for one dim so selecting 0th
        return one_hot[0]

    def backward(self, gradient: np.ndarray):
        """Map only the gradient of the encoded backward."""
        x = self.inputs.pop()
        return gradient[x]

    def update(self):
        """Do nothing as nothing to update."""
        return NotImplemented

    def updates(self):
        """Do nothing as nothing to update."""
        return NotImplemented
