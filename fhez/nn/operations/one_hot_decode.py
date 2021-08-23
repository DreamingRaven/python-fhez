# @Author: George Onoufriou <archer>
# @Date:   2021-08-23T10:27:46+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-23T12:08:31+01:00

import numpy as np
import marshmallow as mar
from fhez.nn.graph.serialise import Serialise
from fhez.nn.graph.node import Node


class OneHotDecode(Node, Serialise):
    """Encode value in one-hot encoded sparse array."""

    @property
    def schema(self):
        """Get marshmallow serialisation schema."""
        schema_dict = {
        }
        return mar.Schema.from_dict(schema_dict)

    @property
    def cost(self):
        """Get non-cost of this node."""
        return 0

    def forward(self, x: np.ndarray):
        """Encode input to sparse matrix."""
        self.inputs.append(x)
        clss = x.argmax()
        return clss

    def backward(self, gradient: np.ndarray):
        """Map only the gradient of the encoded backward."""
        x = self.inputs.pop()
        return x * gradient

    def update(self):
        """Do nothing as nothing to update."""
        return NotImplemented

    def updates(self):
        """Do nothing as nothing to update."""
        return NotImplemented
