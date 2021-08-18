"""Generic decryptor as computational graph node."""
# @Author: George Onoufriou <archer>
# @Date:   2021-08-18T15:05:03+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-18T15:11:34+01:00

import numpy as np
from fhez.nn.graph.node import Node


class Decrypt(Node):
    """Generic decryptor of inputs."""

    @property
    def cost(self):
        """Return no depth/ cost/ **0** of decryption."""
        return 0

    def forward(self, x):
        """Decrypt cyphertext using numpy ufunc API."""
        return np.array(x)

    def backward(self, gradient):
        """Pass gradients back unmodified."""
        return gradient

    def update(self):
        """Do nothing as decryption has no deep-learning parameterisation."""
        return NotImplemented

    def updates(self):
        """Do nothing as decryption has no deep-learning parameterisation."""
        return NotImplemented
