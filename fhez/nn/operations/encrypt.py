"""Generic encryptor as computational graph node."""
# @Author: George Onoufriou <archer>
# @Date:   2021-08-18T15:05:03+01:00
# @Last modified by:   archer
# @Last modified time: 2021-09-14T13:48:07+01:00

import logging
from fhez.nn.graph.node import Node


class Encrypt(Node):
    """Generic encryptor of inputs."""

    def __init__(self, provider=None, **kwargs):
        """Configure provider and encryption parameters.

        Given a provider like FHEz-ReSeal, and an arbitrary number of keyword
        arguments. Setup encryptor to be (re-)used for continued encryption.
        """
        if provider is not None:
            self.provider = provider
        self.parameters = kwargs

    @property
    def provider(self):
        """Get encryption provider to be parameterised for encryption."""
        return self.__dict__.get("_provider")

    @provider.setter
    def provider(self, provider):
        self._provider = provider

    # @property
    # def encryptor(self):
    #     """Get encryptor for specific encryption."""
    #     return self.__dict__.get("_encryptor")
    #
    # @encryptor.setter
    # def encryptor(self, encryptor):
    #     self._encryptor = encryptor

    @property
    def parameters(self):
        """Get parameters to for the encryption provider."""
        return self.__dict__.get("_parameters")

    @parameters.setter
    def parameters(self, parameters):
        self._parameters = parameters

    @property
    def cost(self):
        """Return no depth/ cost/ **0** of encryption."""
        return 0

    def forward(self, x):
        """Encrypt cyphertext using configured FHE provider."""
        if self.provider is None:
            # in the case where no encryption provider has been specified
            # assume we arent to encrypt, but warn!
            logging.warning("{} was not given any encryption provider, \
            assuming should be plaintext, skipping encryption.".format(
                self.__class__.__name__))
            return x
        cyphertext = self.provider(x, **self.parameters)
        return cyphertext

    def backward(self, gradient):
        """Pass gradients back unmodified."""
        return gradient

    def update(self):
        """Do nothing as encryption has no deep-learning parameterisation."""
        return NotImplemented

    def updates(self):
        """Do nothing as encryption has no deep-learning parameterisation."""
        return NotImplemented
