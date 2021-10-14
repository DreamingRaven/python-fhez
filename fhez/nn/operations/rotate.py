"""Generic cyphertext key rotation as abstract node."""
# @Author: George Onoufriou <archer>
# @Date:   2021-08-18T15:05:03+01:00
# @Last modified by:   archer
# @Last modified time: 2021-10-14T09:54:40+01:00

import logging
import numpy as np
from fhez.nn.graph.node import Node


class Rotate(Node):
    """Generic cyphertext key rotation abstraction."""

    def __init__(self, axis=None, encryptor=None, provider=None, **kwargs):
        """Configure provider and encryption parameters.

        Given a provider like FHEz-ReSeal, and an arbitrary number of keyword
        arguments. Setup encryptor to be (re-)used for continued encryption.
        This will simply rotate the keys to new fresh keys, based on an axis
        given.
        E.G it will decrypt, then re-encrypt the given axis, and return a
        structured list of the prior axies.
        """
        if provider is not None:
            self.provider = provider
        if encryptor is not None:
            self.encryptor = encryptor
        self.parameters = kwargs
        self.axis = axis if axis is not None else 0

    @property
    def provider(self):
        """Get encryption provider to be parameterised for encryption."""
        return self.__dict__.get("_provider")

    @provider.setter
    def provider(self, provider):
        self._provider = provider

    @property
    def axis(self):
        """Get axis of key rotation."""
        return self.__dict__.get("_axis")

    @axis.setter
    def axis(self, axis):
        valid = (0, 1)
        assert axis in valid, "{} axis expected to be in {}, got {}".format(
            self.__class__.__name__, valid, axis)
        assert isinstance(axis, int), "axis must be an integer got {}".format(
            type(axis))
        self._axis = axis

    @property
    def encryptor(self):
        """Encryption parameterised object."""
        return self.__dict__.get("_encryptor")

    @encryptor.setter
    def encryptor(self, encryptor):
        """Set encryptor for key rotation.

        Encryptors must be callable and preferably serialisable objects,
        that when called with some input numpy array encrypt said array.

        Encryptor is expected to be a callable as we want to be able to share
        encryptors between multiple objects. We want them to be called, and
        generate new callable objects that share the same parameters, and
        keys as themselves.
        """
        assert callable(encryptor), "{}.encryptor got non callable, {}".format(
            self.__class__.__name__, type(encryptor))
        self._encryptor = encryptor

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
        """Rotate keys using encryption provider on desired axis.

        This function has 3 modes:

        - Encryptor: given some provider or encryptor will encrypt x and return
          the cyphertext or list of cyphertexts (if axis is not 0)
        - Decryptor: given neither provider nor encryptor will just turn x
          into a numpy array, which for our numpyapi implementation means
          turning cyphertexts into plaintexts and plaintexts stay plaintexts
        - Rotator: given a cyphertext-x and a provider or encryptor will
          decrypt x and re-encrypt using the new provider on the desired axis.
        """
        t = np.array(x)  # ensure is numpy array, cyphertexts will decrypt here
        if self.provider is None and self.encryptor is None:
            # in the case where no encryption provider has been specified
            # assume we are to just leave it as a plaintext
            return t
        elif self.provider is not None and self.encryptor is None:
            # if no encryptor provided then use provider to generate one
            try:
                self.encryptor = self.provider(**self.parameters)
            except TypeError:
                # sometimes a provider expects a number to encrypt, if so call
                # again this time with some dummy number
                self.encryptor = self.provider(np.array([1]),
                                               **self.parameters)

        if self.axis == 0:
            return self.encryptor(t)
        elif self.axis == 1:
            accumulator = []
            for i in range(len(t)):
                accumulator.append(self.encryptor(t[i]))
            return accumulator
        else:
            raise ValueError("{}.forward() got unsupported axis {}, {}".format(
                self.__class__.__name__, self.axis, type(self.axis)))
        # cyphertext = self.provider(x, **self.parameters)
        # return cyphertext

    def backward(self, gradient):
        """Pass gradients back unmodified."""
        return gradient

    def update(self):
        """Do nothing as encryption has no deep-learning parameterisation."""
        return NotImplemented

    def updates(self):
        """Do nothing as encryption has no deep-learning parameterisation."""
        return NotImplemented
