# @Author: GeorgeRaven <archer>
# @Date:   2021-02-11T11:31:48+00:00
# @Last modified by:   archer
# @Last modified time: 2021-02-11T11:32:09+00:00
# @License: please see LICENSE file in project root

class ReCache():
    """Core caching object for ReSeal.

    This cache can be enabled (default) to improve/ minimise the need to
    regenerate the transient properties of ReSeal. That is to say ReSeal
    only keeps/ stores the minimum required attributes like the keys,
    the cyphertext and the parameters, as everything else is derived from those
    at the point of need. Thus this class caches the generated intermediaries
    so they can be re-used rather than commiting compute power every time they
    are called. E.G seal.Encryptor and seal.Decryptor are examples of cached
    objects."""

    def __init__(self, enable=None):
        """Object caching.

        If enabled will cache all ReSeal objects not already stored,
        to avoid having to regenrate them."""
        self.enabled = enable if enable is not None else True

    @property
    def context(self):
        if self.__dict__.get("_context") and self.enabled:
            return self._context
        return None

    @context.setter
    def context(self, context):
        self._context = context

    @property
    def keygen(self):
        if self.__dict__.get("_keygen") and self.enabled:
            return self._keygen
        return None

    @keygen.setter
    def keygen(self, keygen):
        self._keygen = keygen

    @property
    def encoder(self):
        if self.__dict__.get("_encoder") and self.enabled:
            return self._encoder
        return None

    @encoder.setter
    def encoder(self, encoder):
        self._encoder = encoder

    @property
    def encryptor(self):
        if self.__dict__.get("_encryptor") and self.enabled:
            return self._encryptor
        return None

    @encryptor.setter
    def encryptor(self, encryptor):
        self._encryptor = encryptor

    @property
    def evaluator(self):
        if self.__dict__.get("_evaluator") and self.enabled:
            return self._evaluator
        return None

    @evaluator.setter
    def evaluator(self, evaluator):
        self._evaluator = evaluator

    @property
    def decryptor(self):
        if self.__dict__.get("_decryptor") and self.enabled:
            return self._decryptor
        return None

    @decryptor.setter
    def decryptor(self, decryptor):
        self._decryptor = decryptor
