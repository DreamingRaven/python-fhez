#!/usr/bin/env python3

# @Author: GeorgeRaven <archer>
# @Date:   2020-06-04T13:45:57+01:00
# @Last modified by:   archer
# @Last modified time: 2021-07-24T15:51:36+01:00
# @License: please see LICENSE file in project root

import os
import sys
import tempfile
import unittest
import numpy as np
import logging as logger

import seal

# backward compatibility
from fhez.recache import ReCache
from fhez.rescheme import ReScheme

# pyseal does not at this point support pickling, so what you see here is a
# workaround using seals save and load function to tempfiles so that we can
# read in those files and uses that as a serialised variant instead.
# We cannot use bytesio as seal only accepts file names not the file object
# otherwise this would have been an easy fix to make.


def _getstate_normal(self):
    """Create and return serialised object state."""
    tf = tempfile.NamedTemporaryFile(prefix="fhe_tmp_get_", delete=False)
    self.save(tf.name)
    with open(tf.name, "rb") as file:
        f = file.read()
    os.remove(tf.name)
    # please note this is an incredibly important step!
    # SEAL uses hexidecimal encoding on its saved files so we decode the bytes
    # back into hexidecimal when we read form their files, being both smaller,
    # and more easily serialised with things like marshmallow and json
    # please do also see _setstate_normal for the encoding stage,
    # and also ReScheme class included in this repository or in this file
    f = f.hex()
    # print(f[:32]) # print the first 32 characters of hexadecimal string
    return {"file_contents": f}


def _setstate_normal(self, d):
    """Regenerate object state from serialised object."""
    tf = tempfile.NamedTemporaryFile(prefix="fhe_tmp_set_", delete=False)
    contents = bytes.fromhex(d["file_contents"])
    with open(tf.name, "wb") as f:
        # back to bytes to write to file
        f.write(contents)
    if d.get("context"):
        self.load(d["context"], tf.name)
    else:
        self.load(tf.name)
    os.remove(tf.name)


# rebind setstate and getstate to workable versions
# https://stackoverflow.com/questions/972/adding-a-method-to-an-existing-object-instance
seal.EncryptionParameters.__getstate__ = _getstate_normal
seal.EncryptionParameters.__setstate__ = _setstate_normal

seal.Ciphertext.__getstate__ = _getstate_normal
seal.Ciphertext.__setstate__ = _setstate_normal

seal.PublicKey.__getstate__ = _getstate_normal
seal.PublicKey.__setstate__ = _setstate_normal

seal.SecretKey.__getstate__ = _getstate_normal
seal.SecretKey.__setstate__ = _setstate_normal

seal.KSwitchKeys.__getstate__ = _getstate_normal
seal.KSwitchKeys.__setstate__ = _setstate_normal

seal.RelinKeys.__getstate__ = _getstate_normal
seal.RelinKeys.__setstate__ = _setstate_normal

seal.GaloisKeys.__getstate__ = _getstate_normal
seal.GaloisKeys.__setstate__ = _setstate_normal


class ReSeal(object):
    """Re-binder/ handler for serialisation of MS-Seal objects.

    This is also a Fully Homomorphic Encryption (FHE) utility library.

    This library is designed to streamline and simplify FHE encryption,
    decryption, abstraction, serialisation, and integration. In particular
    this library is intended for use in deep learning to fascilitate a
    more private echosystem. Thus the need for floating point operations means,
    we use the Cheon-Kim-Kim-Song (CKKS) scheme, However BFV will also be
    supported in future
    MS-Seal can be complex and has multiple quirky objects that require
    unique serialisation handling.

    Table showing noise budget increase as poly modulus degree increases,
    allowing more computations.

    +---------------------+------------------------------+
    | poly_modulus_degree | max coeff_modulus bit-length |
    +=====================+==============================+
    | 1024                | 27                           |
    +---------------------+------------------------------+
    | 2048                | 54                           |
    +---------------------+------------------------------+
    | 4096                | 109                          |
    +---------------------+------------------------------+
    | 8192                | 218                          |
    +---------------------+------------------------------+
    | 16384               | 438                          |
    +---------------------+------------------------------+
    | 32768               | 881                          |
    +---------------------+------------------------------+

    number of slots = poly_modulus_degree/2 for CKKS.
    all encoded inputs are padded to the full length of slots.
    scale is the bit-precision of the encoding, and must not get too close to,
    the total size of coeff_modulus.

    CKKS does not use plain_modulus.
    CKKS coeff_modulus has to be selected carefully.


    :param scheme: What type of encryption scheme to use (BFV or CKKS).
    :type scheme: seal.scheme_type
    :param poly_modulus_degree: polynomials degree / effective length
    :type poly_modulus_degree: int
    :param coefficient_modulus: list of int byte sizes switch down mod chain
    :type coefficient_modulus: list
    :param scale: Computational scale/ fixed point precision.
    :type scale: float
    :param parameters: FHE MS-Seal encryption parameters to use throught.
    :type parameters: seal.EncryptionParameters
    :param ciphertext: The encrypted ciphertext with which to compute with.
    :type ciphertext: seal.Ciphertexts
    :param public_key: The key used for encrypting plaintext to ciphertext.
    :type public_key: seal.PublicKey
    :param private_key: The key used for decrypting ciphertext to plaintext.
    :type private_key: seal.PrivateKey
    :param switch_keys:
    :type switch_keys: seal.KSwitchKeys
    :param relin_keys:
    :type relin_keys: seal.RelinKeys
    :param galois_keys:
    :type galois_keys: seal.GaloisKeys
    :example: ReSeal(scheme=seal.scheme_type.CKKS)
    """

    def __init__(self, scheme: seal.scheme_type = None,
                 poly_modulus_degree: int = None,
                 coefficient_modulus: list = None,
                 scale: int = None,
                 parameters: seal.EncryptionParameters = None,
                 ciphertext: seal.Ciphertext = None,
                 public_key: seal.PublicKey = None,
                 private_key: seal.SecretKey = None,
                 switch_keys: seal.KSwitchKeys = None,
                 relin_keys: seal.RelinKeys = None,
                 galois_keys: seal.GaloisKeys = None,
                 cache: bool = None):
        if scheme:
            if scheme == 1:
                scheme = seal.scheme_type.BFV
            elif scheme == 2:
                scheme = seal.scheme_type.CKKS
            self._scheme = scheme
        if poly_modulus_degree:
            self._poly_modulus_degree = poly_modulus_degree
        if coefficient_modulus:
            self._coefficient_modulus = coefficient_modulus
        if scale:
            self._scale = scale
        if parameters:
            self._parameters = parameters
        if ciphertext:
            self._ciphertext = ciphertext
        if public_key:
            self._public_key = public_key
        if private_key:
            self._private_key = private_key
        if switch_keys:
            self._switch_keys = switch_keys
        if relin_keys:
            self._relin_keys = relin_keys
        if galois_keys:
            self._galois_keys = galois_keys

        cache = cache if cache is not None else True
        self._cache = ReCache(enable=cache)

    def __getstate__(self):
        """Create single unified state to allow serialisation."""
        state = {}
        for key in self.__dict__:
            if key in ["_cache"]:
                pass
            elif key in ["_poly_modulus_degree", "_coefficient_modulus",
                         "_scale"]:
                state[key] = self.__dict__[key]
            else:
                state[key] = self.__dict__[key].__getstate__()
        return state

    def __setstate__(self, state):
        """Rebuild all constituent objects from serialised state."""
        # ensuring scheme type is decoded first and must always exist
        self._scheme = seal.scheme_type(state["_scheme"])
        # the order of the dictionary is very important, we will ensure it is
        # as expected or else we may end up trying to initialise the ciphertext
        # before the context which will fail.
        if state.get("_coefficient_modulus"):
            self._coefficient_modulus = state["_coefficient_modulus"]
        if state.get("_poly_modulus_degree"):
            self._poly_modulus_degree = state["_poly_modulus_degree"]
        if state.get("_scale"):
            self._scale = state["_scale"]
        if state.get("_parameters"):
            parameters = seal.EncryptionParameters(self._scheme)
            parameters.__setstate__(state["_parameters"])
            self._parameters = parameters
        if state.get("_ciphertext"):
            ciphertext = seal.Ciphertext()
            state["_ciphertext"].update({"context": self.context})
            ciphertext.__setstate__(state["_ciphertext"])
            self._ciphertext = ciphertext
        if state.get("_public_key"):
            public_key = seal.PublicKey()
            state["_public_key"].update({"context": self.context})
            public_key.__setstate__(state["_public_key"])
            self._public_key = public_key
        if state.get("_private_key"):
            private_key = seal.SecretKey()
            state["_private_key"].update({"context": self.context})
            private_key.__setstate__(state["_private_key"])
            self._private_key = private_key
        if state.get("_switch_keys"):
            switch_keys = seal.KSwitchKeys()
            state["_switch_keys"].update({"context": self.context})
            switch_keys.__setstate__(state["_switch_keys"])
            self._switch_keys = switch_keys
        if state.get("_relin_keys"):
            relin_keys = seal.RelinKeys()
            state["_relin_keys"].update({"context": self.context})
            relin_keys.__setstate__(state["_relin_keys"])
            self._relin_keys = relin_keys
        if state.get("_galois_keys"):
            galois_keys = seal.GaloisKeys()
            state["_galois_keys"].update({"context": self.context})
            galois_keys.__setstate__(state["_galois_keys"])
            self._galois_keys = galois_keys
        # anything that does not match this sequence will of course
        # fall out the bottom again never to be seen again, so make sure
        # this is up to date.

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.__dict__)

    def duplicate(self):
        """Use state dict to instanciate a new ReSeal without ciphertext."""
        # extract desired keys from out internal dictionary
        d = self.__dict__
        d = {k: d[k] for k, v in d.items() if k not in ("_ciphertext",
                                                        "_cache")}
        # now override new reseal object dict with the keys it should share
        new_reseal = ReSeal()
        for key in d:
            new_reseal.__dict__[key] = d[key]
        return new_reseal

    def __str__(self):
        d = self.__dict__
        d = {k: d[k] for k, v in d.items() if k not in ("_ciphertext",
                                                        "_cache")}
        return "{}({})".format(self.__class__.__name__, d)

    # arithmetic operations

    def __add__(self, other):
        if isinstance(other, (ReSeal, seal.Ciphertext)):
            # if adding ciphertext + ciphertext
            encrypted_result = seal.Ciphertext()
            if isinstance(other, ReSeal):
                other = other.ciphertext
            ciphertext, other = self._homogenise_parameters(
                self.ciphertext, other)
            self.evaluator.add(ciphertext,
                               other, encrypted_result)
            # addition of two ciphertexts does not require relinearization
            # or rescaling (by modulus swapping).
        else:
            # if adding ciphertext + numeric plaintext
            plaintext = self._to_plaintext(other)
            encrypted_result = seal.Ciphertext()
            # switching modulus chain of plaintex to ciphertexts level
            # so computation is possible
            ciphertext, plaintext = self._homogenise_parameters(
                a=self.ciphertext, b=plaintext)
            self.evaluator.add_plain(ciphertext, plaintext,
                                     encrypted_result)
            # no need to drop modulus chain addition is fairly small
        # now we take this encrypted result and return it as a new reseal obj
        # so that it can be used as input to __add__ and __mult__ again
        new_reseal_object = self.duplicate()
        new_reseal_object.ciphertext = encrypted_result
        return new_reseal_object

    def __mul__(self, other):
        if isinstance(other, (ReSeal, seal.Ciphertext)):
            # if multiplying ciphertext * ciphertext
            encrypted_result = seal.Ciphertext()
            if isinstance(other, ReSeal):
                other = other.ciphertext
            ciphertext, other = self._homogenise_parameters(
                self.ciphertext, other)
            self.evaluator.multiply(ciphertext, other, encrypted_result)
            self.evaluator.relinearize_inplace(encrypted_result,
                                               self.relin_keys)
            self.evaluator.rescale_to_next_inplace(encrypted_result)
        else:
            # if multiplying ciphertext * numeric
            plaintext = self._to_plaintext(other)
            encrypted_result = seal.Ciphertext()
            # switching modulus chain of plaintex to ciphertexts level
            # so computation is possible
            ciphertext, plaintext = self._homogenise_parameters(
                a=self.ciphertext, b=plaintext)
            # the computation
            self.evaluator.multiply_plain(ciphertext,
                                          plaintext, encrypted_result)
            # dropping one level of modulus chain to stabalise ciphertext
            self.evaluator.rescale_to_next_inplace(encrypted_result)
        # now we take this encrypted result and return it as a new reseal obj
        # so that it can be used as input to __add__ and __mult__ again
        new_reseal_object = self.duplicate()
        new_reseal_object.ciphertext = encrypted_result
        return new_reseal_object

    def __truediv__(self, other):
        """You cannot divide something fully homomorphically encrypted"""
        raise ArithmeticError().with_traceback(sys.exc_info()[2])

    def __len__(self):
        """Deduce the length of the encrypted vector from its poly mod deg."""
        # TODO ensure rigorous type casting if needed to enforce all return int
        # depending on if BFV or CKKS we can deduce the number of slots
        if self.scheme == seal.scheme_type.BFV:
            return self.poly_modulus_degree
        else:  # CKKS has half as many slots as the poly modulus degree
            # here we use the special syntax // to prevent it returning a float
            return self.poly_modulus_degree // 2
        # TODO if this fails try using the encoder.slot_count()

    # reverse arithmetic operations

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    # helpers

    def new(self):
        r = ReSeal()
        # r.__dict__ == self.__dict__
        d = {
            k: v for (k, v) in self.__dict__.items() if "_ciphertext" not in k}
        r.__dict__ = d
        return r

    def _homogenise_parameters(self, a, b):
        """Function to harmonise encryption parameters between objects.

        In particular this prevents:
            ValueError: encrypted1 and encrypted2 parameter mismatch
        which is caused by the encryption parameters such as scale, and
        modulus chain being mismatched.
        This function varied depending on if two ciphers or cipher and plain
        is supplied.
        """
        if isinstance(a, seal.Ciphertext) and isinstance(b, seal.Ciphertext):
            # find which one is lowest on modulus chain and swap both to that
            a_new, b_new = seal.Ciphertext(), seal.Ciphertext()
            a_chain_id = self.context.get_context_data(
                a.parms_id()).chain_index()
            b_chain_id = self.context.get_context_data(
                b.parms_id()).chain_index()
            if b_chain_id < a_chain_id:
                lowest_parms_id = b.parms_id()
            else:
                lowest_parms_id = a.parms_id()
            self.evaluator.mod_switch_to(a, lowest_parms_id, a_new)
            self.evaluator.mod_switch_to(b, lowest_parms_id, b_new)
            # lie to ms seal about scales since they SHOULD BE CLOSE!
            # TODO should happen before modulus switching where we have
            # a bigger noise budget
            a_new.scale()
            b_new.scale()
            a_new.scale(self.scale)
            b_new.scale(self.scale)
            return (a_new, b_new)
        elif isinstance(a, seal.Ciphertext) and isinstance(b, seal.Plaintext):
            # swap modulus chain of plaintext to be that of ciphertext
            ciphertext, plaintext = seal.Ciphertext(), seal.Plaintext()
            # doing both so they are both copied exactly as each other
            # rather than one being a reference, and the other being a new obj
            self.evaluator.mod_switch_to(a, a.parms_id(), ciphertext)
            self.evaluator.mod_switch_to(b, a.parms_id(), plaintext)
            ciphertext.scale()
            ciphertext.scale(self.scale)
            plaintext.scale()
            return (ciphertext, plaintext)
        elif isinstance(b, seal.Ciphertext) and isinstance(a, seal.Plaintext):
            # same as above by swapping a and b around so code is reused
            flipped_tuple = self._homogenise_parameters(a=b, b=a)
            return (flipped_tuple[1], flipped_tuple[0])
        else:
            # someone has been naughty and not given this function propper
            # encryption based objects to work with.
            raise TypeError("Neither parameters are ciphertext or plaintext.")

    def _to_plaintext(self, data):
        plaintext = seal.Plaintext()
        if isinstance(data, np.ndarray):
            data = data.tolist()

        if isinstance(data, (int, float)):
            self.encoder.encode(data, self.scale, plaintext)
        elif isinstance(data, seal.Plaintext):
            plaintext = data
        elif isinstance(data, seal.DoubleVector):
            vector = data
            self.encoder.encode(vector, self.scale, plaintext)
        else:
            vector = seal.DoubleVector(data)
            self.encoder.encode(vector, self.scale, plaintext)
        return plaintext

    @property
    def cache(self):
        """ReCache object to store intermediaries so they arent regenerated."""
        if self.__dict__.get("_cache"):
            return self._cache
        else:
            self._cache = ReCache()
            return self.cache

    # # # basic primitive building blocks (scheme, poly-mod, coeff)
    # {
    #     "scheme": seal.scheme_type.CKKS,
    #     "poly_mod_deg": 8192,
    #     "coeff_mod": [60, 40, 40, 60],
    #     "scale": pow(2.0, 40),
    #     "cache": True,
    # }

    @property
    def scheme(self):
        """Scheme represents the encryption-scheme to use.

        to specify CKKS (you probably want this one):
            ReSeal(scheme=2) OR ReSeal(scheme=seal.scheme_type.CKKS)
        to specify BFV:
            ReSeal(scheme=1) OR ReSeal(scheme=seal.scheme_type.BFV)
        """
        try:
            return self._scheme
        except AttributeError:
            me = self.__class__.__name__
            raise ValueError(
                "You fkn idiot you forgot to give {}({}=SOMETHING)".format(
                    me, "scheme"))

    @property
    def poly_modulus_degree(self):
        """Number dictating the size of cyphertext and compuational depth."""
        try:
            return self._poly_modulus_degree
        except AttributeError:
            me = self.__class__.__name__
            raise ValueError(
                "You fkn idiot you forgot to give {}({}=SOMETHING)".format(
                    me, "poly_modulus_degree"))

    @property
    def coefficient_modulus(self):
        """list of bit precisions of calculations.

        e.g if 8192 is the poly_modulus_degree the maximum number of bits
        in total in the coefficient modulus chain are 218.
        if coefficient modulus is = [60, 40, 40, 60] thats 200 bits
        """
        try:
            return self._coefficient_modulus
        except AttributeError:
            me = self.__class__.__name__
            raise ValueError(
                "You fkn idiot you forgot to give {}({}=SOMETHING)".format(
                    me, "coefficient_modulus"))

    @property
    def scale(self):
        """2^x where x=bytes scale of computations, similar to a bit precision.

        :example:
            ReSeal(scale=pow(2.0, 40))
        """
        try:
            return self._scale
        except AttributeError:
            me = self.__class__.__name__
            raise ValueError(
                "You fkn idiot you forgot to give {}({}=SOMETHING)".format(
                    me, "scale"))

    # # # Encryptor orchestrators and helpers (parameters, context, keygen)
    @property
    def parameters(self):
        """seal.EncryptionParameters object."""
        if self.__dict__.get("_parameters"):
            return self._parameters
        else:
            if self.scheme == seal.scheme_type.CKKS or self.scheme == 2:
                params = seal.EncryptionParameters(self.scheme)
                params.set_poly_modulus_degree(self.poly_modulus_degree)
                params.set_coeff_modulus(seal.CoeffModulus.Create(
                    self.poly_modulus_degree,
                    self.coefficient_modulus))
            else:
                raise NotImplementedError("BFV parameters not yet implemented")

            self.parameters = params
            return self.parameters

    @parameters.setter
    def parameters(self, parameters):
        self._parameters = parameters

    @property
    def context(self):
        """Specific context object for this particular encryption. (cached)"""
        if self.cache.context:
            return self.cache.context
        context = seal.SEALContext.Create(self.parameters)
        self.cache.context = context
        return context

    @property
    def key_generator(self):
        """Using context create key factory."""
        return seal.KeyGenerator(self.context)

    # # # Keys (public, private, relin)
    @property
    def public_key(self):
        """Public key of encryption."""
        if self.__dict__.get("_public_key"):
            return self._public_key
        else:
            keygen = self.key_generator
            self.public_key = keygen.public_key()
            self.private_key = keygen.secret_key()
            if (self.scheme == seal.scheme_type.CKKS) or (self.scheme == 2):
                self.relin_keys = keygen.relin_keys()
            else:
                raise NotImplementedError("BFV key generation not complete")
            return self.public_key

    @public_key.setter
    def public_key(self, key):
        self._public_key = key

    @property
    def private_key(self):
        """Private key of encryption."""
        if self.__dict__.get("_private_key"):
            return self._private_key
        else:
            keygen = self.key_generator
            self.public_key = keygen.public_key()
            self.private_key = keygen.secret_key()
            if (self.scheme == seal.scheme_type.CKKS) or (self.scheme == 2):
                self.relin_keys = keygen.relin_keys()
            else:
                raise NotImplementedError("BFV key generation not complete")
            return self.private_key

    @private_key.setter
    def private_key(self, key):
        self._private_key = key

    @property
    def relin_keys(self):
        """Relinearisation key to relinearize cyphertext after computation."""
        if self.__dict__.get("_relin_keys"):
            return self._relin_keys
        else:
            keygen = self.key_generator
            self.public_key = keygen.public_key()
            self.private_key = keygen.secret_key()
            if (self.scheme == seal.scheme_type.CKKS) or (self.scheme == 2):
                self.relin_keys = keygen.relin_keys()
            else:
                raise NotImplementedError("BFV key generation not complete")
            return self.relin_keys

    @relin_keys.setter
    def relin_keys(self, key):
        self._relin_keys = key

    # # # workers (encryptor, decryptor, encoder, evaluator)
    @property
    def encoder(self):
        """Encoder to turn vector of complex to polynomial plntxt. (cached)"""
        # BFV does not use an encoder so will always be CKKS variant
        if self.cache.encoder:
            return self.cache.encoder
        encoder = seal.CKKSEncoder(self.context)
        self.cache.encoder = encoder
        return encoder

    @property
    def encryptor(self):
        """Encryptor of polynomial plntxt. (cached)"""
        if self.cache.encryptor:
            return self.cache.encryptor
        encryptor = seal.Encryptor(self.context, self.public_key)
        self.cache.encryptor = encryptor
        return encryptor

    @property
    def evaluator(self):
        """Computation evaluator of cyphertext. (cached)"""
        if self.cache.evaluator:
            return self.cache.evaluator
        evaluator = seal.Evaluator(self.context)
        self.cache.evaluator = evaluator
        return evaluator

    @property
    def decryptor(self):
        """Decryptor of cyphertext. (cached)"""
        if self.cache.decryptor:
            return self.cache.decryptor
        decryptor = seal.Decryptor(self.context, self.private_key)
        self.cache.decryptor = decryptor
        return decryptor

    # # # ciphertext
    @property
    def ciphertext(self):
        """seal.Ciphertext cyphertext object storing encrypted message/data."""
        return self._ciphertext

    @ciphertext.setter
    def ciphertext(self, data):
        if isinstance(data, seal.Ciphertext):
            self._ciphertext = data
        elif isinstance(data, ReSeal):
            # compatibility so old setter "r.ciphertext = r + 2" still works
            self.ciphertext = data.ciphertext
        else:
            plaintext = self._to_plaintext(data)
            ciphertext = seal.Ciphertext()
            self.encryptor.encrypt(plaintext, ciphertext)
            self._ciphertext = ciphertext

    # # # plaintext
    @property
    def plaintext(self):
        """Polynomial plaintext encoded from complex values."""
        seal_plaintext = seal.Plaintext()
        self.decryptor.decrypt(self._ciphertext, seal_plaintext)
        vector_plaintext = seal.DoubleVector()
        self.encoder.decode(seal_plaintext, vector_plaintext)
        return np.array(vector_plaintext)


# Alias Reseal to ReSeal
Reseal = ReSeal
