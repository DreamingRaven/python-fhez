#!/usr/bin/env python3

# @Author: GeorgeRaven <archer>
# @Date:   2020-06-04T13:45:57+01:00
# @Last modified by:   archer
# @Last modified time: 2021-02-09T00:10:03+00:00
# @License: please see LICENSE file in project root

import os
import sys
import tempfile
import unittest
import numpy as np
import marshmallow
import logging as logger

import seal

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


class Reseal(object):
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

    +----------------------------------------------------+
    | poly_modulus_degree | max coeff_modulus bit-length |
    +---------------------+------------------------------+
    | 1024                | 27                           |
    | 2048                | 54                           |
    | 4096                | 109                          |
    | 8192                | 218                          |
    | 16384               | 438                          |
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
    :type poly_modulus_degree: int
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
    :type switch_keys: seal.KSwitchKeys
    :type relin_keys: seal.RelinKeys
    :type galois_keys: seal.GaloisKeys
    :example: Reseal(scheme=seal.scheme_type.CKKS)
    """

    def __init__(self, scheme=None, poly_modulus_degree=None,
                 coefficient_modulus=None, scale=None, parameters=None,
                 ciphertext=None,
                 public_key=None, private_key=None, switch_keys=None,
                 relin_keys=None,
                 galois_keys=None, cache=None):
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
        new_reseal = Reseal()
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
        if isinstance(other, (Reseal, seal.Ciphertext)):
            # if adding ciphertext + ciphertext
            encrypted_result = seal.Ciphertext()
            if isinstance(other, Reseal):
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
        if isinstance(other, (Reseal, seal.Ciphertext)):
            # if multiplying ciphertext * ciphertext
            encrypted_result = seal.Ciphertext()
            if isinstance(other, Reseal):
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
        r = Reseal()
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
            Reseal(scheme=2) OR Reseal(scheme=seal.scheme_type.CKKS)
        to specify BFV:
            Reseal(scheme=1) OR Reseal(scheme=seal.scheme_type.BFV)
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
        """number dictating the size of cyphertext and compuational depth."""
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
        if self.cache.context:
            return self.cache.context
        context = seal.SEALContext.Create(self.parameters)
        self.cache.context = context
        return context

    @property
    def key_generator(self):
        return seal.KeyGenerator(self.context)

    # # # Keys (public, private, relin)
    @property
    def public_key(self):
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
        # BFV does not use an encoder so will always be CKKS variant
        if self.cache.encoder:
            return self.cache.encoder
        encoder = seal.CKKSEncoder(self.context)
        self.cache.encoder = encoder
        return encoder

    @property
    def encryptor(self):
        if self.cache.encryptor:
            return self.cache.encryptor
        encryptor = seal.Encryptor(self.context, self.public_key)
        self.cache.encryptor = encryptor
        return encryptor

    @property
    def evaluator(self):
        if self.cache.evaluator:
            return self.cache.evaluator
        evaluator = seal.Evaluator(self.context)
        self.cache.evaluator = evaluator
        return evaluator

    @property
    def decryptor(self):
        if self.cache.decryptor:
            return self.cache.decryptor
        decryptor = seal.Decryptor(self.context, self.private_key)
        self.cache.decryptor = decryptor
        return decryptor

    # # # ciphertext
    @property
    def ciphertext(self):
        return self._ciphertext

    @ciphertext.setter
    def ciphertext(self, data):
        if isinstance(data, seal.Ciphertext):
            self._ciphertext = data
        elif isinstance(data, Reseal):
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
        seal_plaintext = seal.Plaintext()
        self.decryptor.decrypt(self._ciphertext, seal_plaintext)
        vector_plaintext = seal.DoubleVector()
        self.encoder.decode(seal_plaintext, vector_plaintext)
        return np.array(vector_plaintext)


class ReCache():
    """Core caching object for Reseal."""

    def __init__(self, enable=None):
        """Object caching.

        If enabled will cache all Reseal objects not already stored,
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


class ReScheme(marshmallow.Schema):
    _scheme = marshmallow.fields.Integer()
    _poly_modulus_degree = marshmallow.fields.Integer()
    _coefficient_modulus = marshmallow.fields.List(
        marshmallow.fields.Integer())
    _scale = marshmallow.fields.Float()
    _parameters = marshmallow.fields.Dict(keys=marshmallow.fields.Str(),
                                          values=marshmallow.fields.Str())
    _public_key = marshmallow.fields.Dict(keys=marshmallow.fields.Str(),
                                          values=marshmallow.fields.Str())
    _private_key = marshmallow.fields.Dict(keys=marshmallow.fields.Str(),
                                           values=marshmallow.fields.Str())
    _relin_keys = marshmallow.fields.Dict(keys=marshmallow.fields.Str(),
                                          values=marshmallow.fields.Str())
    _ciphertext = marshmallow.fields.Dict(keys=marshmallow.fields.Str(),
                                          values=marshmallow.fields.Str())


class Reseal_tests(unittest.TestCase):
    """Unit test class aggregating all tests for the encryption class"""

    def setUp(self):
        import time
        self.startTime = time.time()

    def tearDown(self):
        import time  # dont want time to be imported unless testing as unused
        t = time.time() - self.startTime
        print('%s: %.3f' % (self.id(), t))

    def defaults_ckks(self):
        return {
            "scheme": seal.scheme_type.CKKS,
            "poly_mod_deg": 8192,
            "coeff_mod": [60, 40, 40, 60],
            "scale": pow(2.0, 40),
            "cache": True,
        }

    def defaults_ckks_nocache(self):
        options = self.defaults_ckks()
        options["cache"] = False
        return options

    def gen_reseal(self, defaults):
        if defaults["scheme"] == seal.scheme_type.CKKS:
            r = Reseal(scheme=defaults["scheme"],
                       poly_modulus_degree=defaults["poly_mod_deg"],
                       coefficient_modulus=defaults["coeff_mod"],
                       scale=defaults["scale"])
        else:
            raise NotImplementedError("BFV default gen_reseal not implemented")
        return r

    def test_init(self):
        defaults = self.defaults_ckks()
        r = self.gen_reseal(defaults)
        self.assertIsInstance(r, Reseal)

    def test_serialize_deserialize(self):
        defaults = self.defaults_ckks()
        r = self.gen_reseal(defaults)
        d = r.__getstate__()
        r2 = Reseal()
        r2.__setstate__(d)

    def test_param_property(self):
        defaults = self.defaults_ckks()
        r = self.gen_reseal(defaults)
        self.assertIsInstance(r.parameters, seal.EncryptionParameters)

    def test_context_property(self):
        defaults = self.defaults_ckks()
        r = self.gen_reseal(defaults)
        self.assertIsInstance(r.context, seal.SEALContext)

    def test_publickey_property(self):
        defaults = self.defaults_ckks()
        r = self.gen_reseal(defaults)
        self.assertIsInstance(r.public_key, seal.PublicKey)

    def test_privatekey_property(self):
        defaults = self.defaults_ckks()
        r = self.gen_reseal(defaults)
        self.assertIsInstance(r.private_key, seal.SecretKey)

    def test_relinkeys_property(self):
        defaults = self.defaults_ckks()
        r = self.gen_reseal(defaults)
        self.assertIsInstance(r.relin_keys, seal.RelinKeys)

    def test_ciphertext_property(self):
        defaults = self.defaults_ckks()
        r = self.gen_reseal(defaults)
        r.ciphertext = 100
        self.assertIsInstance(r.ciphertext, seal.Ciphertext)
        r.ciphertext = [1, 2, 3, 4, 5, 100]
        self.assertIsInstance(r.ciphertext, seal.Ciphertext)
        r.ciphertext = np.array([1, 2, 3, 4, 5, 100])
        self.assertIsInstance(r.ciphertext, seal.Ciphertext)

    def test_ciphertext_add_plaintext(self):
        defaults = self.defaults_ckks()
        r = self.gen_reseal(defaults)
        data = np.array([1, 2, 3])
        r.ciphertext = data
        r.ciphertext = r + 2
        r = r + 4  # test return object style
        result = r.plaintext
        print("c+p: 6 +", data, "=", np.round(result[:data.shape[0]]))
        rounded_reshaped_result = np.round(result[:data.shape[0]])
        self.assertEqual((data+6).tolist(), rounded_reshaped_result.tolist())

    def test_ciphertext_add_ciphertext(self):
        import copy
        defaults = self.defaults_ckks()
        r = self.gen_reseal(defaults)
        data = np.array([1, 2, 3])
        r.ciphertext = data
        r2 = copy.deepcopy(r)
        r.ciphertext = r + r2
        r = r + r2  # test return object style
        result = r.plaintext
        print("c+c: 3 *", data, "=", np.round(result[:data.shape[0]]))
        rounded_reshaped_result = np.round(result[:data.shape[0]])
        self.assertEqual((data*3).tolist(), rounded_reshaped_result.tolist())

    def test_ciphertext_multiply_plaintext(self):
        defaults = self.defaults_ckks()
        r = self.gen_reseal(defaults)
        data = np.array([1, 2, 3])
        r.ciphertext = data
        r.ciphertext = r * 2
        r = r * 4  # test return object style
        result = r.plaintext
        print("c*p: 8 *", data, "=", np.round(result[:data.shape[0]]))
        rounded_reshaped_result = np.round(result[:data.shape[0]])
        self.assertEqual((data*8).tolist(), rounded_reshaped_result.tolist())

    def test_ciphertext_multiply_ciphertext(self):
        import copy
        defaults = self.defaults_ckks()
        r = self.gen_reseal(defaults)
        data = np.array([100, 200, 300])
        r.ciphertext = data
        r2 = copy.deepcopy(r)
        r.ciphertext = r * r2
        r = r * r2  # test return object style
        result = r.plaintext
        print("c*c:", data, " ^ 3 =", np.round(result[:data.shape[0]]))
        rounded_reshaped_result = np.round(result[:data.shape[0]])
        self.assertEqual((data * data * data).tolist(),
                         rounded_reshaped_result.tolist())

    def test_encrypt_decrypt(self):
        defaults = self.defaults_ckks()
        r = self.gen_reseal(defaults)
        data = np.array([1, 2, 3])
        r.ciphertext = data
        result = r.plaintext
        rounded_reshaped_result = np.round(result[:data.shape[0]])
        self.assertEqual((data).tolist(), rounded_reshaped_result.tolist())

    def test_complex_arithmetic(self):
        defaults = self.defaults_ckks()
        r = self.gen_reseal(defaults)
        data = np.array([2, 3, 4, 5, 6, 0.5, 8, 9])
        r.ciphertext = data
        r2 = r.new()
        # print("original", r.plaintext[:data.shape[0]])
        r2.ciphertext = 20 * r
        # print("20 * original", r2.plaintext[:data.shape[0]])
        r2.ciphertext = r + r2
        r2 = r2 * r  # test return object style
        expected = ((data * 20) + data) * data
        result = r2.plaintext
        rounded_reshaped_result = np.round(result[:data.shape[0]])
        self.assertEqual(expected.tolist(),
                         rounded_reshaped_result.tolist())

    def test_pickle(self):
        import pickle
        defaults = self.defaults_ckks()
        r = self.gen_reseal(defaults)
        r.ciphertext = np.array([1, 2, 3])
        dump = pickle.dumps(r)
        rp = pickle.loads(dump)
        self.assertIsInstance(rp, Reseal)

    def test_deepcopy(self):
        import copy
        defaults = self.defaults_ckks()
        r = self.gen_reseal(defaults)
        r.ciphertext = np.array([1, 2, 3])
        rp = copy.deepcopy(r)
        self.assertIsInstance(rp, Reseal)

    def test_cache(self):
        defaults = self.defaults_ckks()
        r = self.gen_reseal(defaults)
        self.assertIsInstance(r.cache, ReCache)

    def test_validity(self):
        defaults = self.defaults_ckks()
        r = self.gen_reseal(defaults)
        r.ciphertext = np.array([1, 2, 3])
        ReScheme().validate(r.__getstate__())

    def test_len(self):
        defaults = self.defaults_ckks()
        r = self.gen_reseal(defaults)
        r.ciphertext = np.array([1, 2, 3])
        self.assertIsInstance(len(r), int)


class ReArray(np.lib.mixins.NDArrayOperatorsMixin):
    """1D Reseal array as multidimensional numpy array.

    This class implements a custom numpy container to allow ReSeal to be used
    in conjunction with numpy for its more complex arithmetic, so we dont
    re-invent the wheel. This class assumes the first dimension is the batch
    size I.e given array.shape=(64,32,32,3) this will be 64 examples in this
    batch, each example of shape (32,32,3) this example is what is encrypted,
    the batch size is not. If you do not care for batch size simply set it to
    1, I.e (1,32,32,3). Examples are flattened becoming
    (batchsize, examplesize) where the arithmetic operations are applied to
    each array distinctly by flattening the filter for example.

    You may be asking why handle batches ourselves and not leave it externally.
    The answer is always because they must all share the same exact parameters,
    thus there is a need to handle a "seed". If they dont all share parameters
    then they become inoperable together. It is still possible to handle this
    manually by creating a Reseal object seed outside of this class and pass
    this in each time but this can be quite clunky. This also allows us to
    optimise somewhat during serialisation as we can hanlde the duplicate data
    ourselves and not worry the user with the intricacies of serialising this
    encryption.
    """

    def __init__(self, plaintext: np.ndarray, **reseal_args):
        self.seed = reseal_args  # automatic seed generation for encryption
        self.cyphertext = plaintext  # automatic encryption

    @property
    def seedling(self):
        return self.seed.duplicate()

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        """Create a Reseal object seed to allow sharing of encryption keys."""
        if isinstance(seed, Reseal):
            self._seed = Reseal
        else:
            self._seed = Reseal(**seed)
        # call encryptor to test if it exists or to generate it
        self.seed.encryptor

    @property
    def cyphertext(self):
        return self._cyphertext

    @cyphertext.setter
    def cyphertext(self, data):
        if isinstance(data, np.ndarray):
            self._cyphertext = []
            view = data.view()
            # capture original data form so we can return to it later
            # and use it to interpret multidimensional operations
            self.origin = {
                "shape": data.shape,
                "size": data.size,
            }
            # reshape data to (batchsize, examplesize)
            view.shape = (self.origin["shape"][0],
                          int(self.origin["size"] / self.origin["shape"][0]))
            # checking if cyphertext is too small to fit data into
            if view.shape[1] > len(self.seed):
                raise OverflowError(
                    "Data too big or encryption too small to fit:",
                    "data {} -> {} > {} reseal.len".format(
                        self.origin["shape"][1:],
                        view.shape[1],
                        len(self.seed)))
            # iterate through, encrypt (using same seed), and append to list
            # for later use
            for sample in view:
                seedling = self.seedling
                seedling.ciphertext = sample
                self.cyphertext.append(seedling)
        else:
            raise TypeError("data.setter got an {} instead of {}".format(
                type(data), np.ndarray
            ))

    @property
    def origin(self):
        return self._origin

    @origin.setter
    def origin(self, origin: dict):
        self._origin = origin

    def __repr__(self):
        return "object: {}".format(self.__class__.__name__)

    def __str__(self):
        d = self.__dict__
        d = {k: d[k] for k, v in d.items() if k not in []}
        return "{}({})".format(self.__class__.__name__, d)

    def __array__(self):
        accumulator = []
        for example in self.cyphertext:
            accumulator.append(
                # cutting off padding/ excess
                example.plaintext[
                    :self.origin["size"]//self.origin["shape"][0]
                ])
        data = np.array(accumulator)
        data.shape = self.origin["shape"]
        return data

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "__call__":
            # called for most arithemtic operations
            if ufunc == "add":
                pass
            elif ufunc == "subtract":
                pass
            elif ufunc == "multiply":
                pass
            else:
                # everything else should bottom out as we do not implement
                # e.g floor_divide, true_divide, etc
                pass
        elif method == "reduce":
            pass
        elif method == "reduceat":
            pass
        elif method == "accumulate":
            pass
        elif method == "outer":
            pass
        elif method == "inner":
            pass
        print("called method: {}, ufunc: {}, inputs: {}, kwargs: {}".format(
            method, ufunc, inputs, kwargs))
        return NotImplemented

    def multiply(self, other):
        for sample in self.data:
            pass

    def divide(self, other):
        raise ArithmeticError(
            "Cannot divide by or on FHE cyphertext. Consider approximating.")

    def add(self, other):
        for sample in self.data:
            pass

    def subtract(self, other):
        raise ArithmeticError(
            "Cannot subtract by or on FHE cyphertext. Try adding a negative.")


class ReArray_tests(unittest.TestCase):
    """Testing ReSeal custom numpy container"""

    def setUp(self):
        import time
        self.startTime = time.time()
        self.reseal_args = {
            "scheme": seal.scheme_type.CKKS,
            "poly_modulus_degree": 8192,
            "coefficient_modulus": [60, 40, 40, 60],
            "scale": pow(2.0, 40),
            "cache": True,
        }

    def tearDown(self):
        import time  # dont want time to be imported unless testing as unused
        t = time.time() - self.startTime
        print('%s: %.3f' % (self.id(), t))

    @property
    def data(self):
        array = np.arange(64*32*32*3)
        array.shape = (64, 32, 32, 3)
        return array

    def test_object_creation(self):
        """Checking that the object creation is completed properly."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        self.assertIsInstance(re, ReArray)

    def test_error_slot_overflow(self):
        """Testing that correctly errors when the data overflows encryption."""
        data = np.arange(64*320*320*3)
        data.shape = (64, 320, 320, 3)  # making it waay to big
        with self.assertRaises(OverflowError):
            ReArray(plaintext=data, **self.reseal_args)

    def test__error_data_type(self):
        """Testing that correctly errors when the data overflows encryption."""
        data = np.arange(64*32*32*3)
        data.shape = (64, 32, 32, 3)
        with self.assertRaises(TypeError):
            ReArray(plaintext=data.tolist(), **self.reseal_args)

    def test_str(self):
        re = ReArray(plaintext=self.data, **self.reseal_args)
        self.assertIsInstance(re.__str__(), str)

    def test_repr(self):
        re = ReArray(plaintext=self.data, **self.reseal_args)
        self.assertIsInstance(re.__repr__(), str)

    def test_decrypt(self):
        """Ensure data is intact when decrypted."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        out = re.__array__()
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, self.data.shape)

    def test_numpify(self):
        """Ensure data is intact when decrypted."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        out = np.array(re)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, self.data.shape)

    def test_pickle(self):
        """Ensure that pickling is still possible at this higher dimension."""
        import pickle
        re = ReArray(plaintext=self.data, **self.reseal_args)
        dump = pickle.dumps(re)
        re = pickle.loads(dump)
        self.assertIsInstance(re, ReArray)
        out = np.array(re)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, self.data.shape)

    # multiplication

    def test_multiply_re(self):
        """Multiply cyphertext by cyphertext."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        re = re * re
        out = np.array(re)

    def test_multiply_broadcast(self):
        """Multiply cyphertext by scalar value broadcast."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        re = re * 2
        out = np.array(re)

    def test_multiply_array(self):
        """Multiply cyphertext by (3) numpy array."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        re = re * np.array([2, 3, 4])
        out = np.array(re)

    def test_multiply_broadcast_reverse(self):
        """Multiply cyphertext by scalar value broadcast."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        re = 2 * re
        out = np.array(re)

    def test_multiply_array_reverse(self):
        """Multiply cyphertext by (3) numpy array."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        re = np.array([2, 3, 4]) * re
        out = np.array(re)

    def test_multiply_ndarray(self):
        re = ReArray(plaintext=self.data, **self.reseal_args)
        filter = np.arange(3*3*3)
        filter.shape = (3, 3, 3)
        re = re * filter
        out = np.array(re)

    # addition

    def test_add_re(self):
        """Add cyphertext to cyphertext."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        re = re + re
        out = np.array(re)

    def test_add_broadcast(self):
        """Add cyphertext by scalar value broadcast."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        re = re + 2
        out = np.array(re)

    def test_add_array(self):
        """Add cyphertext by (3) numpy array value broadcast."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        re = re + np.array([2, 3, 4])
        out = np.array(re)

    def test_add_broadcast_reverse(self):
        """Add cyphertext by scalar value broadcast."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        re = 2 + re
        out = np.array(re)

    def test_add_array_reverse(self):
        """Add cyphertext by (3) numpy array value broadcast."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        re = np.array([2, 3, 4]) + re
        out = np.array(re)

    def test_add_ndarray(self):
        re = ReArray(plaintext=self.data, **self.reseal_args)
        filter = np.arange(3*3*3)
        filter.shape = (3, 3, 3)
        re = re + filter
        out = np.array(re)

    # subtraction

    def test_subtract_re(self):
        """Subtract cyphertext by cyphertext."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        re = re - re
        out = np.array(re)

    def test_subtract_broadcast(self):
        """Subtract cyphertext by scalar value broadcast."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        re = re - 2
        out = np.array(re)

    def test_subtract_array(self):
        """Subtract cyphertext by (3) numpy array value broadcast."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        re = re - np.array([2, 3, 4])
        out = np.array(re)

    def test_subtract_broadcast_reverse(self):
        """Subtract cyphertext by scalar value broadcast."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        re = 2 - re
        out = np.array(re)

    def test_subtract_array_reverse(self):
        """Subtract cyphertext by (3) numpy array value broadcast."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        re = np.array([2, 3, 4]) - re
        out = np.array(re)

    # division

    def test_true_divide_re(self):
        """True divide cyphertext by cyphertext."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        with self.assertRaises(TypeError):
            re = re / re
        out = np.array(re)

    def test_true_divide_broadcast(self):
        """Divide cyphertext by scalar value broadcast."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        with self.assertRaises(TypeError):
            re = re / 2
        out = np.array(re)

    def test_true_divide_array(self):
        """Divide cyphertext by (3) numpy array value broadcast."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        with self.assertRaises(TypeError):
            re = re / np.array([2, 3, 4])
        out = np.array(re)

    def test_true_divide_broadcast_reverse(self):
        """Divide cyphertext by scalar value broadcast."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        with self.assertRaises(TypeError):
            re = 2 / re
        out = np.array(re)

    def test_true_divide_array_reverse(self):
        """Divide cyphertext by (3) numpy array value broadcast."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        with self.assertRaises(TypeError):
            re = np.array([2, 3, 4]) / re
        out = np.array(re)

    # floor division

    def test_floor_divide_re(self):
        """Floor divide cyphertext by cyphertext."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        with self.assertRaises(TypeError):
            re = re // re
        out = np.array(re)

    def test_floor_divide_broadcast(self):
        """Divide cyphertext by scalar value broadcast."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        with self.assertRaises(TypeError):
            re = re // 2
        out = np.array(re)

    def test_floor_divide_array(self):
        """Divide cyphertext by (3) numpy array value broadcast."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        with self.assertRaises(TypeError):
            re = re // np.array([2, 3, 4])
        out = np.array(re)

    def test_floor_divide_broadcast_reverse(self):
        """Divide cyphertext by scalar value broadcast."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        with self.assertRaises(TypeError):
            re = 2 // re
        out = np.array(re)

    def test_floor_divide_array_reverse(self):
        """Divide cyphertext by (3) numpy array value broadcast."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        with self.assertRaises(TypeError):
            re = np.array([2, 3, 4]) // re
        out = np.array(re)


if __name__ == "__main__":
    # run all the unit-tests
    print("now testing:", __file__, "...")
    unittest.main()
