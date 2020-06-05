#!/usr/bin/env python3

# @Author: GeorgeRaven <archer>
# @Date:   2020-06-04T13:45:57+01:00
# @Last modified by:   archer
# @Last modified time: 2020-06-05T22:09:16+01:00
# @License: please see LICENSE file in project root

import os
import tempfile
import unittest

import seal


def getstate_normal(self):
    """Create and return serialised object state."""
    tf = tempfile.NamedTemporaryFile(prefix="fhe_tmp_get_", delete=False)
    self.save(tf.name)
    with open(tf.name, "rb") as file:
        f = file.read()
    os.remove(tf.name)
    return {"file_contents": f}


def setstate_normal(self, d):
    """Regenerate object state from serialised object."""
    tf = tempfile.NamedTemporaryFile(prefix="fhe_tmp_set_", delete=False)
    with open(tf.name, "wb") as f:
        f.write(d["file_contents"])
    if d.get("context"):
        self.load(d["context"], tf.name)
    else:
        self.load(tf.name)
    os.remove(tf.name)


# rebind setstate and getstate to workable versions
# https://stackoverflow.com/questions/972/adding-a-method-to-an-existing-object-instance
seal.EncryptionParameters.__getstate__ = getstate_normal
seal.EncryptionParameters.__setstate__ = setstate_normal

seal.Ciphertext.__getstate__ = getstate_normal
seal.Ciphertext.__setstate__ = setstate_normal

seal.PublicKey.__getstate__ = getstate_normal
seal.PublicKey.__setstate__ = setstate_normal

seal.SecretKey.__getstate__ = getstate_normal
seal.SecretKey.__setstate__ = setstate_normal

seal.KSwitchKeys.__getstate__ = getstate_normal
seal.KSwitchKeys.__setstate__ = setstate_normal

seal.RelinKeys.__getstate__ = getstate_normal
seal.RelinKeys.__setstate__ = setstate_normal

seal.GaloisKeys.__getstate__ = getstate_normal
seal.GaloisKeys.__setstate__ = setstate_normal


class Reseal(object):
    """Re-binder/ handler for serialisation of Seal objects.

    MS-Seal can be complex and has multiple quirky objects that require
    unique serialisation handling.
    """

    def __init__(self, scheme=None, poly_modulus_degree=None,
                 coefficient_modulus=None, parameters=None, ciphertext=None,
                 public_key=None, private_key=None, switch_keys=None,
                 relin_keys=None,
                 galois_keys=None):

        self._scheme = scheme if scheme is not None else seal.scheme_type.CKKS

        # CKKS specific parameters with defaults
        if (self._scheme == 2) or (self._scheme == seal.scheme_type.CKKS):
            self._poly_modulus_degree = poly_modulus_degree if \
                poly_modulus_degree is not None else 8192
            self._coefficient_modulus = coefficient_modulus if \
                coefficient_modulus is not None else [60, 40, 40, 60]
        # BFV specific parameters with defaults
        else:
            raise NotImplementedError("BFV scheme init not yet implemented")

        if parameters is not None:
            self._parameters = parameters
        if ciphertext is not None:
            self._ciphertext = ciphertext
        if public_key is not None:
            self._public_key = public_key
        if private_key is not None:
            self._private_key = private_key
        if switch_keys is not None:
            self._switch_keys = switch_keys
        if relin_keys is not None:
            self._relin_keys = relin_keys
        if galois_keys is not None:
            self._galois_keys = galois_keys

    def __getstate__(self):
        """Create single unified state to allow serialisation."""
        state = {}
        for key in self.__dict__:
            if key in ["_poly_modulus_degree", "_coefficient_modulus"]:
                state[key] = self.__dict__[key]
            else:
                state[key] = self.__dict__[key].__getstate__()
        return state

    def __setstate__(self, state):
        """Rebuild all constituent objects from serialised state."""
        # note: seal getstate of CKKS is the int 2 which can be used directly
        # so do not be confused if you see it as seal.scheme_type.CKKS or as 2
        self._scheme = state["_scheme"]
        for key in state:
            if key == "_scheme":
                pass  # skip already unpacked first
            elif key == "_parameters":
                parameters = seal.EncryptionParameters(self._scheme)
                parameters.__setstate__(state[key])
                self._parameters = parameters
            elif key == "_ciphertext":
                ciphertext = seal.Ciphertext()
                ciphertext.__setstate__(state[key])
                self._ciphertext = ciphertext
            elif key == "_public_key":
                public_key = seal.PublicKey()
                state[key].update({"context": self.context})
                public_key.__setstate__(state[key])
                self._public_key = public_key
            elif key == "_private_key":
                private_key = seal.SecretKey()
                state[key].update({"context": self.context})
                private_key.__setstate__(state[key])
                self._private_key = private_key
            elif key == "_switch_keys":
                switch_keys = seal.KSwitchKeys()
                state[key].update({"context": self.context})
                switch_keys.__setstate__(state[key])
                self._switch_keys = switch_keys
            elif key == "_relin_keys":
                relin_keys = seal.RelinKeys()
                state[key].update({"context": self.context})
                relin_keys.__setstate__(state[key])
                self._relin_keys = relin_keys
            elif key == "_galois_keys":
                galois_keys = seal.GaloisKeys()
                state[key].update({"context": self.context})
                galois_keys.__setstate__(state[key])
                self._galois_keys = galois_keys

    def __str__(self):
        return str(self.__dict__)

    # # # basic primitive building blocks (scheme, poly-mod, coeff)

    @property
    def scheme(self):
        return self._scheme

    @property
    def poly_modulus_degree(self):
        return self._poly_modulus_degree

    @property
    def coefficient_modulus(self):
        return self._coefficient_modulus

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
        return seal.SEALContext.Create(self.parameters)

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

    # # # ciphertext

    @property
    def ciphertext(self):
        return self._ciphertext

    @ciphertext.setter
    def ciphertext(self, data):
        if isinstance(data, seal.Ciphertext):
            self._ciphertext = data
        else:
            if isinstance(data, seal.Plaintext):
                plaintext = data
            else:
                plaintext = seal.Plaintext()
                self.encoder.encode(data, self.scale, plaintext)
            ciphertext = seal.Ciphertext()
            self.encryptor.encrypt(plaintext, ciphertext)
            self._ciphertext = ciphertext


class Reseal_tests(unittest.TestCase):
    """Unit test class aggregating all tests for the encryption class"""

    def defaults_ckks(self):
        return {
            "scheme": seal.scheme_type.CKKS,
            "poly_mod_deg": 8192,
            "coeff_mod": [60, 40, 40, 60],
        }

    def gen_reseal(self, defaults):
        if defaults["scheme"] == seal.scheme_type.CKKS:
            r = Reseal(scheme=defaults["scheme"],
                       poly_modulus_degree=defaults["poly_mod_deg"],
                       coefficient_modulus=defaults["coeff_mod"])
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
        import numpy as np
        defaults = self.defaults_ckks()
        r = self.gen_reseal(defaults)
        r.ciphertext = 100
        self.assertIsInstance(r.ciphertext, seal.Ciphertext)
        r.ciphertext = [1, 2, 3, 4, 5, 100]
        self.assertIsInstance(r.ciphertext, seal.Ciphertext)
        r.ciphertext = np.array[1, 2, 3, 4, 5, 100]
        self.assertIsInstance(r.ciphertext, seal.Ciphertext)

    def test_pickle(self):
        import pickle
        defaults = self.defaults_ckks()
        r = self.gen_reseal(defaults)
        self.assertIsInstance(r.relin_keys, seal.RelinKeys)
        dump = pickle.dumps(r)
        rp = pickle.loads(dump)
        print("original:", r)
        print("unpickled:", rp)

    def test_deepcopy(self):
        import copy
        defaults = self.defaults_ckks()
        r = self.gen_reseal(defaults)
        self.assertIsInstance(r.relin_keys, seal.RelinKeys)
        rc = copy.deepcopy(r)
        print("original:", r)
        print("copied:", rc)


if __name__ == "__main__":
    # run all the unit-tests
    print("now testing:", __file__, "...")
    unittest.main()
