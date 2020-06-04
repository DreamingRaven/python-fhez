#!/usr/bin/env python3

# @Author: GeorgeRaven <archer>
# @Date:   2020-06-04T13:45:57+01:00
# @Last modified by:   archer
# @Last modified time: 2020-06-04T16:56:06+01:00
# @License: please see LICENSE file in project root

import os
import tempfile
import unittest

import seal


# def force_context(self, context):
#     self.context = context


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
# seal.PublicKey.set_context = force_context

seal.SecretKey.__getstate__ = getstate_normal
seal.SecretKey.__setstate__ = setstate_normal
# seal.SecretKey.set_context = force_context

seal.KSwitchKeys.__getstate__ = getstate_normal
seal.KSwitchKeys.__setstate__ = setstate_normal
# seal.KSwitchKeys.set_context = force_context

seal.RelinKeys.__getstate__ = getstate_normal
seal.RelinKeys.__setstate__ = setstate_normal
# seal.RelinKeys.set_context = force_context

seal.GaloisKeys.__getstate__ = getstate_normal
seal.GaloisKeys.__setstate__ = setstate_normal
# seal.GaloisKeys.set_context = force_context


class Reseal(object):
    """Re-binder/ handler for serialisation of Seal objects.

    MS-Seal can be complex and has multiple quirky objects that require
    unique serialisation handling.
    """

    def __init__(self, parameters=None, ciphertext=None, public_key=None,
                 private_key=None, switch_keys=None, relin_keys=None,
                 galois_keys=None):
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

        print(self.__dict__)

    def __getstate__(self):
        pass

    def __setstate__(self):
        pass


class Reseal_tests(unittest.TestCase):
    """Unit test class aggregating all tests for the encryption class"""

    def test_init(self):
        scheme = seal.scheme_type.CKKS
        poly_mod_deg = 8192
        coeff_mod = [60, 40, 40, 60]

        params = seal.EncryptionParameters(scheme)
        params.set_poly_modulus_degree(poly_mod_deg)
        params.set_coeff_modulus(
            seal.CoeffModulus.Create(poly_mod_deg,
                                     coeff_mod))
        Reseal(parameters=params)


if __name__ == "__main__":
    # run all the unit-tests
    print("now testing:", __file__, "...")
    unittest.main()
