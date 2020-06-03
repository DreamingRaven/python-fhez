#!/usr/bin/env python3

# @Author: GeorgeRaven <archer>
# @Date:   2020-05-22T10:23:21+01:00
# @Last modified by:   archer
# @Last modified time: 2020-05-22T15:47:34+01:00
# @License: please see LICENSE file in project root

from fhe import Fhe
from logger import Logger
import unittest
import numpy as np
import copy


class Evector(object):
    """Encrypted vector arithmetic and processor class."""

    def __init__(self, array, **kwargs):
        defaults = {
            "pylog": Logger(),
            "fhe_plaintext": None,
            "fhe_ciphertext": None,
            "fhe_scheme_type": Fhe().scheme_type["ckks"],
            "fhe_poly_modulus_degree": 8192,
            "fhe_coeff_modulus": [60, 40, 40, 60],
            "fhe_context": None,
            "fhe_scale": pow(2.0, 40),
            "fhe_public_key": None,
            "fhe_secret_key": None,
            "fhe_relin_keys": None,
            "fhe_encryptor": None,
            "fhe_encoder": None,
            "fhe_decryptor": None,
            # "": None,
        }
        self.state = self._merge_dictionary(defaults, kwargs)

        if(isinstance(array, (np.ndarray))):
            self.data = array
        else:
            self.data = np.array(array)

        if(self.state["fhe_scheme_type"] == Fhe().scheme_type["ckks"]):
            self.state["pylog"](
                "CKKS poly modulus of {}, slots availiable: {}".format(
                    self.state["fhe_poly_modulus_degree"],
                    self.state["fhe_poly_modulus_degree"]/2))
        else:
            raise NotImplementedError(
                "Non CKKS schemes have not been implemented yet" +
                "please come back later")
        # self.fhe = Fhe(self.state)

    def __str__(self):
        """Turning self/ object print statements to something we define.

        This is usually more descriptive that the default python uses,
        all we need to do here is define a nice output for usually programmers
        to see what the hell this is.
        """

        return "Evector object:\nstate: {},\ndata: {}".format(
            self.state, self.data)

    def _merge_dictionary(self, *dicts, to_copy=True):
        """Given multiple dictionaries, merge together in order.

        :param *dicts: dictionaries merged from low to high priority.
        :type *dicts: dict list
        :return: None.
        :rtype: None
        """
        if(to_copy):
            dicts = copy.deepcopy(dicts)
        result = {}
        for dictionary in dicts:
            result.update(dictionary)  # merge each dictionary in order
        return result

    def encrypt(self):
        """Encrypt whole vector ready for computation.

        Takes a numpy array, and encryption keys, and changes it to a
        seal.Ciphertext array, padding with 0's any unfilled slots.
        """

        # create FHE object with all our current setting like keys etc
        fhe = Fhe(self.state)
        plaintext = self.data

        # shape will change need to store it now so can return to origin format
        self.state["fhe_data_shape"] = self.data.shape

        # encrypting using availiable keys and generating any if not here
        ciphertext = fhe.encrypt(fhe_plaintext=plaintext)
        self.data = ciphertext

        # in the case keys have been generated, we now copy the exact
        # encryption keys used to create this ciphertext, will have
        # no effect if the keys were passed in originally, as they will
        # be equal.
        self.state["fhe_public_key"] = fhe.state["fhe_public_key"]
        self.state["fhe_secret_key"] = fhe.state["fhe_secret_key"]
        self.state["fhe_relin_keys"] = fhe.state["fhe_relin_keys"]

        # similarly we also get the exact context used for reuse
        self.state["fhe_contex"] = fhe.state["fhe_context"]

    def decrypt(self):
        """Decrypt vector and return to original format.

        Takes seal.Ciphertext array, and private encryption keys, to decrypts
        and return array to original numpy format and shape by stripping any
        added 0's"""
        fhe = Fhe(self.state)
        ciphertext = self.data
        plaintext = fhe.decrypt(fhe_ciphertext=ciphertext)

        def testing(a):
            print(a)

        # get rid of all those noisy zeros we padded earlier
        plaintext = np.array(list(map(testing, plaintext)))
        # plaintext.resize(self.state["fhe_data_shape"])
        # plaintext = np.resize(plaintext, self.state["fhe_data_shape"])

        self.data = plaintext
        print(self.data, self.data.shape, self.state["fhe_data_shape"])

    def add(self):
        """Add evector with another numeric like object.

        Detects what type of addition this is, whether that be, encrypted +
        unencrypted or encrypted + encrypted so that the correct computation
        can be conducted, then calls underlying seal implementation of this
        addition via our abstraction Fhe(). We shoul for ease of use, also
        support addition of unecnrypted + unencrypted data but this
        is more so people do not have to extract and then re-instantiate the
        class each time they need to do some small operation.
        """
        pass

    def multiply(self):
        """Multiply evector with another numberic like object.

        Detects what type of multiplication this is, whether that be, Encrypted
        * unencrypted or encrypted * encrypted so that the correct computation
        can be conducted, then calls underlying seal implementation of this
        multiplication via our abstraction Fhe()."""
        pass

    def save(self):
        """Save the encrypted vector to file-like object.

        Storing plaintext values is easily conducted without this library,
        so instead we only save encrypted values. We take seal.ciphertext
        and pickle it using the pybind11 implementation by Huelse et al."""
        pass


class Evector_tests(unittest.TestCase):
    """Unit test class aggregating all tests for the encrypted vector class."""

    data = np.array([[1, 2, 3], [4, 5, 6]])

    def test_init(self):
        """Check that init is functioning as intended."""
        e = Evector(self.data)
        self.assertEqual(e.data.tolist(), self.data.tolist())

    def test_encrypt(self):
        """Check that encrypto encrypts in expected manner."""
        e = Evector(self.data)
        e.encrypt()
        # TODO check of type ciphertext

    def test_decrypt(self):
        """Check decryptor correctly decrypts original encryption."""
        e = Evector(self.data)
        e.encrypt()
        # TODO of type ciphertext
        e.decrypt()
        # TODO matches origin


if __name__ == "__main__":
    # run all the unit-tests
    print("now testing:", __file__, "...")
    unittest.main()
