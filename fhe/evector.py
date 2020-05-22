#!/usr/bin/env python3

# @Author: GeorgeRaven <archer>
# @Date:   2020-05-22T10:23:21+01:00
# @Last modified by:   archer
# @Last modified time: 2020-05-22T14:47:08+01:00
# @License: please see LICENSE file in project root

from fhe import Fhe
import unittest
import numpy as np
import copy


class Evector(object):
    """Encrypted vector arithmetic and processor class."""

    def __init__(self, array, **kwargs):
        defaults = {
            "pylog": print,
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
        # shape will change need to store it now so can return to origin format
        self.state["fhe_data_shape"] = self.data.shape

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
        """Encrypt whole vector ready for computation."""
        pass

    def decrypt(self):
        """Decrypt vector and return to original format."""
        pass

    def add(self):
        """Add evector with another numeric like object."""
        pass

    def multiply(self):
        """Multiply evector with another numberic like object."""
        pass

    def save(self):
        """Save the encrypted vector to file-like object."""
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
