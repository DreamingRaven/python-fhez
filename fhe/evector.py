#!/usr/bin/env python3

# @Author: GeorgeRaven <archer>
# @Date:   2020-05-22T10:23:21+01:00
# @Last modified by:   archer
# @Last modified time: 2020-05-22T13:20:13+01:00
# @License: please see LICENSE file in project root

from fhe import Fhe
import unittest
import copy


class Evector(object):
    """Encrypted vector arithmetic and processor class."""

    def __init__(self, **kwargs):
        defaults = {
            "pylog": print,
            "fhe_plaintext": None,
            "fhe_ciphertext": None,
            "fhe_scheme_type": Fhe().scheme["ckks"],
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

        pass

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


class Evector_tests(unittest.TestCase):
    """Unit test class aggregating all tests for the encrypted vector class."""

    def test_init(self):
        Evector()


if __name__ == "__main__":
    # run all the unit-tests
    print("now testing:", __file__, "...")
    unittest.main()
