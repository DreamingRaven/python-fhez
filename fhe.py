#!/usr/bin/env python3

# @Author: GeorgeRaven <archer>
# @Date:   2020-03-21T11:30:56+00:00
# @Last modified by:   archer
# @Last modified time: 2020-03-21T11:53:55+00:00
# @License: please see LICENSE file in project root

import os
import unittest
import seal  # github.com/Huelse/SEAL-Python or DreamingRaven/seal-python
import numpy as np
import pickle


class Fhe(object):
    """Fully Homomorphic Encryption (FHE) utility library.

    This library is designed to streamline and simplify FHE encryption,
    decryption, abstraction, serialisation, and integration. In particular
    this library is intended for use in deep learning to fascilitate a
    more private echosystem.


    :param args: Dictionary of overides.
    :param logger: Function address to print/ log to (default: print).
    :type args: dictionary
    :type logger: function address
    :example: Fhe()
    """

    def __init__(self, args=None, logger=None):
        """Init class with defaults.

        optionally accepts dictionary of default and logger overrides.

        :param args: Optional arguments to override defaults.
        :type args: dict
        :param logger: Optional logger function to override default print.
        :type logger: function
        :return: Fhe object.
        :rtype: object
        """
        args = args if args is not None else dict()
        self.home = os.path.expanduser("~")
        defaults = {
            "pylog": logger if logger is not None else print,
            "fhe_scheme_type": seal.scheme_type.CKKS,
            "fhe_poly_modulus_degree": 8192,
        }
        self.args = self._merge_dictionary(defaults, args)
        # final adjustments to newly defined dictionary

    __init__.__annotations__ = {"args": dict, "logger": print,
                                "return": object}

    def _merge_dictionary(self, *dicts):
        """Given multiple dictionaries, merge together in order.

        :param *dicts: dictionaries merged from low to high priority.
        :type *dicts: dict list
        :return: None.
        :rtype: None
        """
        result = {}
        for dictionary in dicts:
            result.update(dictionary)  # merge each dictionary in order
        return result

    _merge_dictionary.__annotations__ = {"*dicts": dict, "return": dict}

    def create_context(self, fhe_scheme_type=None,
                       fhe_poly_modulus_degree=None):
        """Create an encryption context for encrypting and decrypting data,
        according to an implementation shceme.

        :param fhe_scheme_type: seal.scheme_type to use for context.
        :type fhe_scheme_type: seal.scheme_type
        :return: Seal context to use for encryption.
        :rtype: seal.SEALContext
        """
        scheme = fhe_scheme_type if fhe_scheme_type is not None \
            else self.args["fhe_scheme_type"]
        poly_mod_deg = fhe_poly_modulus_degree if fhe_poly_modulus_degree is \
            not None else self.args["fhe_poly_modulus_degree"]
        context = None

        # self.args["pylog"](help(seal.CoeffModulus))
        self.args["pylog"](seal.CoeffModulus.MaxBitCount(poly_mod_deg),
                           "is the max number of bits we can use in the poly",
                           "modulus degree")

        params = seal.EncryptionParameters(scheme)
        params.set_poly_modulus_degree(poly_mod_deg)
        # params.set_coeff_modulus(seal.CoeffModulus.Create())
        self.args["fhe_context"] = context
        return context

    create_context.__annotations__ = {"fhe_scheme_type": seal.scheme_type,
                                      "return": seal.SEALContext}

    def debug(self):
        """Display current internal state of all values.

        :return: Returns the internal dictionary.
        :rtype: dict
        """
        self.args["pylog"](self.args)
        return self.args

    debug.__annotations__ = {"return": None}

    def __setitem__(self, key, value):
        """Set a single arg or state by, (key, value).

        :param key: Key to replace in internal dictionary.
        :type key: string
        :param value: Item to insert into the internal dictionary.
        :type value: any
        :return: Either the value held in that key or None.
        :rtype: any
        """
        self.args[key] = value

    __setitem__.__annotations__ = {"key": str, "value": any, "return": None}

    def __getitem__(self, key):
        """Get a single arg or state by, (key, value).

        :param key: Key to look up in internal dictionary.
        :type key: string
        :return: Either the value held in that key or None.
        :rtype: any
        """
        try:
            return self.args[key]
        except KeyError:
            return None  # does not exist is the same as None, gracefull catch

    __getitem__.__annotations__ = {"key": str, "return": any}

    def __delitem__(self, key):
        """Delete a single arg or state by, (key, value).

        :param key: Key to delete in internal dictionary.
        :type key: string
        :return: Either the value held in that key or None.
        :rtype: any
        """
        try:
            del self.args[key]
        except KeyError:
            pass  # job is not done but equivalent outcomes so will not error

    __delitem__.__annotations__ = {"key": str, "return": None}


class Fhe_tests(unittest.TestCase):
    """Unit test class aggregating all tests for the encryption class"""

    def test_init(self):
        """Test fhe object initialisation, by using magic function"""
        self.assertEqual(Fhe({"fhe_data": [30]})["fhe_data"], [30])

    def test_magic_get(self):
        obj = Fhe({"test": 30})
        self.assertEqual(obj["test"], 30)

    def test_magic_set(self):
        obj = Fhe({"test": 30})
        obj["test"] = 40
        self.assertEqual(obj["test"], 40)

    def test_magic_del(self):
        obj = Fhe({"test": 30})
        del obj["test"]
        self.assertEqual(obj["test"], None)

    def test_merge_dictionary(self):
        self.assertEqual(Fhe()._merge_dictionary({"x": 1, "y": 1},
                                                 {"x": 2}), {"x": 2, "y": 1})

    def test_create_context(self):
        context = Fhe().create_context()
        print(type(context))


if __name__ == "__main__":
    # run all the unit-tests
    unittest.main()
