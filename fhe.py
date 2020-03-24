#!/usr/bin/env python3

# @Author: GeorgeRaven <archer>
# @Date:   2020-03-21T11:30:56+00:00
# @Last modified by:   archer
# @Last modified time: 2020-03-21T11:53:55+00:00
# @License: please see LICENSE file in project root

import os
import unittest
# import seal  # github.com/Huelse/SEAL-Python or DreamingRaven/seal-python
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
            "fhe_data": None,
            "pylog": logger if logger is not None else print,

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

    def debug(self):
        """Display current internal state of all values.

        :return: Returns the internal dictionary.
        :rtype: dict
        """
        self.args["pylog"](self.args)
        return self.args

    debug.__annotations__ = {"return": None}

    def __setitem__(self, key, value):
        """Set a single arg or state by, (key, value)."""
        self.args[key] = value

    __setitem__.__annotations__ = {"key": str, "value": any, "return": None}

    def __getitem__(self, key):
        """Get a single arg or state by, (key, value)."""
        try:
            return self.args[key]
        except KeyError:
            return None  # does not exist is the same as None, gracefull catch

    __getitem__.__annotations__ = {"key": str, "return": any}

    def __delitem__(self, key):
        """Delete a single arg or state by, (key, value)."""
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
        self.assertEqual(Fhe()._merge_dictionary({}, {}), {})
        self.assertEqual(Fhe()._merge_dictionary({"x": 1, "y": 1},
                                                 {"x": 2}), {"x": 2, "y": 1})


if __name__ == "__main__":
    # run all the unit-tests
    unittest.main()
