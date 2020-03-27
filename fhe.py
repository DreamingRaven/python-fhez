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
            "fhe_coeff_modulus": [60, 40, 40, 60],
            "fhe_context": None,
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
                       fhe_poly_modulus_degree=None,
                       fhe_coeff_modulus=None):
        """Create an encryption context for encrypting and decrypting data,
        according to an implementation shceme.

        :param fhe_scheme_type: seal.scheme_type to use for context.
        :type fhe_scheme_type: seal.scheme_type
        :param fhe_poly_modulus_degree: .
        :type fhe_poly_modulus_degree: seal.scheme_type
        :param fhe_scheme_type: seal.scheme_type to use for context.
        :type fhe_scheme_type: seal.scheme_type
        :return: Seal context to use for encryption.
        :rtype: seal.SEALContext
        """
        # manage all inputs
        scheme = fhe_scheme_type if fhe_scheme_type is not None \
            else self.args["fhe_scheme_type"]
        poly_mod_deg = fhe_poly_modulus_degree if fhe_poly_modulus_degree is \
            not None else self.args["fhe_poly_modulus_degree"]
        coeff_mod = fhe_coeff_modulus if fhe_coeff_modulus is not None else \
            self.args["fhe_coeff_modulus"]

        self.args["pylog"](seal.CoeffModulus.MaxBitCount(poly_mod_deg),
                           "is the max number of bits we can use in the poly",
                           "modulus degree")
        # check if we exceed the maximum number of bits in coefficient modulus
        max_bit_count = seal.CoeffModulus.MaxBitCount(poly_mod_deg)
        bit_count = np.array(self.args["fhe_coeff_modulus"]).sum()
        if(bit_count <= max_bit_count):

            params = seal.EncryptionParameters(scheme)
            params.set_poly_modulus_degree(poly_mod_deg)
            params.set_coeff_modulus(
                seal.CoeffModulus.Create(poly_mod_deg,
                                         self.args["fhe_coeff_modulus"]))

            context = seal.SEALContext.Create(params)
            self.args["fhe_context"] = context
            # self.log_parameters(context)
            return context
        else:
            self.args["pylog"](self.args["fhe_coeff_modulus"],
                               "exceeds the maximum number of bits for a",
                               "poly_modulus_degree of:",
                               poly_mod_deg, "which is a cumulative total of:",
                               max_bit_count)
            return None

    create_context.__annotations__ = {"fhe_scheme_type": seal.scheme_type,
                                      "return": seal.SEALContext}

    def log_parameters(self, fhe_context=None):
        """Log encryption parameters by context.

        Temporary original version from github huelse/seal-python example
        :param fhe_context: Seal context used to manipulate data.
        :type fhe_context: seal.SEALContext
        :return: None.
        :rtype: None
        """
        context = fhe_context if fhe_context is not None else \
            self.args["fhe_context"]
        context_data = context.key_context_data()
        if context_data.parms().scheme() == seal.scheme_type.BFV:
            scheme_name = "BFV"
        elif context_data.parms().scheme() == seal.scheme_type.CKKS:
            scheme_name = "CKKS"
        else:
            scheme_name = "unsupported scheme"
        print("/")
        print("| Encryption parameters:")
        print("| scheme: " + scheme_name)
        print("| poly_modulus_degree: " +
              str(context_data.parms().poly_modulus_degree()))
        print("| coeff_modulus size: ", end="")
        coeff_modulus = context_data.parms().coeff_modulus()
        coeff_modulus_sum = 0
        for j in coeff_modulus:
            coeff_modulus_sum += j.bit_count()
        print(str(coeff_modulus_sum) + "(", end="")
        for i in range(len(coeff_modulus) - 1):
            print(str(coeff_modulus[i].bit_count()) + " + ", end="")
        print(str(coeff_modulus[-1].bit_count()) + ") bits")
        if context_data.parms().scheme() == seal.scheme_type.BFV:
            print("| plain_modulus: " +
                  str(context_data.parms().plain_modulus().value()))
        print("\\")

    log_parameters.__annotations__ = {"fhe_context": seal.SEALContext,
                                      "return": None}

    def generate_keys(self, fhe_context=None):
        """Generate public, private, and relin keys.

        Given encryption context keys will be stored in internal dictionary,
        and returned as a seperate dictionary.
        :param fhe_context: Seal encryption context to use.
        :type fhe_context: seal.SEALContext
        :return: Dictionary containing public, secret, and relin keys.
        :rtype: dict
        """
        context = fhe_context if fhe_context is not None else \
            self.args["fhe_context"]
        keygen = seal.KeyGenerator(context)
        key_dict = {
            "fhe_public_key": keygen.public_key(),
            "fhe_secret_key": keygen.secret_key(),
            "fhe_relin_keys": keygen.relin_keys(),
        }
        self.args = self._merge_dictionary(self.args, key_dict)
        return key_dict

    generate_keys.__annotations__ = {"fhe_context": seal.SEALContext,
                                     "return": dict}

    def get_encryptor(self, fhe_context=None, fhe_public_key=None):
        """Get encryptor object.

        :param fhe_context: Seal context to use for encryption.
        :type fhe_context: seal.SEALContext
        :param fhe_public_key: Public key to be used for encryption.
        :type fhe_public_key: seal.PublicKey
        :return: Encryptor object to encrypt with.
        :rtype: seal.Encryptor
        """
        context = fhe_context if fhe_context is not None else \
            self.args["fhe_context"]
        public_key = fhe_public_key if fhe_public_key is not None else \
            self.args["fhe_public_key"]
        encryptor = seal.Encryptor(context, public_key)
        self.args["fhe_encryptor"] = encryptor
        return encryptor

    get_encryptor.__annotations__ = {"fhe_context": seal.SEALContext,
                                     "fhe_public_key": seal.PublicKey,
                                     "return": seal.Encryptor}

    def get_evaluator(self, fhe_context=None):
        """Get evaluator object.

        :param fhe_context: Seal context to use for evaluation.
        :type fhe_context: seal.SEALContext
        :return: Evaluator object to evaluate circuits with.
        :rtype: seal.Evaluator
        """
        context = fhe_context if fhe_context is not None else \
            self.args["fhe_context"]
        evaluator = seal.Evaluator(context)
        self.args["fhe_evaluator"] = evaluator
        return evaluator

    get_evaluator.__annotations__ = {"fhe_context": seal.SEALContext,
                                     "return": seal.Evaluator}

    def get_decryptor(self, fhe_context=None, fhe_secret_key=None):
        """Get decryptor object.

        :param get_decryptor: Seal context to use for decryption.
        :type fhe_context: seal.SEALContext
        :param fhe_private_key: Private key to be used for decryption.
        :type fhe_private_key: seal.PrivateKey
        :return: Decryptor object to decrypt with.
        :rtype: seal.Decryptor
        """
        context = fhe_context if fhe_context is not None else \
            self.args["fhe_context"]
        secret_key = fhe_secret_key if fhe_secret_key is not None else \
            self.args["fhe_secret_key"]
        decryptor = seal.Decryptor(context, secret_key)
        self.args["fhe_decryptor"] = decryptor
        return decryptor

    get_decryptor.__annotations__ = {"fhe_context": seal.SEALContext,
                                     "fhe_secret_key": seal.SecretKey,
                                     "return": seal.Decryptor}

    def get_encoder(self, fhe_scheme_type=None, fhe_context=None):
        """Get encoder depending on current scheme

        :param fhe_scheme_type: Intended scheme type for checking.
        :type fhe_scheme_type: seal.scheme_type
        :param fhe_context: Seal context to use for encoding.
        :type fhe_context: seal.SEALContext
        :return: Encoder object.
        :rtype: seal.CKKSEncoder
        """
        context = fhe_context if fhe_context is not None else \
            self.args["fhe_context"]
        scheme_type = fhe_scheme_type if fhe_scheme_type is not None else \
            self.args["fhe_scheme_type"]

        if(scheme_type != fhe_scheme_type):
            self.args["pylog"](
                "scheme mismatch, be carefull the context is correct")
            # TODO complete this to make it rebuild the context
        if(scheme_type == seal.scheme_type.CKKS):
            return self.get_encoder_ckks(fhe_context=context)
        elif(scheme_type == seal.scheme_type.BFV):
            self.args["pylog"](
                "BFV scheme is not currentley supported")
            # TODO: complete this to make it error

    get_encoder.__annotations__ = {"fhe_scheme_type": seal.scheme_type,
                                   "fhe_context": seal.SEALContext,
                                   "return": [seal.CKKSEncoder]}

    def get_encoder_ckks(self, fhe_context=None):
        """Get encoder object for CKKS scheme.

        :param fhe_context: Seal context to use for encoding.
        :type fhe_context: seal.SEALContext
        :return: Encoder object.
        :rtype: seal.CKKSEncoder
        """
        context = fhe_context if fhe_context is not None else \
            self.args["fhe_context"]
        encoder = seal.CKKSEncoder(context)
        self.args["fhe_encoder"] = encoder
        return encoder

    get_encoder_ckks.__annotations__ = {"fhe_context": seal.SEALContext,
                                        "return": seal.CKKSEncoder}

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
        self.assertEqual(Fhe({"fhe_data": [30], "pylog": null_printer})
                         ["fhe_data"], [30],
                         msg="Init did not create our expected object.")

    def test_magic_get(self):
        obj = Fhe({"test": 30, "pylog": null_printer})
        self.assertEqual(obj["test"], 30, msg="Magic get did not get.")

    def test_magic_set(self):
        obj = Fhe({"test": 30, "pylog": null_printer})
        obj["test"] = 40
        self.assertEqual(obj["test"], 40, msg="Magic set did not set.")

    def test_magic_del(self):
        obj = Fhe({"test": 30, "pylog": null_printer})
        del obj["test"]
        self.assertEqual(obj["test"], None, msg="Magic del did not delete.")

    def test_merge_dictionary(self):
        self.assertEqual(
            Fhe({"pylog": null_printer})
            ._merge_dictionary({"x": 1, "y": 1},
                               {"x": 2}), {"x": 2, "y": 1},
            msg="Fhe()._merge_dictionary did not merge dictionaries properly.")

    def test_create_context(self):
        context = Fhe({"pylog": null_printer}).create_context()
        self.assertIsInstance(
            context,
            seal.SEALContext,
            msg="Fhe().create_context() did not return a seal.SEALContext.")

    def test_generate_keys(self):
        fhe = Fhe({"pylog": null_printer})
        fhe.create_context()
        result = fhe.generate_keys()
        self.assertIsInstance(
            result, dict,
            msg="Fhe().generate_keys did not return expected dictionary.")
        for key in result:
            self.assertIsNotNone(
                result[key], msg="Fhe().generate_keys[{}] is None".format(key))
            if(key == "fhe_public_key"):
                self.assertIsInstance(result[key], seal.PublicKey)
            elif(key == "fhe_secret_key"):
                self.assertIsInstance(result[key], seal.SecretKey)
            elif(key == "fhe_relin_keys"):
                self.assertIsInstance(result[key], seal.RelinKeys)
            else:
                self.assertFalse(
                    1,
                    msg="result['{}'] is not a key we expect".format(key))

    def test_get_encryptor(self):
        fhe = Fhe({"pylog": null_printer})
        context = fhe.create_context()
        keys = fhe.generate_keys()
        # without overrides
        encryptor = fhe.get_encryptor()
        self.assertIsInstance(encryptor, seal.Encryptor)
        self.assertIsInstance(fhe["fhe_encryptor"], seal.Encryptor)
        # with overrides
        encryptor = fhe.get_encryptor(fhe_context=context,
                                      fhe_public_key=keys["fhe_public_key"])
        self.assertIsInstance(encryptor, seal.Encryptor)
        self.assertIsInstance(fhe["fhe_encryptor"], seal.Encryptor)

    def test_get_evaluator(self):
        fhe = Fhe({"pylog": null_printer})
        context = fhe.create_context()
        keys = fhe.generate_keys()
        # without overrides
        evaluator = fhe.get_evaluator()
        self.assertIsInstance(evaluator, seal.Evaluator)
        self.assertIsInstance(fhe["fhe_evaluator"], seal.Evaluator)
        # with overrides
        evaluator = fhe.get_evaluator(fhe_context=context)
        self.assertIsInstance(evaluator, seal.Evaluator)
        self.assertIsInstance(fhe["fhe_evaluator"], seal.Evaluator)

    def test_get_decryptor(self):
        fhe = Fhe({"pylog": null_printer})
        context = fhe.create_context()
        keys = fhe.generate_keys()
        # without overrides
        decryptor = fhe.get_decryptor()
        self.assertIsInstance(decryptor, seal.Decryptor)
        self.assertIsInstance(fhe["fhe_decryptor"], seal.Decryptor)
        # with overrides
        decryptor = fhe.get_decryptor(fhe_context=context,
                                      fhe_secret_key=keys["fhe_secret_key"])
        self.assertIsInstance(decryptor, seal.Decryptor)
        self.assertIsInstance(fhe["fhe_decryptor"], seal.Decryptor)

    def test_get_encoder_ckks(self):
        fhe = Fhe({"pylog": null_printer})
        context = fhe.create_context()
        keys = fhe.generate_keys()
        # without overrides
        encoder = fhe.get_encoder_ckks()
        self.assertIsInstance(encoder, seal.CKKSEncoder)
        self.assertIsInstance(fhe["fhe_encoder"], seal.CKKSEncoder)
        # with overrides
        encoder = fhe.get_encoder_ckks(fhe_context=context)
        self.assertIsInstance(encoder, seal.CKKSEncoder)
        self.assertIsInstance(fhe["fhe_encoder"], seal.CKKSEncoder)

    def test_get_encoder(self):
        # testing CKKS version not BFV yet
        # TODO add section for BFV when its ready
        fhe = Fhe({"pylog": null_printer,
                   "fhe_scheme_type": seal.scheme_type.CKKS})
        context = fhe.create_context()
        keys = fhe.generate_keys()
        # without overrides
        encoder = fhe.get_encoder()
        self.assertIsInstance(encoder, seal.CKKSEncoder)
        self.assertIsInstance(fhe["fhe_encoder"], seal.CKKSEncoder)
        # with overrides
        encoder = fhe.get_encoder(fhe_context=context)
        self.assertIsInstance(encoder, seal.CKKSEncoder)
        self.assertIsInstance(fhe["fhe_encoder"], seal.CKKSEncoder)


def null_printer(*args):
    pass


if __name__ == "__main__":
    # run all the unit-tests
    unittest.main()
