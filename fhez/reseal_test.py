# @Author: George Onoufriou <archer>
# @Date:   2021-07-24T15:47:41+01:00
# @Last modified by:   archer
# @Last modified time: 2021-07-24T15:51:06+01:00

import unittest
import time
import numpy as np

import seal

# backward compatibility
from fhez.recache import ReCache
from fhez.rescheme import ReScheme
from fhez.reseal import ReSeal


class ReSeal_tests(unittest.TestCase):
    """Unit test class aggregating all tests for the encryption class"""

    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
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
            r = ReSeal(scheme=defaults["scheme"],
                       poly_modulus_degree=defaults["poly_mod_deg"],
                       coefficient_modulus=defaults["coeff_mod"],
                       scale=defaults["scale"])
        else:
            raise NotImplementedError("BFV default gen_reseal not implemented")
        return r

    def test_init(self):
        defaults = self.defaults_ckks()
        r = self.gen_reseal(defaults)
        self.assertIsInstance(r, ReSeal)

    def test_serialize_deserialize(self):
        defaults = self.defaults_ckks()
        r = self.gen_reseal(defaults)
        d = r.__getstate__()
        r2 = ReSeal()
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
        self.assertIsInstance(rp, ReSeal)

    def test_deepcopy(self):
        import copy
        defaults = self.defaults_ckks()
        r = self.gen_reseal(defaults)
        r.ciphertext = np.array([1, 2, 3])
        rp = copy.deepcopy(r)
        self.assertIsInstance(rp, ReSeal)

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


if __name__ == "__main__":
    # run all the unit-tests
    print("now testing:", __file__, "...")
    unittest.main()
