# @Author: George Onoufriou <archer>
# @Date:   2021-07-24T15:39:47+01:00
# @Last modified by:   archer
# @Last modified time: 2021-10-20T10:27:30+01:00
import time
import unittest
import numpy as np

import seal
from fhez.rearray import ReArray


class ReArray_tests(unittest.TestCase):
    """Testing ReSeal custom numpy container"""

    def setUp(self):
        self.startTime = time.time()
        self.reseal_args = {
            "scheme": seal.scheme_type.CKKS,
            "poly_modulus_degree": 8192,
            "coefficient_modulus": [60, 40, 40, 60],
            "scale": pow(2.0, 40),
            "cache": True,
        }

    def tearDown(self):
        t = time.time() - self.startTime
        print('%s: %.3f' % (self.id(), t))

    def arithmetic_evaluator(self, re, other, func, experiment=False):
        self.assertIsInstance(re, ReArray)
        out = np.around(np.array(re), 1).astype(int)
        comparitor = np.around(func(self.data, other)).astype(int)
        if experiment:
            print("out:{}, comparitor:{}".format(out.shape, comparitor.shape))
            print("origin", self.data)
            print("out:", out)
            print("comparitor:", comparitor)
        self.assertEqual(out.tolist(),
                         comparitor.tolist())

    def test_numpy_bug(self):
        a = np.around(np.add(self.data, self.data)).tolist()
        b = np.around(np.add(self.data, self.data)).tolist()
        self.assertEqual(a, b)

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
        other = re
        func = np.multiply
        re = func(re, other)
        self.arithmetic_evaluator(re, np.array(other), func)

    def test_multiply_broadcast(self):
        """Multiply cyphertext by scalar value broadcast."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        other = 2
        func = np.multiply
        re = func(re, other)
        self.arithmetic_evaluator(re, other, func)

    def test_multiply_array(self):
        """Multiply cyphertext by (3) numpy array."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        other = np.array([2, 3, 4])
        func = np.multiply
        re = func(re, other)
        self.arithmetic_evaluator(re, other, func)

    def test_multiply_broadcast_reverse(self):
        """Multiply cyphertext by scalar value broadcast."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        other = 2
        func = np.multiply
        re = func(other, re)
        self.arithmetic_evaluator(re, other, func)

    def test_multiply_array_reverse(self):
        """Multiply cyphertext by (3) numpy array."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        other = np.array([2, 3, 4])
        func = np.multiply
        re = func(other, re)
        self.arithmetic_evaluator(re, other, func)

    def test_multiply_ndarray(self):
        re = ReArray(plaintext=self.data, **self.reseal_args)
        filter = np.arange(3*3*3)
        filter.shape = (3, 3, 3)
        with self.assertRaises(ArithmeticError):
            re = re * filter

    # addition

    def test_add_re(self):
        """Add cyphertext to cyphertext."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        other = re
        func = np.add
        re = func(re, other)
        self.arithmetic_evaluator(re, np.array(other), func)

    def test_add_broadcast(self):
        """Add cyphertext by scalar value broadcast."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        other = 2
        func = np.add
        re = func(re, other)
        self.arithmetic_evaluator(re, other, func)

    def test_add_array(self):
        """Add cyphertext by (3) numpy array value broadcast."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        other = np.array([2, 3, 4])
        func = np.add
        re = func(re, other)
        self.arithmetic_evaluator(re, other, func)

    def test_add_broadcast_reverse(self):
        """Add cyphertext by scalar value broadcast."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        other = 2
        func = np.add
        re = func(other, re)
        self.arithmetic_evaluator(re, other, func)

    def test_add_array_reverse(self):
        """Add cyphertext by (3) numpy array value broadcast."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        other = np.array([2, 3, 4])
        func = np.add
        re = func(other, re)
        self.arithmetic_evaluator(re, other, func)

    def test_add_ndarray(self):
        re = ReArray(plaintext=self.data, **self.reseal_args)
        filter = np.arange(3*3*3)
        filter.shape = (3, 3, 3)
        with self.assertRaises(ArithmeticError):
            re = re + filter

    # subtraction

    def test_subtract_re(self):
        """Subtract cyphertext by cyphertext."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        with self.assertRaises(TypeError):
            re = re - re

    def test_subtract_broadcast(self):
        """Subtract cyphertext by scalar value broadcast."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        with self.assertRaises(TypeError):
            re = re - 2

    def test_subtract_array(self):
        """Subtract cyphertext by (3) numpy array value broadcast."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        with self.assertRaises(TypeError):
            re = re - np.array([2, 3, 4])

    def test_subtract_broadcast_reverse(self):
        """Subtract cyphertext by scalar value broadcast."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        with self.assertRaises(TypeError):
            re = 2 - re

    def test_subtract_array_reverse(self):
        """Subtract cyphertext by (3) numpy array value broadcast."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        with self.assertRaises(TypeError):
            re = np.array([2, 3, 4]) - re

    # division

    def test_true_divide_re(self):
        """True divide cyphertext by cyphertext."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        with self.assertRaises(TypeError):
            re = re / re

    def test_true_divide_broadcast(self):
        """Divide cyphertext by scalar value broadcast."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        with self.assertRaises(TypeError):
            re = re / 2

    def test_true_divide_array(self):
        """Divide cyphertext by (3) numpy array value broadcast."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        with self.assertRaises(TypeError):
            re = re / np.array([2, 3, 4])

    def test_true_divide_broadcast_reverse(self):
        """Divide cyphertext by scalar value broadcast."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        with self.assertRaises(TypeError):
            re = 2 / re

    def test_true_divide_array_reverse(self):
        """Divide cyphertext by (3) numpy array value broadcast."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        with self.assertRaises(TypeError):
            re = np.array([2, 3, 4]) / re

    # floor division

    def test_floor_divide_re(self):
        """Floor divide cyphertext by cyphertext."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        with self.assertRaises(TypeError):
            re = re // re

    def test_floor_divide_broadcast(self):
        """Divide cyphertext by scalar value broadcast."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        with self.assertRaises(TypeError):
            re = re // 2

    def test_floor_divide_array(self):
        """Divide cyphertext by (3) numpy array value broadcast."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        with self.assertRaises(TypeError):
            re = re // np.array([2, 3, 4])

    def test_floor_divide_broadcast_reverse(self):
        """Divide cyphertext by scalar value broadcast."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        with self.assertRaises(TypeError):
            re = 2 // re

    def test_floor_divide_array_reverse(self):
        """Divide cyphertext by (3) numpy array value broadcast."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        with self.assertRaises(TypeError):
            re = np.array([2, 3, 4]) // re

    def test_sum(self):
        data = np.array([
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0]
        ])
        re = ReArray(plaintext=data, **self.reseal_args)
        sum = np.sum(re, axis=0)  # can only sum first axis
        self.assertIsInstance(sum, ReArray)
        plain_sum = np.array(sum)
        truth = np.sum(data, axis=0)
        np.testing.assert_array_almost_equal(plain_sum, truth,
                                             decimal=1,
                                             verbose=True)

    def test_equality(self):
        """Check that ReArray param equality is being calculated properly."""
        a_arg = self.reseal_args = {
            "scheme": seal.scheme_type.CKKS,
            "poly_modulus_degree": 8192,
            "coefficient_modulus": [60, 40, 40, 60],
            "scale": pow(2.0, 40),
            "cache": True,
        }
        b_arg = self.reseal_args = {
            "scheme": seal.scheme_type.CKKS,
            "poly_modulus_degree": 8192,
            "coefficient_modulus": [60, 40, 60],  # <-- changed this
            "scale": pow(2.0, 40),
            "cache": True,
        }
        # TODO check changing every attribute of rearray not just coef_mod
        a = ReArray(np.array([1]), **a_arg)
        b = ReArray(np.array([1]), **b_arg)

        self.assertEqual(a, a)
        self.assertNotEqual(a, b)


if __name__ == "__main__":
    # run all the unit-tests
    print("now testing:", __file__, "...")
    unittest.main()
