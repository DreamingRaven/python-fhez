#!/usr/bin/env python3

# @Author: GeorgeRaven <archer>
# @Date:   2021-02-11T11:36:15+00:00
# @Last modified by:   archer
# @Last modified time: 2021-02-16T16:01:36+00:00
# @License: please see LICENSE file in project root
import unittest
import numpy as np

from fhe.reseal import ReSeal


class ReArray(np.lib.mixins.NDArrayOperatorsMixin):
    """1D ReSeal array as multidimensional numpy array.

    This class implements a custom numpy container to allow ReSeal to be used
    in conjunction with numpy for its more complex arithmetic, so we dont
    re-invent the wheel. This class assumes the first dimension is the batch
    size I.e given array.shape=(64,32,32,3) this will be 64 examples in this
    batch, each example of shape (32,32,3) this example is what is encrypted,
    the batch size is not. If you do not care for batch size simply set it to
    1, I.e (1,32,32,3). Examples are flattened becoming
    (batchsize, examplesize) where the arithmetic operations are applied to
    each array distinctly by flattening the filter for example.

    You may be asking why handle batches ourselves and not leave it externally.
    The answer is always because they must all share the same exact parameters,
    thus there is a need to handle a "seed". If they dont all share parameters
    then they become inoperable together. It is still possible to handle this
    manually by creating a ReSeal object seed outside of this class and pass
    this in each time but this can be quite clunky. This also allows us to
    optimise somewhat during serialisation as we can handle the duplicate data
    ourselves and not worry the user with the intricacies of serialising this
    encryption.
    """
    # numpy remap class attribute NOT instance attribute!!!
    remap = {}

    def __init__(self,
                 plaintext: np.ndarray = None,
                 seed: ReSeal = None,
                 clone=None,
                 cyphertext=None,
                 **reseal_args):
        if clone is None:
            # automatic seed generation for encryption
            self.seed = reseal_args if seed is None else seed
            # automatic encryption
            self.cyphertext = plaintext
        else:
            # bootstrap ReArray object based on other ReArray object
            d = clone.__dict__
            d = {k: d[k] for k, v in d.items() if k not in ["_cyphertext"]}
            self.__dict__ = d
            self._cyphertext = cyphertext

    @property
    def seedling(self):
        """An independent clone/ sibling of the seed"""
        return self.seed.duplicate()

    @property
    def seed(self):
        """Seed ReSeal object to base all encryption parameters from."""
        return self._seed

    @seed.setter
    def seed(self, seed):
        """Create a ReSeal object seed to allow sharing of encryption keys."""
        if isinstance(seed, ReSeal):
            self._seed = ReSeal
        else:
            self._seed = ReSeal(**seed)
        # call encryptor to test if it exists or to generate it
        self.seed.encryptor

    @property
    def cyphertext(self):
        return self._cyphertext

    @cyphertext.setter
    def cyphertext(self, data):
        if isinstance(data, np.ndarray):
            self._cyphertext = []
            view = data.view()
            # capture original data form so we can return to it later
            # and use it to interpret multidimensional operations
            self.origin = {
                "shape": data.shape,
                "size": data.size,
            }
            # reshape data to (batchsize, examplesize)
            view.shape = (self.origin["shape"][0],
                          int(self.origin["size"] / self.origin["shape"][0]))
            # checking if cyphertext is too small to fit data into
            if view.shape[1] > len(self.seed):
                raise OverflowError(
                    "Data too big or encryption too small to fit:",
                    "data {} -> {} > {} reseal.len".format(
                        self.origin["shape"][1:],
                        view.shape[1],
                        len(self.seed)))
            # iterate through, encrypt (using same seed), and append to list
            # for later use
            for sample in view:
                seedling = self.seedling
                seedling.ciphertext = sample
                self.cyphertext.append(seedling)
        else:
            raise TypeError("data.setter got an {} instead of {}".format(
                type(data), np.ndarray
            ))

    @property
    def origin(self):
        return self._origin

    @origin.setter
    def origin(self, origin: dict):
        self._origin = origin

    @property
    def shape(self):
        return self.origin["shape"]

    @property
    def size(self):
        return self.origin["size"]

    def __repr__(self):
        d = self.__dict__
        d = {k: d[k] for k, v in d.items() if k not in ["_cyphertext"]}
        return "{}({})".format(self.__class__.__name__, d)

    def __str__(self):
        d = self.__dict__
        d = {k: d[k] for k, v in d.items() if k not in ["_cyphertext"]}
        return "{}({})".format(self.__class__.__name__, d)

    def __getitem__(self, indices):
        """Get cyphertexts from encrypted internal 1D list."""
        # converting all indices to tuples if not already
        if not isinstance(indices, tuple):
            return self.cyphertext[indices]
        else:
            raise IndexError("{}[{}] invalid can only slice 1D not {}D".format(
                self.__class__.__name__, indices, len(indices)))

    def __array__(self):
        accumulator = []
        for example in self.cyphertext:
            accumulator.append(
                # cutting off padding/ excess
                example.plaintext[
                    :self.origin["size"]//self.origin["shape"][0]
                ])
        data = np.array(accumulator)
        data.shape = self.origin["shape"]
        return data

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """numpy element wise universal functions."""
        if len(inputs) > 2:
            raise ValueError("More inputs than expected 2 in ufunc")
        # if inputs are wrong way around flip and call again
        elif not isinstance(inputs[0], ReArray):
            return self.__array_ufunc__(ufunc, method, *inputs[::-1], **kwargs)
        # using ReArray objects remap class attribute to dispatch properly
        try:
            # assuming inputs[0] == self then look up function remap
            return inputs[0].remap[method][ufunc](inputs[0], inputs[1])
        except KeyError:
            pass
        # everything else should bottom out as we do not implement
        # e.g floor_divide, true_divide, etc
        return NotImplemented

    def _broadcast(self, other):
        """Broadcast shape to our current shape."""
        return np.broadcast_to(other, self.shape)
        # return np.broadcast_to(other, (1,) + self.shape[1:])

    def _pre_process_other(self, other):
        try:
            other = self._broadcast(other)
        except ValueError:
            raise ArithmeticError("shapes: {}, {} not broadcastable".format(
                self.shape, other.shape))
        return other

    def implements(remap, np_func, method):
        """Python decorator to remap numpy functions to our own funcs."""
        # ensuring subdicts exist
        if remap.get(method) is None:
            remap[method] = {}

        def decorator(func):
            # adding mapping to class' attribute "remap"
            remap[method][np_func] = func
            return func
        return decorator

    @ implements(remap, np.multiply, "__call__")
    def multiply(self, other):
        """Multiplicative Hadmard Product (element-wise multiplication)."""
        other = self._pre_process_other(other)
        accumulator = []
        for i in range(len(self.cyphertext)):
            if isinstance(other[i], ReSeal):
                t = self[i] * other[i]
            else:
                t = self[i] * other[i].flatten()
            accumulator.append(t)
        return ReArray(clone=self, cyphertext=accumulator)

    @ implements(remap, np.add, "__call__")
    def add(self, other):
        """Additive Hadmard Product (element-wise addition)"""
        other = self._pre_process_other(other)
        accumulator = []
        for i in range(len(self.cyphertext)):
            if isinstance(other[i], ReSeal):
                t = self[i] * other[i]
            else:
                t = self[i] * other[i].flatten()
            accumulator.append(t)
        return ReArray(clone=self, cyphertext=accumulator)
        # for row_s, row_o in zip(self.cyphertext, other):
        #     print(row_s, type(row_s), row_o, type(row_o))
        #     accumulator.append(row_s + row_o)
        # return accumulator


class ReArray_tests(unittest.TestCase):
    """Testing ReSeal custom numpy container"""

    def setUp(self):
        import time
        import seal
        self.startTime = time.time()
        self.reseal_args = {
            "scheme": seal.scheme_type.CKKS,
            "poly_modulus_degree": 8192,
            "coefficient_modulus": [60, 40, 40, 60],
            "scale": pow(2.0, 40),
            "cache": True,
        }

    def tearDown(self):
        import time  # dont want time to be imported unless testing as unused
        t = time.time() - self.startTime
        print('%s: %.3f' % (self.id(), t))

    @ property
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
        re = re * re
        # self.assertIsInstance(re, ReArray)
        # out = np.around(np.array(re))
        # self.assertEqual(out.tolist(),
        #                  np.around(self.data * self.data).tolist())

    def test_multiply_broadcast(self):
        """Multiply cyphertext by scalar value broadcast."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        re = re * 2
        # self.assertIsInstance(re, ReArray)
        # out = np.around(np.array(re))
        # self.assertEqual(out.tolist(),
        #                  np.around(self.data * 2).tolist())

    def test_multiply_array(self):
        """Multiply cyphertext by (3) numpy array."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        re = re * np.array([2, 3, 4])
        # self.assertIsInstance(re, ReArray)
        # out = np.around(np.array(re))
        # self.assertEqual(out.tolist(),
        #                  np.around(self.data * np.array([2, 3, 4])).tolist())

    def test_multiply_broadcast_reverse(self):
        """Multiply cyphertext by scalar value broadcast."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        re = 2 * re
        # self.assertIsInstance(re, ReArray)
        # out = np.around(np.array(re))
        # self.assertEqual(out.tolist(),
        #                  np.around(2 * self.data).tolist())

    def test_multiply_array_reverse(self):
        """Multiply cyphertext by (3) numpy array."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        re = np.array([2, 3, 4]) * re
        # self.assertIsInstance(re, ReArray)
        # out = np.around(np.array(re))
        # self.assertEqual(out.tolist(),
        #                  np.around(np.array([2, 3, 4]) * self.data).tolist())

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
        re = re + re
        self.assertIsInstance(re, ReArray)
        # out = np.around(np.array(re))
        # self.assertEqual(out.tolist(),
        #                  np.around(self.data + self.data).tolist())

    def test_add_broadcast(self):
        """Add cyphertext by scalar value broadcast."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        re = re + 2
        self.assertIsInstance(re, ReArray)
        # out = np.around(np.array(re))
        # self.assertEqual(out.tolist(),
        #                  np.around(self.data + 2).tolist())

    def test_add_array(self):
        """Add cyphertext by (3) numpy array value broadcast."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        re = re + np.array([2, 3, 4])
        self.assertIsInstance(re, ReArray)
        # out = np.around(np.array(re))
        # self.assertEqual(out.tolist(),
        #                  np.around(self.data + np.array([2, 3, 4])).tolist())

    def test_add_broadcast_reverse(self):
        """Add cyphertext by scalar value broadcast."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        re = 2 + re
        # self.assertIsInstance(re, ReArray)
        # out = np.around(np.array(re))
        # self.assertEqual(out.tolist(),
        #                  np.around(2 + self.data).tolist())

    def test_add_array_reverse(self):
        """Add cyphertext by (3) numpy array value broadcast."""
        re = ReArray(plaintext=self.data, **self.reseal_args)
        re = np.array([2, 3, 4]) + re
        # self.assertIsInstance(re, ReArray)
        # out = np.around(np.array(re))
        # self.assertEqual(out.tolist(),
        #                  np.around(np.array([2, 3, 4]) + self.data).tolist())

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


if __name__ == "__main__":
    # run all the unit-tests
    print("now testing:", __file__, "...")
    unittest.main()
