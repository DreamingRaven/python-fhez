#!/usr/bin/env python3

# @Author: GeorgeRaven <archer>
# @Date:   2021-02-11T11:36:15+00:00
# @Last modified by:   archer
# @Last modified time: 2021-10-20T10:31:53+01:00
# @License: please see LICENSE file in project root
import numpy as np
import logging as logger
from fhez.reseal import ReSeal
import functools


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
            if cyphertext is not None:
                # check if cyphertext or cyphertexts iterable
                if hasattr(cyphertext, "__iter__"):
                    self._cyphertext = cyphertext
                else:
                    self._cyphertext = [cyphertext]
            else:
                self.cyphertext = plaintext

    def __call__(self, plaintext):
        """Generate clone of ReArray object but with different data."""
        return ReArray(clone=self, plaintext=plaintext)

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

    # @shape.setter
    # def shape(self, shape):
    #     return self.origin["shape"]

    @property
    def size(self):
        return self.origin["size"]

    # @size.setter
    # def size(self, size):
    #     return self.origin["size"]

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

    def __len__(self):
        """Matching numpys len function"""
        return self.shape[0]

    def __array__(self, dtype=None):
        accumulator = []
        for example in self.cyphertext:
            accumulator.append(
                # cutting off padding/ excess
                example.plaintext[
                    :self.origin["size"]//self.origin["shape"][0]
                ])
        data = np.array(accumulator)
        data.shape = self.origin["shape"]
        return data.astype(dtype) if dtype is not None else data

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """numpy element wise universal functions."""
        # print("ufunc: {}, method: {}".format(ufunc, method))
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

    @implements(remap, np.multiply, "__call__")
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

    @implements(remap, np.add, "__call__")
    def add(self, other):
        """Additive Hadmard Product (element-wise addition)"""
        other = self._pre_process_other(other)
        accumulator = []
        for i in range(len(self.cyphertext)):
            if isinstance(other[i], ReSeal):
                t = self[i] + other[i]
            else:
                t = self[i] + other[i].flatten()
            accumulator.append(t)
        return ReArray(clone=self, cyphertext=accumulator)
        # for row_s, row_o in zip(self.cyphertext, other):
        #     print(row_s, type(row_s), row_o, type(row_o))
        #     accumulator.append(row_s + row_o)
        # return accumulator

    @implements(remap, np.add, "reduce")
    def sum(self, axis=None, out=None):
        """Reduce sum of cyphertext."""
        if axis == 0:
            # print("origin", np.array(self), self.shape, self.size)
            cyphertext = functools.reduce(lambda x, y: x+y, self.cyphertext)
            # print("summation", cyphertext,
            # np.array(cyphertext.plaintext).shape)
            # print("preview", np.array(cyphertext.plaintext))
            result = ReArray(cyphertext=cyphertext, clone=self)
            # create a copy of shape, and change it to be summed version
            shape = list(self.shape)
            shape[0] = 1
            shape = tuple(shape)
            # modify origin of this new object as it is different
            result.origin = {"shape": shape,
                             "size": self.size//len(self.cyphertext)}
            # print("out shape", result.shape, result.size)
            return result
        else:
            # we CANNOT fold a single cyphertext, can only sum between
            # cyphertests which for us is axis 0 since we store cyphertexts
            # as a list anything else is impossible
            return NotImplemented

    @implements(remap, np.equal, "__call__")
    def equal(self, other):
        """Check if two ReArray objects are equal.

        Quality to us means that they are the same parms, private-key, etc.
        It does not necessarily check the contents of the cyphertext.
        We cannot always guarantee we have the private-keys to evaluate
        the contents.
        """
        if repr(self) == repr(other):
            return True
        else:
            return False
        return NotImplemented

    @implements(remap, np.not_equal, "__call__")
    def not_equal(self, other):
        """Check two ReArray objects are totally equal in params."""
        return not self.equal(other=other)
