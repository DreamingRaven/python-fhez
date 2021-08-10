#!/usr/bin/env python3
"""
Encrypted Array mixin.

Many usefull members for consistent encryption decryption addition etc in as
uniform a format as possible.
"""

# @Author: GeorgeRaven <archer>
# @Date:   2020-06-04T13:45:57+01:00
# @Last modified by:   archer
# @Last modified time: 2021-07-15T15:14:17+01:00
# @License: please see LICENSE file in project root

import os
import sys
import tempfile
import unittest
import numpy as np
import logging as logger

# backward compatibility
from fhez.recache import ReCache
from fhez.rescheme import ReScheme


class Erray(np.lib.mixins.NDArrayOperatorsMixin):
    """One dimensional encrypted array."""

    # numpy remap class attribute NOT instance attribute!!!
    remap = {}

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
        """Broadcast others shape to our current shape."""
        return np.broadcast_to(other, self.shape)

    def _pre_process_other(self, other):
        """Unify compatibility of both members of operation."""
        try:
            other = self._broadcast(other)
        except ValueError:
            raise ArithmeticError("shapes: {}, {} not broadcastable".format(
                self.shape, other.shape))
        return other

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
        t = self + other.flatten()
        return ReArray(clone=self, cyphertext=accumulator)
        # for row_s, row_o in zip(self.cyphertext, other):
        #     print(row_s, type(row_s), row_o, type(row_o))
        #     accumulator.append(row_s + row_o)
        # return accumulator

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.__dict__)

    def __str__(self):
        d = self.__dict__
        d = {k: d[k] for k, v in d.items() if k not in ("_ciphertext",
                                                        "_cache")}
        return "{}({})".format(self.__class__.__name__, d)

    def __len__(self):
        """Matching numpys len function"""
        return self.shape[0]

    @property
    def shape(self):
        """Get the multi dimensional shape that this vector represents."""
        return self.__dict__.get("_shape")

    @property
    def size(self):
        """Get the total number of elements in this vector."""
        return self.__dict__.get("_size")
