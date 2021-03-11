# @Author: GeorgeRaven <archer>
# @Date:   2021-03-08T21:09:26+00:00
# @Last modified by:   archer
# @Last modified time: 2021-03-11T21:55:58+00:00
# @License: please see LICENSE file in project root
import numpy as np
from fhe.rearray import ReArray
from fhe.reseal import ReSeal
import logging as logger

from tqdm import tqdm


class Block():
    """Neural network block abstraction.

    Block objects are those that can forward and backpropogate. This also
    abstracts standard Block utilities making them avaliable in all inherited
    children.
    """

    @property
    def cache(self):
        if self.__dict__.get("_cache") is None:
            self._cache = {}
        return self._cache

    @cache.setter
    def cache(self, cache):
        self._cache = cache

    @property
    def x(self):
        """Plaintext x for backward pass"""
        if self.cache.get("x") is None:
            self.cache["x"] = []
        return self.cache["x"]

    @x.setter
    def x(self, x):
        self.cache["x"] = x

    @property
    def gradient(self):
        return self.cache.get("gradient")

    @gradient.setter
    def gradient(self, gradient):
        self.cache["gradient"] = gradient

    def to_plaintext(self, x):
        if isinstance(x, ReArray):
            return np.array(x)
        elif isinstance(x, ReSeal):
            return np.array(x.plaintext)
        else:
            return np.array(x)

    def probe_shape(self, lst: list):
        """Get the shape of a list, assuming each sublist is the same length.

        This function is recursive, sending the sublists down and terminating
        once a type error is thrown by the final point being a non-list
        """
        if isinstance(lst, list):
            # try appending current length with recurse of sublist
            try:
                return (len(lst),) + self.probe_shape(lst[0])
            # once we bottom out and get some non-list type abort and pull up
            except (AttributeError, IndexError):
                return (len(lst),)
        else:
            return lst.shape

    def fwd(func):
        """Forward decorator, unpacking + stashing x to use in backward."""

        def inner(self, x):
            self.x.append(x)
            logger.debug("{}.{} x.shape={}".format(
                self.__class__.__name__,
                func.__name__,
                self.probe_shape(x)))
            temp = func(self, x)
            logger.debug("{}.{} return.shape={}".format(
                self.__class__.__name__,
                func.__name__,
                self.probe_shape(temp)))
            return temp
        return inner

    # def bwd(func):
    #     """Backward decorator to use decrypted or decrypt stashed x."""
    #
    #     def inner(self, gradient=1):
    #         if self.is_activation:
    #             self.x = np.array(self.x)
    #         elif self.is_layer:
    #             self.x = np.squeeze(np.array(self.x), axis=0)
    #
    #         if len(self.x) == 0:
    #             raise ValueError("{}.{}(gradient={}) {}".format(
    #                 self.__class__.__name__,
    #                 func.__name__),
    #                 gradient,
    #                 "has no cached x/ input yet, please run a forward pass")
    #
    #         # if the start of gradient chain I.E is some numeric
    #         if isinstance(gradient, (int, float)):
    #             gradient = np.array([gradient])
    #             gradient = np.broadcast_to(gradient, (1, len(self.x[0])))
    #
    #         logger.debug("{}.{} gradient.shape={}, x.shape={}".format(
    #             self.__class__.__name__,
    #             func.__name__,
    #             gradient.shape,
    #             self.probe_shape(self.x)))
    #
    #         accumulator = []
    #         for i in tqdm(range(len(self.x)), desc="{}.{}".format(
    #                 self.__class__.__name__, func.__name__),
    #                 position=0, leave=False, ncols=80, colour="blue"
    #         ):
    #             # pop and decrypt cached x
    #             x = self.to_plaintext(self.x[i])
    #             accumulator.append(func(self, gradient, x))
    #         del self.x
    #         self.x = []
    #         df_dx = np.array(accumulator)
    #
    #         logger.debug("{}.{} gradient.shape={}".format(
    #             self.__class__.__name__,
    #             func.__name__,
    #             df_dx.shape))
    #
    #         return df_dx
    #     return inner

    def bwd(func):
        """Backward decorator to use decrypted or decrypt stashed x."""

        def inner(self, gradient=1):

            if len(self.x) == 0:
                raise ValueError("{}.{}(gradient={}) {}".format(
                    self.__class__.__name__,
                    func.__name__),
                    gradient,
                    "has no cached x/ input yet, please run a forward pass")
            try:
                # we want to call activation function before going any further
                # this ensures that the gradient is properly handled or if
                # we have to process it here first
                gradient = self.activation_function.backward(gradient)
            except AttributeError:
                pass

            # if the start of gradient chain I.E is some numeric
            if isinstance(gradient, (int, float)):
                gradient = np.array([gradient])
                gradient = np.broadcast_to(gradient, (1, len(self.x[0]),))

            logger.debug("{}.{} gradient.shape={}, x.shape={}".format(
                self.__class__.__name__,
                func.__name__,
                gradient.shape,
                self.probe_shape(self.x)))

            if self.is_activation:
                # activations map one input to one output, thus the same in rev
                accumulator = []
                for i in tqdm(range(len(self.x)), desc="{}.{}".format(
                        self.__class__.__name__, func.__name__),
                        position=0, leave=False, ncols=80, colour="blue"
                ):
                    # pop and decrypt cached x
                    x = self.to_plaintext(self.x.pop(0))
                    accumulator.append(func(self, gradient[i], x))
                df_dx = np.array(accumulator)
            elif self.is_layer:
                # squeeze that additional axis added by appending to a list
                # TODO: loop to handle multiple input passes and move
                # activation function into decorator so it is not looped
                x = np.squeeze(np.array(self.x), axis=0)
                # pass this now squeezed x into backprop
                df_dx = func(self, gradient, x)
            else:
                # checking value of attributes to prevent circular import
                # rather than using isinstance
                raise TypeError("{} is neither a layer or activation".format(
                    self.__class__.__name__))

            logger.debug("{}.{} return.shape={}".format(
                self.__class__.__name__,
                func.__name__,
                df_dx.shape))

            return df_dx
        return inner
