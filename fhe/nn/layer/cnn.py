#!/usr/bin/env python3

# @Author: GeorgeRaven <archer>
# @Date:   2020-09-16T11:33:51+01:00
# @Last modified by:   archer
# @Last modified time: 2021-03-03T11:13:52+00:00
# @License: please see LICENSE file in project root

import logging as logger
import numpy as np
import unittest
import copy

from tqdm import tqdm

import seal
from fhe.rearray import ReArray
from fhe.nn.layer.layer import Layer


class Layer_CNN(Layer):

    @Layer.fwd
    def forward(self, x: (np.array, ReArray)):
        """Take lst of batches of x, return activated output lst of layer."""
        # if no cross correlation object exists yet, create it as inherit Layer
        if self.__dict__.get("cc") is None:
            self.cc = Cross_Correlation()
            self.cc.weights = self.weights
            self.cc.bias = self.bias
            self.cc.stride = self.stride
        cross_correlated = self.cc.forward(x)
        logger.debug("calculating activation")
        activated = []
        for i in tqdm(range(len(cross_correlated)), desc="{}.{}".format(self.__class__.__name__, "forward")):
            # if(i % 10 == 0) or (i == len(cross_correlated) - 1):
            #     logger.debug("calculating activation: {}".format(
            #         len(cross_correlated)))
            t = self.activation_function.forward(cross_correlated.pop(0))
            activated.append(t)
        logger.debug("returning CNN activation")
        return activated

    # @Layer.bwd
    def backward(self, gradient):
        """Calculate the local gradient of this CNN.

        Given the gradient that precedes us,
        what is the local gradient after us.
        """
        # if gradient not given like if its the start of the chain then 1
        gradient = gradient if gradient is not None else 1
        # calculate gradient of activation function
        activation_gradient = self.activation_function.backward(gradient)
        # calculate gradient with respect to cross correlation
        local_gradient = self.cc.backward(activation_gradient)
        # return local gradient
        return local_gradient

    def update(self):
        self.cc.update()


class Cross_Correlation():

    def __init__(self):
        self._cache = {}

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        # initialise weights from tuple dimensions
        # TODO: properly implement xavier weight initialisation over np.rand
        if isinstance(weights, tuple):
            # https://www.coursera.org/specializations/deep-learning
            # https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78
            self._weights = np.random.rand(*weights)
        else:
            self._weights = weights

    @property
    def bias(self):
        if self.__dict__.get("_bias") is not None:
            return self._bias
        else:
            self.bias = 0
            return self.bias

    @bias.setter
    def bias(self, bias):
        self._bias = bias

    @property
    def stride(self):
        if self.__dict__.get("_stride") is not None:
            return self._stride
        else:
            self.stride = np.ones(len(self.weights))
            return self.stride

    @stride.setter
    def stride(self, stride):
        self._stride = stride

    @property
    def x_plain(self):
        """Plaintext x for backward pass"""
        return self._cache["x"]

    @x_plain.setter
    def x_plain(self, x):
        # if isinstance(x, Reseal):
        #     self._cahce["x"] = x.plaintext
        # else:
        self._cache["x"] = x

    @property
    def windows(self):
        if self._cache.get("windows") is not None:
            return self._cache["windows"]

    @windows.setter
    def windows(self, windows):
        self._cache["windows"] = windows

    def forward(self, x):
        logger.debug("CNN forward, batch: {}, range: {}".format(
            self.probe_shape(x), range(len(x))))
        # stride over x using our convolutional filter
        # lets say x = (64, 32, 32, 3) or x_1D = (3, 100, 1, 8)
        # also could be x = (64, 32, 32, crypt), x = (3, 100, 1, crypt)
        # and filter is (32, 3, 3, 3) or x_1D = (1, 100, 1, 8)

        # if no cross correlation windows have been specified create them
        # and cache them for later re-use as uneccessary to re-compute
        if self.windows is None:
            self.windows = self.windex(data=x.shape[1:],
                                       filter=self.weights.shape[1:],
                                       stride=self.stride[1:])
            self.windows = list(map(self.windex_to_slice, self.windows))
        logger.info("Windows processing = {}".format(len(self.windows)))
        # store each cross correlation
        cc = []
        # apply each window and do it by index so can state progress
        for i in tqdm(range(len(self.windows)), desc="{}.{}".format(
                self.__class__.__name__, "forward")):
            # if(i % 10 == 0) or (i == len(self.windows) - 1):
            #     logger.debug("convolving:{}/{}".format(i, len(self.windows)-1))
            # create a primer for application of window without having to
            # modify x but instead the filter itself
            cc_primer = np.zeros(x.shape[1:])
            # now we have a sparse vectore that can be used to convolve
            cc_primer[self.windows[i]] = self.weights
            t = cc_primer * x
            t = t + (self.bias/t.size)  # commuting addition before sum
            cc.append(t)
        return cc  # return the now biased convolution ready for activation

    def backward(self, gradient):
        # calculate local gradient
        x = self.x_plain  # plaintext of x for backprop
        # df/dbias is easy as its addition so its same as previous gradient
        self.bias_gradient = gradient * 1  # uneccessary but here for clarity
        # df/dweights is also simple as it is a chain of addition with a single
        # multiplication against the input so the derivative is just gradient
        # multiplied by input
        self.weights_gradients = x * gradient
        local_gradient = 0  # dont care as end of computational chain
        return local_gradient

    def update(self, learning_rate=None):
        """We need to update 2 things, both the biases and the weights"""
        learning_rate = learning_rate if learning_rate is not None else 0.001
        # new_parameter = old_parameter - learning_rate * gradient_of_parameter
        self.bias = self.bias - (learning_rate * self.bias_gradient)
        self.weights = self.weights - (learning_rate * self.weights_gradients)

    def windex(self, data: list, filter: list, stride: list,
               dimension: int = 0, partial: list = []):
        """Recursive window index (windex).

        This function takes 3 lists; data, filter, and stride.
        Data is a regular multidimensional list, so in the case of a 32x32
        pixel image you would expect a list of shape (32,32,3) 3 being the RGB
        channels.
        Filter is the convolutional filter which we seek to find all windows of
        inside the data. So for data (32,32,3) a standard filter could be
        applied of shape (3,3,3).
        Stride is a 1 dimensional list representing the strides for each
        dimension, so a stride list such as [1,2,3] on data (32,32,3) and
        filter (3,3,3), would move the window 1 in the first 32 dimension,
        2 in the second 32 dim, and 3 in the 3 dimension.

        This function returns a 1D list of all windows, which are themselves
        lists.
        These windows are the same length as the number of dimensions, and each
        dimension consists of indexes with which to slice the original data to
        create the matrix with which to convolve (cross correlate).
        An example given: data.shape=(4,4), filter.shape=(2,2), stride=[1,1]
        list of windows indexes = [
            [[0, 1], [0, 1]], # first window
            [[0, 1], [1, 2]], # second window
            [[0, 1], [2, 3]], # ...
            [[1, 2], [0, 1]],
            [[1, 2], [1, 2]],
            [[1, 2], [2, 3]],
            [[2, 3], [0, 1]],
            [[2, 3], [1, 2]],
            [[2, 3], [2, 3]],
        ]

        We get the indexes rather than the actual data for two reasons:
            - we want to be able to cache this calculation and use it for
              homogenus data that could be streaming into a convolutional
              neural networks, cutting the time per epoch down.
            - we want to use pure list slicing so that we can work with non-
              standard data, E.G Fully Homomorphically Encrypted lists.
        """
        # get shapes of structural lists
        d_shape = data if isinstance(data, tuple) else self.probe_shape(
            data)
        f_shape = filter if isinstance(filter, tuple) else self.probe_shape(
            filter)
        # logger.debug(
        #     "data.shape: {}, filter.shape: {}, stride: {}, dimension: {}".format(
        #         d_shape, f_shape, stride, dimension))
        # if we are not at the end/ last dimension
        if len(stride) > dimension:
            # creating a list matching dimension len so we can slice
            window_heads = list(range(d_shape[dimension]))
            # using dimension list to calculate strides using slicing
            window_heads = window_heads[::stride[dimension]]
            # creating window container to hold each respective window
            windows = []
            # iterate through first index/ head of window
            for window_head in window_heads:
                # copy partial window up till now to branch it to mutliple windows
                current_partial_window = copy.deepcopy(partial)
                # create index range of window in this dimension
                window = list(range(window_head, window_head +
                                    f_shape[dimension]))
                # if window end "-1" is within data bounds
                if (window[-1]) < d_shape[dimension]:
                    # add this dimensions window indexes to partial
                    current_partial_window.append(window)
                    # pass partial to recurse and build it up further
                    subwindow = self.windex(data, filter, stride, dimension+1,
                                            current_partial_window)
                    # logger.debug("subwindow {}: {}".format(dimension,
                    #                                        subwindow))
                    # since we want to create a flat list we want to extend if
                    # the list is still building the partial window or append
                    # if concatenating the partial windows to a single list
                    if (len(stride)-1) > dimension:
                        windows.extend(subwindow)
                    else:
                        windows.append(subwindow)
                else:
                    # discarding illegal windows that are out of bounds
                    pass
            return windows
        else:
            # this is the end of the sequence, can do no more so return
            return partial

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

    def windex_to_slice(self, window):
        """Convert x sides of window expression into slices to slice np."""
        slicedex = ()
        for dimension in window:
            t = (slice(dimension[0], dimension[-1]+1),)
            slicedex += t
        return slicedex


class cnn_tests(unittest.TestCase):
    """Unit test class aggregating all tests for the cnn class"""

    @property
    def data(self):
        array = np.random.rand(2, 15, 15, 3)
        return array

    @property
    def reseal_args(self):
        return {
            "scheme": seal.scheme_type.CKKS,
            "poly_modulus_degree": 8192*2,  # 438
            # "coefficient_modulus": [60, 40, 40, 60],
            "coefficient_modulus":
                [45, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 45],
            "scale": pow(2.0, 30),
            "cache": True,
        }

    def setUp(self):
        import time

        self.weights = (1, 3, 3, 3)  # if tuple allows cnn to initialise itself
        self.stride = [1, 3, 3, 3]  # stride list per-dimension
        self.bias = 0  # assume no bias at first

        self.startTime = time.time()

    def tearDown(self):
        import time  # dont want time to be imported unless testing as unused
        t = time.time() - self.startTime
        print('%s: %.3f' % (self.id(), t))

    def test_numpy_matrix(self):
        cnn = Layer_CNN(weights=self.weights,
                        bias=self.bias,
                        stride=self.stride)
        cnn.forward(x=self.data)
        # cnn.backward(gradient=1)
        # cnn.update()

    def test_rearray(self):
        cnn = Layer_CNN(weights=self.weights,
                        bias=self.bias,
                        stride=self.stride)
        activations = cnn.forward(x=ReArray(self.data, **self.reseal_args))
        accumulator = []
        for i in range(len(activations)):
            if(i % 10 == 0) or (i == len(activations) - 1):
                logger.debug("decrypting: {}".format(len(activations)))
            t = np.array(activations.pop(0))
            accumulator.append(t)
        plaintext_activations = np.around(np.array(accumulator), 2)
        compared_activations = np.around(cnn.forward(x=self.data), 2)
        self.assertListEqual(plaintext_activations.flatten()[:200].tolist(),
                             compared_activations.flatten()[:200].tolist())

    # def test_rearray_cnn_ann(self):
    #     cnn = Layer_CNN(weights=self.weights,
    #                     bias=self.bias,
    #                     stride=self.stride)
    #     activations = cnn.forward(x=ReArray(self.data, **self.reseal_args))
    #     np_acti = cnn.forward(x=self.data)
    #
    #     from fhe.nn.layer.ann import Layer_ANN
    #
    #     dense = Layer_ANN(weights=(len(activations),), bias=0)
    #     y_hat_np = np.sum(np.array(dense.forward(np_acti)))
    #     y_hat_re = np.sum(np.array(dense.forward(activations)))
    #     self.assertEqual(y_hat_np, y_hat_re)

    def test_rearray_backprop(self):
        cnn = Layer_CNN(weights=self.weights,
                        bias=self.bias,
                        stride=self.stride)
        cnn_copy = copy.deepcopy(cnn)
        re_acti = cnn.forward(x=ReArray(self.data, **self.reseal_args))
        np_acti = cnn_copy.forward(x=self.data)

        from fhe.nn.layer.ann import Layer_ANN

        dense = Layer_ANN(weights=(len(re_acti),), bias=0)
        dense_copy = copy.deepcopy(dense)
        y_hat_re = np.sum(np.array(dense.forward(re_acti)))
        y_hat_np = np.sum(np.array(dense_copy.forward(np_acti)))
        gradient = cnn.backward(dense.backward(1))
        print("Gradient:", gradient)
        self.assertEqual(y_hat_np, y_hat_re)


if __name__ == "__main__":
    logger.basicConfig(  # filename="{}.log".format(__file__),
        level=logger.DEBUG,
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S")
    # run all the unit-tests
    print("now testing:", __file__, "...")
    unittest.main()
