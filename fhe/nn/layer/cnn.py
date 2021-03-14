#!/usr/bin/env python3

# @Author: GeorgeRaven <archer>
# @Date:   2020-09-16T11:33:51+01:00
# @Last modified by:   archer
# @Last modified time: 2021-03-13T21:44:13+00:00
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
            self.cc = Cross_Correlation(weights=self.weights,
                                        bias=self.bias,
                                        stride=self.stride)
        cross_correlated = self.cc.forward(x)
        activated = []
        for i in tqdm(range(len(cross_correlated)), desc="{}.{}".format(
            self.__class__.__name__, "forward"),
            ncols=80, colour="blue"
        ):
            t = self.activation_function.forward(cross_correlated.pop(0))
            activated.append(t)
        return activated

    @Layer.bwd
    def backward(self, gradient, x):
        """Calculate the local gradient of this CNN.

        Given the gradient that precedes us,
        what is the local gradient after us.
        """
        ag = gradient
        x = np.array(x)
        # calculate gradient with respect to cross correlation
        df_dx = self.cc.backward(ag, x)
        # return local gradient
        return df_dx

    def update(self):
        self.cc.update()


class Cross_Correlation(Layer):

    @property
    def windows(self):
        if self.cache.get("windows") is not None:
            return self.cache["windows"]

    @windows.setter
    def windows(self, windows):
        self.cache["windows"] = windows

    @Layer.fwd
    def forward(self, x):
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
        # store each cross correlation
        cc = []
        # apply each window and do it by index so can state progress
        for i in tqdm(range(len(self.windows)), desc="{}.{}".format(
                self.__class__.__name__, "forward"),
            ncols=80, colour="blue"
        ):
            # create a primer for application of window without having to
            # modify x but instead the filter itself
            cc_primer = np.zeros(x.shape[1:])
            # now we have a sparse vectore that can be used to convolve
            cc_primer[self.windows[i]] = self.weights
            t = cc_primer * x
            t = t + (self.bias/(t.size/len(t)))  # commute addition before sum
            cc.append(t)
        return cc  # return the now biased convolution ready for activation

    def backward(self, gradient, x):
        # df/dbias is easy as its addition so its same as previous gradient
        self.bias_gradient = gradient * 1  # uneccessary but here for clarity
        # for each window find what it corresponds to in x so we see what
        # specifically the weights were multiplied by in each batch
        # sum all of what the weights were multiplied by together per batch
        # then average out the batches to get a stable gradient
        # the only trick here is that batches are the first dimension and
        # the window expression explicitly ignores this so use a lambda
        # to apply the windows in each batch seperateley
        print("x", x.shape)
        print("weights", self.weights.shape)
        print("gradient", gradient.shape)
        print("windows", self.probe_shape(self.windows))
        # for each window slice apply window to cached x to find what weights
        # were multiplied against
        per_batch_windows = []
        for i in tqdm(range(len(self.windows)), desc="{}.{}".format(
                self.__class__.__name__, "backward-window"),
            ncols=80, colour="blue"
        ):
            batch_window = np.array(
                list(map(lambda a: a[self.windows[i]], x)))
            per_batch_windows.append(batch_window)
        windows = np.array(per_batch_windows)

        print("windows2", windows.shape)
        print("windows_sum", windows.sum(axis=0).shape)
        # # for the number of outputs
        # for i in range(len(windows)):
        #     # for the number of batches in that output
        #     for j in range(len(windows[i])):
        len_diff = len(windows.shape) - len(gradient.shape)
        reshape = np.ones((len_diff,))
        reshape = tuple(map(tuple, reshape))
        reshape = gradient.shape + reshape
        t = np.reshape(gradient, reshape)
        # t = np.reshape(gradient, gradient.shape + tuple(
        #     map(tuple, np.ones((len_diff,)))))
        print("rehsaped_gradient", t.shape)

        self.weights_gradient = (windows * gradient).sum(axis=0)
        # get local gradient of weights with respect to cross correlation
        # weight_total = None
        # # now loop through the length of the now summed windows, which is
        # # effectiveley the batch size so we can sum them up too but also
        # # allow us to calculate the average of these
        # for i in tqdm(range(len(per_batch_sum)), desc="{}.{}".format(
        #     self.__class__.__name__, "backward-weight-avg"),
        #         ncols=80, colour="blue"):
        #     if weight_total is None:
        #         weight_total = per_batch_sum[i]
        #     else:
        #         weight_total += per_batch_sum[i]
        # # final calculation of average
        # weight_avg = weight_total / len(per_batch_sum)
        #
        # # print(x[0][self.windows[i]].shape)
        # # df/dweights is also simple as it is a chain of addition with a single
        # # multiplication against the input so the derivative is just gradient
        # # multiplied by input
        # self.weights_gradients = per_batch_sum * gradient
        local_gradient = 0  # dont care as end of computational chain for now
        # TODO finish calculating gradient of inputs with respect to cc outputs
        return local_gradient

    def update(self, learning_rate=None):
        """We need to update 2 things, both the biases and the weights"""
        learning_rate = learning_rate if learning_rate is not None else 0.001
        # new_parameter = old_parameter - learning_rate * gradient_of_parameter
        self.bias = self.bias - (learning_rate * self.bias_gradient)
        self.weights = self.weights - (learning_rate * self.weights_gradients)

    def windex(self, data: list, filter: list, stride: list,
               dimension: int = 0, partial: list = []):
        """Recursive window index or Windex.

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
                # copy partial window up till now to branch it to mutliple
                # windows
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

    # def test_rearray(self):
    #     cnn = Layer_CNN(weights=self.weights,
    #                     bias=self.bias,
    #                     stride=self.stride)
    #     activations = cnn.forward(x=ReArray(self.data, **self.reseal_args))
    #     accumulator = []
    #     for i in range(len(activations)):
    #         if(i % 10 == 0) or (i == len(activations) - 1):
    #             logger.debug("decrypting: {}".format(len(activations)))
    #         t = np.array(activations.pop(0))
    #         accumulator.append(t)
    #     plaintext_activations = np.around(np.array(accumulator), 2)
    #     compared_activations = np.around(cnn.forward(x=self.data), 2)
    #     self.assertListEqual(plaintext_activations.flatten()[:200].tolist(),
    #                          compared_activations.flatten()[:200].tolist())
    #
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
        level=logger.INFO,
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S")
    # run all the unit-tests
    print("now testing:", __file__, "...")
    unittest.main()
