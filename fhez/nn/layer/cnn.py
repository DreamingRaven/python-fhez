"""Convolutional Neural Network (CNN) as node abstraction."""
# @Author: GeorgeRaven <archer>
# @Date:   2020-09-16T11:33:51+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-13T15:30:04+01:00
# @License: please see LICENSE file in project root

import copy
import numpy as np
from fhez.nn.graph.node import Node


class CNN(Node):
    """Convolutional Neural Network."""

    def __init__(self,
                 weights: np.ndarray = None,
                 stride: np.ndarray = None,
                 bias: np.ndarray = None,
                 optimiser=None):
        """Create CNN object with any desired default parameters."""
        if weights is not None:
            self.weights = weights
        if stride is not None:
            self.stride = stride
        if bias is not None:
            self.bias = bias
        if optimiser is not None:
            self.optimiser = optimiser

    @property
    def cost(self):
        """Get computational cost of this CNN node."""
        return 6

    def forward(self, x: np.ndarray):
        """Compute convolutional filter forward pass and sums."""
        self.inputs.append(x)
        if self.windows is None:
            # self.windows = self.windex(data=x.shape[1:],
            #                            filter=self.weights.shape[1:],
            #                            stride=self.stride[1:])
            self.windows = self.windex(data=x.shape,
                                       filter=self.weights.shape,
                                       stride=self.stride)
            self.windows = list(map(self.windex_to_slice, self.windows))
        print("CNN WINDOWS 0:", self.windows[0])
        print("CNN WINDOWS 1:", self.windows[1])
        raise NotImplementedError("CNN forward not yet implemented.")

    def backward(self, gradient: np.ndarray):
        """Compute computational filter gradient and input gradient."""
        raise NotImplementedError("CNN backward not yet implemented.")

    def update(self):
        """Update node state/ weights for a single example."""
        self.updater(parm_names=[], it=1)

    def updates(self):
        """Update node state/ weights for multiple examples simultaneously."""
        self.updater(parm_names=[])

    @property
    def weights(self):
        """Get cross convolved filter n-dimensional weights or error."""
        if self.__dict__.get("_w") is None:
            raise ValueError("No weight values have been given to this CNN.")
        return self._w

    @weights.setter
    def weights(self, weights):
        """Set cross convolved filter n-dimensional weights."""
        # initialise weights from tuple dimensions
        # TODO: properly implement xavier weight initialisation over np.rand
        if isinstance(weights, tuple):
            # https://www.coursera.org/specializations/deep-learning
            # https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78
            self._w = np.random.rand(*weights)
            # ensure initial product of weights * x is in range 0-1
            # since each product of these wieghts is summed lets ensure they
            # are smaller than 1 when they are summed by dividing the weights
            # (not X as its a cyphertext) by the number of elements being sumd
            self._w = self.weights / self.weights.size
        else:
            self._w = weights

    @property
    def bias(self):
        """Get sum of products bias coefficient."""
        if self.__dict__.get("_b") is not None:
            return self._b
        else:
            self.bias = np.array([0])
            return self.bias

    @bias.setter
    def bias(self, bias: np.ndarray):
        """Set sum of products bias coefficient."""
        self._b = bias

    @property
    def stride(self):
        """Get stride over convolutions."""
        if self.__dict__.get("_stride") is None:
            self.stride = np.ones(len(self.weights), dtype=int)
            return self.stride
        return self._stride

    @stride.setter
    def stride(self, stride: np.ndarray):
        """Set stride over convolutions.

        Note: Stride **MUST** be integers, we cannot have partial strides.
        """
        self._stride = stride.astype(int)

    @property
    def windows(self):
        """Get current list of windows for cross correlation."""
        if self.cache.get("windows") is not None:
            return self.cache["windows"]
        return None

    @windows.setter
    def windows(self, windows):
        """Set current list of windows into the data."""
        self.cache["windows"] = windows

    # UTILITY

    def windex(self, data: list, filter: list, stride: list,
               dimension: int = 0, partial: list = []):
        """
        Recursive window index or Windex.

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

        .. code-block:: python

            list_of_window_indexes = [
                [[0, 1], [0, 1]], # 0th window
                [[0, 1], [1, 2]], # 1st window
                [[0, 1], [2, 3]], # ...
                [[1, 2], [0, 1]],
                [[1, 2], [1, 2]],
                [[1, 2], [2, 3]],
                [[2, 3], [0, 1]],
                [[2, 3], [1, 2]],
                [[2, 3], [2, 3]], # T_x-1 window
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
