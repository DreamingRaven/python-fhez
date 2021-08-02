# @Author: George Onoufriou <archer>
# @Date:   2021-07-26T16:53:04+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-02T14:18:42+01:00

import numpy as np
from fhez.nn.graph.node import Node
from fhez.nn.optimiser.adam import Adam


class Linear(Node):
    """Linear activation function computational graph abstraction."""

    def __init__(self, m=np.array([1]), c=np.array([0]), optimiser=Adam()):
        """Initialise weighted and biased linear function."""
        self.m = m
        self.c = c
        self.optimiser = optimiser

    @property
    def optimiser(self):
        """Get current optimiser object."""
        if self.__dict__.get("_optimiser") is None:
            self._optimiser = Adam()  # default optimiser if none avaliable
        return self._optimiser

    @optimiser.setter
    def optimiser(self, optimiser):
        """Set current optimiser object."""
        self._optimiser = optimiser

    @property
    def m(self):
        """Slope."""
        if self.__dict__.get("_m") is None:
            # defaults to identity y=mx+c where c=0 m=1 so y=x
            self.m = 1
        return self._m

    @m.setter
    def m(self, m):
        self._m = np.array(m)

    @property
    def c(self):
        """Intercept."""
        if self.__dict__.get("_c") is None:
            # defaults to identity y=mx+c where c=0 m=1 so y=x
            self.c = 0
        return self._c

    @c.setter
    def c(self, c):
        self._c = np.array(c)

    def forward(self, x):
        """Get linear forward propogation."""
        # cache input for later re-use
        self.inputs.append(x)
        # return computed forward propogation of node
        return self.m * x + self.c

    def backward(self, gradient):
        """Get gradients of backward prop."""
        # get any cached values required
        x = np.array(self.inputs.pop())
        # calculate gradients respect to inputs and other parameters
        dfdx = self.m * gradient
        dfdm = x * gradient
        dfdc = 1 * gradient
        # assign gradients to dictionary for later retrieval and use
        self.gradients.append({"dfdx": dfdx,
                               "dfdm": dfdm,
                               "dfdc": dfdc})
        # return the gradient with respect to input for immediate use
        return dfdx

    def update(self):
        """Update any weights and biases for a single example."""
        gradients = self.gradients.pop()
        parameters = {
            "m": self.m,
            "c": self.c
        }
        update = self.optimiser.optimise(parms=parameters, grads=gradients)
        self.c = update["c"]
        self.m = update["m"]

    def updates(self):
        """Update any weights and biases based on an avg of all examples."""
        # we store our gradients with names, this is because we want to be
        # able to identify, hold, or modify individual gradients easier
        # than say if they were stored in an array.

        # cumulate like gradients into sums
        batch_sums = {}
        grad_count = {}  # in case some gradients have been held
        for _ in range(len(self.gradients)):
            # for each examples gradient
            gradient_dict = self.gradients.pop()
            for key, value in gradient_dict.items():
                # if no sum already start at 0
                if batch_sums.get(key) is None:
                    batch_sums[key] = 0
                if grad_count.get(key) is None:
                    grad_count[key] = 0
                # add gradient to sum of gradients
                batch_sums[key] += value
                # iterate gradient specific counter by one to keep track
                grad_count[key] += 1

        # now get the average of what we have counted and summed
        avg_gradients = {}
        for key, value in batch_sums.items():
            avg_gradients[key] = value / grad_count[key]
        # here only for compatibility but still wanted to be explicit they are
        # averages
        gradients = avg_gradients

        # explicitly state parameters we want to update
        parameters = {
            "m": self.m,
            "c": self.c
        }
        # call optimiser to calculate probably better weights
        update = self.optimiser.optimise(parms=parameters, grads=gradients)
        # use update dictionary to grab the new weights and set what we want
        self.c = update["c"]
        self.m = update["m"]

    @property
    def cost(self):
        """Get the computational cost of this Node."""
        return 2
