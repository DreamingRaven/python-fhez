# @Author: George Onoufriou <archer>
# @Date:   2021-07-26T16:53:04+01:00
# @Last modified by:   archer
# @Last modified time: 2021-09-10T16:26:44+01:00

import numpy as np
import marshmallow as mar
from fhez.fields.numpyfield import NumpyField
from fhez.nn.graph.node import Node
from fhez.nn.optimiser.adam import Adam
from fhez.nn.graph.serialise import Serialise


class Linear(Node, Serialise):
    """Linear activation function computational graph abstraction."""

    def __init__(self, m=np.array([1]), c=np.array([0]), optimiser=Adam()):
        """Initialise weighted and biased linear function."""
        self.m = m
        self.c = c
        self.optimiser = optimiser

    @property
    def schema(self):
        """Get Marshmallow schema representation of this class.

        Marshmallow schemas allow for easy and trustworthy serialisation
        and deserialisation of arbitrary objects either to inbulit types or
        json formats. This is an inherited member of the abstract class
        Serialise.

        .. note::

            Anything not listed here will inevitably be lost, ensure anything
            important is identified and expressley stated its type and
            structure.
        """
        schema_dict = {
            "_m": mar.fields.Float(),
            "_c": NumpyField(),
        }
        return mar.Schema.from_dict(schema_dict)

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
        self.updater(parm_names=["m", "c"], it=1)

    def updates(self):
        """Update any weights and biases based on an avg of all examples."""
        self.updater(parm_names=["m", "c"])

    @property
    def cost(self):
        """Get the computational cost of this Node."""
        return 2
