"""Graph node abstraction for neural networks."""
# @Author: George Onoufriou <archer>
# @Date:   2021-07-15T15:43:16+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-02T16:08:32+01:00

import abc
from collections import deque
from fhez.nn.optimiser.optimiser import DefaultOptimiser


class Node(abc.ABC):
    """Abstract class for neural network nodes for traversal/ computation."""

    # # # Caching
    # This section deals with caching/ enabling/ disabling caching
    # it is the responsibility of subclassers to respect this flag but we help
    # with some properties such as "inputs" being cache aware
    @property
    def is_cache_enabled(self):
        """Get status of whether or not caching is enabled."""
        if self.__dict__.get("_is_cache_enabled") is None:
            # cache enabled by default
            self._is_cache_enabled = True
        return self._is_cache_enabled

    @is_cache_enabled.setter
    def is_cache_enabled(self, state: bool):
        """Set the state of the cache."""
        self._is_cache_enabled = state

    def enable_cache(self):
        """Enable caching."""
        self.is_cache_enabled = True

    def disable_cache(self):
        """Disable caching."""
        self.is_cache_enabled = False

    @property
    def cache(self):
        """Get caching dictionary of auxilary data."""
        # initialise empty cache if it does not exist already
        if self.__dict__.get("_cache") is None:
            self._cache = {}
        return self._cache

    @cache.setter
    def cache(self, cache):
        self._cache = cache

    # # # Utility Methods
    # these methods help implementers respect flags such as enabled cache,
    # while also alleviating some of the repeate code needing to be implemented

    @property
    def inputs(self):
        """Get cached input stack.

        Neural networks backpropogation requires cached inputs to calculate
        the gradient with respect to x and the weights. This is a utility
        method that initialises a stack and allows you to easily append
        or pop off of it so that the computation can occur in FILO.
        """
        if self.cache.get("_inputs") is None:
            self.cache["_inputs"] = deque()
        if self.is_cache_enabled:
            # if cache enabled return real stack
            return self.cache["_inputs"]
        # if cache disabled return dud que
        return deque()

    @property
    def gradients(self):
        """Get cached input stack.

        For neural networks to calculate any given weight update, it needs to
        remember atleast the last gradient in the case of stocastic descent,
        or multiple gradients if implementing batch normalised gradient
        descent. This is a helper method that initialises a stack so that
        implementation can be offloaded and made-uniform between all subclasses
        """
        if self.cache.get("_gradients") is None:
            self.cache["_gradients"] = deque()
        if self.is_cache_enabled:
            # if cache enabled return real stack
            return self.cache["_gradients"]
        # if cache disabled return dud que
        return deque()

    @property
    def optimiser(self):
        """Get optimiser object, E.G Stocastic Gradient Descent."""
        if self.__dict__.get("_optimiser") is None:
            self._optimiser = DefaultOptimiser()  # let defaults default
        return self._optimiser

    @optimiser.setter
    def optimiser(self, optimiser):
        self._optimiser = optimiser

    def updater(self, parm_names: list, it=None):
        """Private function to universaly update any Node instance.

        To simplify the process of updating so and to reduce code duplication,
        this function serves to derive all the important information given
        a parameter dictionary. It will then infer from the dictionary the
        attributes with which to modify.
        """
        it = it if it is not None else len(self.gradients)
        # we store our gradients with names, this is because we want to be
        # able to identify, hold, or modify individual gradients easier
        # than say if they were stored in an array.

        # cumulate like gradients into sums
        batch_sums = {}
        grad_count = {}  # in case some gradients have been held
        # for however many gradients we are told to iterate
        for _ in range(it):
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

        # get data based on name thanks to the magic of getattr
        parameters = {}
        for i in parm_names:
            parameters[i] = getattr(self, i)
        # call optimiser to calculate probably better weights
        update = self.optimiser.optimise(parms=parameters, grads=gradients)
        # use update dictionary to grab the new weights and set what we want
        for key, value in update.items():
            setattr(self, key, value)

    # # # Abstract Methods
    # These abstract methods are intended to notify node implementers of any
    # required functions since they will be extensiveley used in the
    # computational graph, and will error if un-populated from subclasses
    @abc.abstractmethod
    def forward(self, x):
        """Calculate forward pass for singular example."""

    @abc.abstractmethod
    def backward(self, gradient):
        """Calculate backward pass for singular example."""

    def forwards(self, xs):
        """Calculate forward pass for multiple examples simultaneously."""
        accumulator = []
        for i in xs:
            accumulator.append(self.forward(x=i))
        return accumulator

    def backwards(self, gradients):
        """Calculate backward pass for multiple examples simultaneously."""
        accumulator = []
        for i in gradients:
            accumulator.append(self.backward(gradient=i))
        return accumulator

    @abc.abstractmethod
    def update(self):
        """Update node state/ weights for a single example."""

    @abc.abstractmethod
    def updates(self):
        """Update node state/ weights for multiple examples simultaneously."""

    @property
    @abc.abstractmethod
    def cost(self):
        """Get the computational cost per forward example of the node."""


class IO(Node):
    """An input output node that is primarily used to link and join nodes."""

    def forward(self, x):
        """Pass input directly to output."""
        return x

    def backward(self, gradient):
        """Pass gradient directly to output."""
        return gradient

    def update(self):
        """Do nothing."""

    def updates(self):
        """Do nothing."""

    @property
    def cost(self):
        """Get no cost of a this transitional node."""
