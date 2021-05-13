#!/usr/bin/env python3

# @Author: GeorgeRaven <archer>
# @Date:   2021-02-22T11:46:18+00:00
# @Last modified by:   archer
# @Last modified time: 2021-05-13T13:32:50+01:00
# @License: please see LICENSE file in project root
import numpy as np
from fhe.nn.activation.activation import Activation
import logging as logger
from tqdm import tqdm
import unittest


class ReLU_Approximation(Activation):

    @Activation.fwd
    def forward(self, x):
        if self.__dict__.get("q"):
            pass
        else:
            self.q = 1
        q = self.q
        # https://www.researchgate.net/publication/345756894_On_Polynomial_Approximations_for_Privacy-Preserving_and_Verifiable_ReLU_Networks
        #
        return ((4/(3*np.pi*q)) * (x * x)) + (0.5*x) + (q/(3*np.pi))

    @Activation.bwd
    def backward(self, gradient: np.array, x: np.array):
        """Calculate gradient with respect to x, given forward gradient.

        x should be a single cached input that was previously activated.
        gradient should be a single gradient with which to backpropogate with
        """
        # calculate local gradient but using normal sigmoid derivative
        # as this is approximate and is faster this way
        # \frac{d\sigma}{dx} = (1-\sigma(x))\sigma(x)

        if len(gradient.shape) > 1:
            raise ValueError(
                "gradient:{}, with shape: {} has too many dimensions".format(
                    gradient, gradient.shape
                ))

        df_dbatch_accumulator = []
        # iterate through each batch and calculate the per batch gradient
        for i in tqdm(range(len(x)), desc="{}.backward.batch".format(
                self.__class__.__name__),
                position=1, leave=False, ncols=80, colour="green"
        ):
            # all values of x should be summed if they arent already a single
            # value (may be the case commuting summation)
            batch = np.sum(x[i])
            # df_dbatch = (1 - self.sigmoid(batch)) * self.sigmoid(batch) * \
            #     gradient[i]
            df_dbatch = ((3/(3*np.pi*self.q))+0.5) * gradient[i]
            df_dbatch_accumulator.append(df_dbatch)
        df_dx = np.array(df_dbatch_accumulator)
        # return shape should be (num_batches,) which are later accumulated
        return df_dx

    def update(self):
        # new_parameter = old_parameter - learning_rate * gradient_of_parameter
        raise NotImplementedError(
            "Sigmoid approximation has no parameters to update")

    def sigmoid(self, x):
        return (1/(1+np.exp(-x)))


class Relu_Test(unittest.TestCase):

    @ property
    def x(self):
        x = [
            [0.0, 0.2, 0.3],
            [0.8, 0.9, 1.0]
        ]
        return np.array(x)

    def test_forward(self):
        print(self.x.shape)
        activation_function = ReLU_Approximation()
        activation = activation_function.forward(self.x)
        truth = activation_function.sigmoid(self.x)
        print(activation)
        print(truth)

    def test_backward(self):
        activation_function = ReLU_Approximation()
        activation = activation_function.forward(self.x)
        df_dx = activation_function.backward()
        print(activation.shape)
        print(df_dx.shape)

    def test_update(self):
        pass


if __name__ == "__main__":
    logger.basicConfig(  # filename="{}.log".format(__file__),
        level=logger.INFO,
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S")
    # run all the unit-tests
    print("now testing:", __file__, "...")
    unittest.main()
