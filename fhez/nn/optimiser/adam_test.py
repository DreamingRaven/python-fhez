# @Author: George Onoufriou <archer>
# @Date:   2021-07-27T14:02:55+01:00
# @Last modified by:   archer
# @Last modified time: 2021-07-30T16:03:22+01:00

import time
import unittest
import numpy as np

from fhez.nn.optimiser.adam import Adam
from fhez.nn.activation.linear import Linear
from fhez.nn.loss.mae import MSE
import copy


class AdamTest(unittest.TestCase):
    """Test Adaptive Moment optimizer."""

    def setUp(self):
        """Start timer and init variables."""
        self.start_time = time.time()

    def tearDown(self):
        """Calculate and print time delta."""
        t = time.time() - self.start_time
        print('%s: %.3f' % (self.id(), t))

    def linear(self, x, m, c):
        """Calculate standard linear function for testing against."""
        return (m * x) + c

    def test_init(self):
        """Check Adam can be initialised using defaults."""
        optimiser = Adam()
        self.assertIsInstance(optimiser, Adam)
        self.assertIsInstance(optimiser.cache, dict)
        self.assertIsInstance(optimiser.alpha, float)
        self.assertIsInstance(optimiser.beta_1, float)
        self.assertIsInstance(optimiser.beta_2, float)
        self.assertIsInstance(optimiser.epsilon, float)

    # def descend(self, truth, parms, optimiser, x=2, it=5):
    #     """Use given optimiser with linear to descend."""
    #     parameters = copy.deepcopy(parms)
    #     for _ in range(it):
    #         # calculate linear result
    #         y_hat = self.linear(x=x, m=parameters["m"], c=parameters["c"])
    #         # calculate desired result
    #         y = self.linear(x=x, m=truth["m"], c=truth["c"])
    #         loss = mae(y=y, y_hat=y_hat)
    #         mae_grad = 1 if y_hat > y else -1
    #         # print(loss)
    #         gradients = {
    #             "dfdm": x * mae_grad * loss,
    #             "dfdc": 1 * mae_grad * loss,
    #         }
    #         update = optimiser.optimise(parms=parameters, grads=gradients)
    #         self.assertIsInstance(update, dict)
    #         # check keys all still exist
    #         self.assertEqual(update.keys(), parameters.keys())
    #         # check there has been some update/ change that they are different
    #         self.assertNotEqual(update, parameters)
    #         parameters = update
    #     return parameters
    #
    # def get_loss(self, parameters, truth, x):
    #     # calculate linear result
    #     y_hat = self.linear(x=x, m=parameters["m"], c=parameters["c"])
    #     # calculate desired result
    #     y = self.linear(x=x, m=truth["m"], c=truth["c"])
    #     loss = mae(y=y, y_hat=y_hat)
    #     return loss

    @property
    def optimiser(self):
        return Adam()

    @property
    def x(self):
        return 2

    @property
    def nn(self):
        return Linear

    @property
    def lossfunc(self):
        return MSE

    def test_optimise(self):
        optimiser = self.optimiser
        x = self.x
        lossfunc = self.lossfunc()
        parameters = {
            "m": 0.4,
            "c": 0.5
        }
        truth = {
            "m": 0.9,
            "c": 0.1,
        }
        nn = self.nn(**parameters)
        nn_optimal = self.nn(**truth)

        # get predicted and optimal output
        y_hat = nn.forward(x)
        y = nn_optimal.forward(x)

        # calculate the loss and gradient with respect to y_hat
        loss = lossfunc.forward(y=y, y_hat=y_hat)
        dloss_y_hat = lossfunc.backward(loss)
        # TODO: calculate backprop of loss function
        # TODO apply chain rule to loss function backprop to update wieghts
        raise NotImplementedError("This function is not complete.")
        # chain rule effect of parameters on y_hat

    # def test_update_linear(self):
    #     """Check adam update/ optimisation."""
    #     optimiser = Adam()
    #     x = 2
    #     parameters = {
    #         # "m": 2,
    #         # "c": 3,
    #         "m": 0.4,
    #         "c": 0.5,
    #     }
    #     truth = {
    #         "m": 0.9,
    #         "c": 0.1,
    #     }
    #     original_loss = self.get_loss(parameters=parameters, truth=truth, x=x)
    #     # print(parameters)
    #     parameters = self.descend(optimiser=optimiser,
    #                               truth=truth,
    #                               parms=parameters, x=x)
    #     second_loss = self.get_loss(parameters=parameters, truth=truth, x=x)
    #     self.assertLess(second_loss, original_loss)
    #
    #     # now reverse see if we still descend
    #     truth = {
    #         "m": 0.1,
    #         "c": 0.9,
    #     }
    #     third_loss = self.get_loss(parameters=parameters, truth=truth, x=x)
    #     # print(parameters)
    #     parameters = self.descend(optimiser=optimiser,
    #
    #                               truth=truth,
    #                               parms=parameters, x=x)
    #     fourth_loss = self.get_loss(parameters=parameters, truth=truth, x=x)
    #     self.assertLess(fourth_loss, third_loss)
    #     # print(parameters)
    #     # print(original_loss, second_loss, third_loss, fourth_loss)

    def test_momentum(self):
        """Check Adam 1st moment operating properly, and updating vars."""
        # expresley setting variables so we can KNOW and answer to verify out
        beta_1 = 0.9
        optimiser = Adam(alpha=0.001,
                         beta_1=beta_1,
                         beta_2=0.999,
                         epsilon=1e-8)
        x = 1
        parameters = {
            "m": 2,
            "c": 3,
        }
        truth = {
            "m": 6,
            "c": 7,
        }
        # calculate linear result
        y_hat = self.linear(x=x, m=parameters["m"], c=parameters["c"])
        # calculate desired result
        y = self.linear(x=x, m=truth["m"], c=truth["c"])
        loss = y - y_hat
        gradients = {
            "dfdm": x * loss,
            "dfdc": 1 * loss,
        }
        name = "m"
        m_hat = optimiser.momentum(gradient=gradients["dfd{}".format(name)],
                                   param_name=name)

        # check that internal state has been modified properly
        self.assertEqual(optimiser.cache[name]["t_m"], 2)
        m_true = (beta_1 * 0) + (1 - beta_1) * gradients["dfd{}".format(name)]
        self.assertEqual(optimiser.cache[name]["m"], m_true)
        # check it has returned a correct value
        m_hat_true = m_true / (1 - beta_1**1)
        self.assertEqual(m_hat, m_hat_true)

    def test_rmsprop(self):
        """Check Adam 2nd moment operating properly, and updating vars."""
        # expresley setting variables so we can KNOW and answer to verify out
        beta_1 = 0.9
        beta_2 = 0.999
        optimiser = Adam(alpha=0.001,
                         beta_1=beta_1,
                         beta_2=beta_2,
                         epsilon=1e-8)
        x = 1
        parameters = {
            "m": 2,
            "c": 3,
        }
        truth = {
            "m": 6,
            "c": 7,
        }
        # calculate linear result
        y_hat = self.linear(x=x, m=parameters["m"], c=parameters["c"])
        # calculate desired result
        y = self.linear(x=x, m=truth["m"], c=truth["c"])
        loss = y - y_hat
        gradients = {
            "dfdm": x * loss,
            "dfdc": 1 * loss,
        }
        name = "m"
        v_hat = optimiser.rmsprop(gradient=gradients["dfd{}".format(name)],
                                  param_name=name)

        # check that internal state has been modified properly
        self.assertEqual(optimiser.cache[name]["t_v"], 2)
        m_true = (beta_1 * 0) + (1 - beta_2) * \
            gradients["dfd{}".format(name)] ** 2
        self.assertEqual(optimiser.cache[name]["v"], m_true)
        # check it has returned a correct value
        m_hat_true = m_true / (1 - beta_2**1)
        self.assertEqual(v_hat, m_hat_true)
