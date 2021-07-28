# @Author: George Onoufriou <archer>
# @Date:   2021-07-27T14:02:55+01:00
# @Last modified by:   archer
# @Last modified time: 2021-07-28T21:45:56+01:00

import time
import unittest
import numpy as np

from fhez.nn.optimiser.adam import Adam
from fhez.rearray import ReArray as Erray  # aliasing for later adjust


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

    def test_update_linear(self):
        """Check adam update/ optimisation."""
        optimiser = Adam()
        x = 1
        parameters = {
            # "m": 2,
            # "c": 3,
            "m": 0.2,
            "c": 0.5,
        }
        truth = {
            "m": 0.9,
            "c": 0.1,
        }
        for _ in range(20):
            # calculate linear result
            y_hat = self.linear(x=x, m=parameters["m"], c=parameters["c"])
            # calculate desired result
            y = self.linear(x=x, m=truth["m"], c=truth["c"])
            loss = np.absolute(y - y_hat)
            print(loss)
            gradients = {
                "dfdm": x * loss,
                "dfdc": 1 * loss,
            }
            update = optimiser.optimise(parms=parameters, grads=gradients)
            self.assertIsInstance(update, dict)
            # check keys all still exist
            self.assertEqual(update.keys(), parameters.keys())
            # check there has been some update/ change that they are different
            self.assertNotEqual(update, parameters)
            parameters = update
            print(parameters)

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
