"""Adaptive moment optimiser, Adam."""
# @Author: George Onoufriou <archer>
# @Date:   2021-07-27T10:22:51+01:00
# @Last modified by:   archer
# @Last modified time: 2021-07-27T16:54:57+01:00

# SOURCES
# https://arxiv.org/abs/1412.6980
# https://openreview.net/pdf?id=ryQu7f-RZ
# https://www.youtube.com/watch?v=JXQT_vxqwIs&t=276s
# https://keras.io/api/optimizers/adam/

import numpy as np


class Adam():
    """Adaptive moment optimiser abstraction.

    Sources:

    - https://arxiv.org/abs/1412.6980
    - https://openreview.net/pdf?id=ryQu7f-RZ
    - https://www.youtube.com/watch?v=JXQT_vxqwIs&t=276s
    - https://keras.io/api/optimizers/adam/
    - https://machinelearningjourney.com/index.php/2021/01/09/adam-optimizer/
    - https://www.geeksforgeeks.org/intuition-of-adam-optimizer/
    """

    def __init__(self,
                 alpha: float = 0.001,
                 beta_1: float = 0.9,
                 beta_2: float = 0.999,
                 epsilon: float = 1e-8):
        """Create Adam object with defaults."""
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    # HYPERPARAMETERS

    @property
    def alpha(self):
        r"""Get learning rate hyperparameter.

        :returns: alpha :math:`\alpha`, defaults to :math:`0.001`
        :rtype: float
        """
        if self.__dict__.get("_alpha") is None:
            self._alpha = 0.001  # SHOULD BE TUNED BY USER
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float):
        """Set learning rate hyperparameter."""
        self._alpha = alpha

    @property
    def beta_1(self):
        r"""Get first order moment exponential decay rate.

        :returns: beta_1 :math:`\beta_1`, defaults to :math:`0.9`
        :rtype: float
        """
        if self.__dict__.get("_beta_1") is None:
            self._beta_1 = 0.9  # standard beta_1 default
        return self._beta_1

    @beta_1.setter
    def beta_1(self, beta_1: float):
        """Set first order moment exponential decay rate."""
        self._beta_1 = beta_1

    @property
    def beta_2(self):
        r"""Get second order moment exponential decay rate.

        :returns: beta_2 :math:`\beta_2`, defaults to :math:`0.999`
        :rtype: float
        """
        if self.__dict__.get("_beta_2") is None:
            self._beta_2 = 0.999  # standard beta_2 default
        return self._beta_2

    @beta_2.setter
    def beta_2(self, beta_2: float):
        """Set second order moment exponential decay rate."""
        self._beta_2 = beta_2

    @property
    def epsilon(self):
        r"""Get epsilon.

        :returns:

            epsilon :math:`\epsilon` (not :math:`\varepsilon`), defaults to
            :math:`1e^{-8}`

        :rtype: float
        """
        if self.__dict__.get("_epsilon") is None:
            self._epsilon = 1e-8  # standard epsilon default
        return self._epsilon

    @epsilon.setter
    def epsilon(self, epsilon: float):
        r"""epsilon :math:`\epsilon` smoothing term.

        :math:`\epsilon` is meant to smooth and prevent division by zero.
        """
        self._epsilon = epsilon

    # Other Properties

    @property
    def V_d(self):
        """Get dictionary key (name), value (array) of parameters."""
        if self.__dict__.get("_V_d") is None:
            self._V_d = {}
        return self._V_d

    @V_d.setter
    def V_d(self, V_d):
        """."""
        self._V_d = V_d

    @property
    def S_d(self):
        """Get dictionary key (name), value (array) of parameters."""
        if self.__dict__.get("_S_d") is None:
            self._S_d = {}
        return self._S_d

    @S_d.setter
    def S_d(self, S_d):
        """."""
        self._S_d = S_d

    # CALCULATIONS

    def momentum(self):
        r"""Calculate momentum, and correction.

        calculate momentum
        :math:`m = \beta_1m \; – (1-\beta_1) \nabla_\theta J(\theta)`

        calculate momentum correction/ decay
        :math:`\hat{m} = \frac{m}{1 – \beta_1^t}`
        """
        # calculate momentum
        # calculate momentum correction

    def rmsprop(self):
        r"""Calculate rmsprop, and correction.

        calculate rmsprop
        :math:`s = \beta_2 s + (1-\beta_2)
        \nabla_\theta J(\theta) \odot \nabla_\theta J(\theta)`

        calculate rmsprop correction/ decay
        :math:`\hat{s} = \frac{s}{1 – \beta_2^t}`
        """
        # calculate rmsprop
        # calculate rmsprop correction

    def optimise(self, parms: dict, grads: dict):
        """Update given params based on gradients using Adam.

        Params and grads keys are expected to be `x` and `dfdx` respectiveley.
        They should match although the x in this case should re replaced by
        any uniquely identifying string sequence.
        :arg parms: Dictionary of keys (param name), values (param value)
        :type parms: dict[str, float]
        :arg grads: Dictionary of keys (param name), values (param gradient)
        :type grads: dict[str, float]
        :return: Dictionary of keys (param name), values (proposed new value)
        :rtype: dict[str, float]
        :example: Adam().optimise({"b": 1},{"dfdb": 200})
        """
        for key, value in parms.items():
            print(grads.get("dfd{}".format(key)))
        return parms
