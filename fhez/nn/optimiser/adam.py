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
        r"""Epsilon :math:`\epsilon` smoothing term.

        :math:`\epsilon` is meant to smooth and prevent division by zero.
        """
        self._epsilon = epsilon

    # Other Properties

    @property
    def cache(self):
        """Cache of iteration specific values.

        This cache is a dictionary of keys (the parameter name) and values
        (the parameter specific variables). For example in this cache you can
        expect to get the previous iterations moment, and number of iterations.
        """
        if self.__dict__.get("_cache") is None:
            self._cache = {}
        return self._cache

    @cache.setter
    def cache(self, cache):
        self._cache = cache

    @property
    def m_t(self):
        """Biased first moment vector."""
        if self.__dict__.get("_m_t") is None:
            self._m_t = {}
        return self._m_t

    @m_t.setter
    def m_t(self, m_t):
        """."""
        self._m_t = m_t

    @property
    def v_t(self):
        """Biased second raw moment vector."""
        if self.__dict__.get("_v_t") is None:
            self._v_t = {}
        return self._v_t

    @v_t.setter
    def v_t(self, v_t):
        """."""
        self._v_t = v_t

    # CALCULATIONS

    def momentum(self, gradient: float, param_name: str):
        r"""Calculate momentum, of a single parameter-category/ name.

        - retrieve previous momentum from cache dictionary using key
          (param_name) and number of iterations

        - calculate current momentum using previous:
          :math:`m_t = \beta_1 * m_{t-1} + (1-\beta_1) * g_t`

        - Save current momentum into cache dictionary using key

        - calculate current momentum correction/ decay:
          :math:`\hat{m_t} = \frac{m_t}{1 – \beta_1^t}`

        :arg gradient: gradient at current timestep, usually minibatch
        :arg param_name: key used to look up parameters in m_t dictionary
        :arg t: current iteration
        :type gradient: float
        :type param_name: str
        :return: :math:`\hat{m_t}` corrected/ averaged momentum
        :rtype: float
        :example: Adam().momentum(gradient=100, param_name="w")
        """
        # sanity check to ensure key in dictionary
        if self.cache.get(param_name) is None:
            self.cache[param_name] = {}
        # retrieve number of iterations
        i = self.cache[param_name].get("t_m")
        i = i if i is not None else 1  # starts from 1
        # retrieve previous momentum m_{t-1}
        m_prev = self.cache[param_name].get("m")
        m_prev = m_prev if m_prev is not None else 0

        # calculate momentum
        m_t = (self.beta_1 * m_prev) + ((1-self.beta_1) * gradient)
        # calculate momentum-correction
        m_hat = m_t/(1 - self.beta_1**i)

        # save non corrected current momentum back
        self.cache[param_name]["m"] = m_t
        # increment number of specific iterations of this function
        self.cache[param_name]["t_m"] = i + 1
        # return \hat{m_t} corrected/ averaged momentum
        return m_hat

    def rmsprop(self, gradient: float, param_name: str):
        r"""Calculate momentum, of a single parameter-category/ name.

        - retrieve previous momentum from cache dictionary using key
          (param_name) and number of iterations

        - calculate current momentum using previous:
          :math:`m_t = \beta_1 * m_{t-1} + (1-\beta_1) * g_t`

        - Save current momentum into cache dictionary using key

        - calculate current momentum correction/ decay:
          :math:`\hat{m_t} = \frac{m_t}{1 – \beta_1^t}`

        :arg gradient: gradient at current timestep, usually minibatch
        :arg param_name: key used to look up parameters in m_t dictionary
        :arg t: current iteration
        :type gradient: float
        :type param_name: str
        :return: :math:`\hat{m_t}` corrected/ averaged momentum
        :rtype: float
        :example: Adam().momentum(gradient=100, param_name="w")
        """
        # calculate rmsprop
        # calculate rmsprop correction
        return None

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
