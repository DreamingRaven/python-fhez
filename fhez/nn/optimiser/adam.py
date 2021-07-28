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

    # CALCULATIONS

    def momentum(self, gradient: float, param_name: str, ord: int = 1):
        r"""Calculate momentum, of a single parameter-category/ name.

        This function can calculate either 1st order momentum or 2nd order
        momentum (rmsprop) since they are both almost identical.

        where moment is 1 (I.E first order):

        - current moment :math:`m_t = \beta_1 * m_{t-1} + (1-\beta_1) * g_t`

        - decayed moment :math:`\hat{m_t} = \frac{m_t}{1 – \beta_1^t}`

        where moment is 2 (I.E second order/ RMSprop):

        - current moment :math:`v_t = \beta_2 * v_{t-1} + (1-\beta_2) * g_t^2`

        - decayed moment :math:`\hat{v_t} = \frac{v_t}{1 – \beta_2^t}`

        Steps taken:

        - retrieve previous momentum from cache dictionary using key
          (param_name) and number of iterations

        - calculate current momentum using previous momentum:

        - Save current momentum into cache dictionary using key

        - calculate current momentum correction/ decay:

        - return decayed momentum

        :arg gradient: gradient at current timestep, usually minibatch
        :arg param_name: key used to look up parameters in m_t dictionary
        :arg ord: the order of momentum to calculate defaults to 1
        :type gradient: float
        :type param_name: str
        :type ord: int
        :return: :math:`\hat{m_t}` corrected/ averaged momentum of order ord
        :rtype: float
        :example: Adam().momentum(gradient=100, param_name="w", ord=1)
        """
        # sanity check to ensure key in dictionary
        if self.cache.get(param_name) is None:
            self.cache[param_name] = {}
        # retrieve number of iterations
        i = self.cache[param_name].get("t_m") if ord == 1 else \
            self.cache[param_name].get("t_v")
        i = i if i is not None else 1  # starts from 1
        # retrieve previous momentum m_{t-1}
        m_prev = self.cache[param_name].get("m") if ord == 1 else \
            self.cache[param_name].get("v")
        m_prev = m_prev if m_prev is not None else 0
        # get beta we are using here
        beta = self.beta_1 if ord == 1 else self.beta_2
        # multiply gradient if ord = 2
        gradient = gradient if ord == 1 else (gradient * gradient)

        # calculate momentum
        m_t = (beta * m_prev) + ((1-beta) * gradient)
        # calculate momentum-correction
        m_hat = m_t/(1 - beta**i)

        # save non corrected current momentum back
        self.cache[param_name]["m" if ord == 1 else "v"] = m_t
        # increment number of specific iterations of this function
        self.cache[param_name]["t_m" if ord == 1 else "t_v"] = i + 1
        # return \hat{m_t} corrected/ averaged momentum
        return m_hat

    def rmsprop(self, gradient: float, param_name: str):
        """Get second order momentum."""
        return self.momentum(gradient=gradient, param_name=param_name, ord=2)

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
        out = {}
        for key, value in parms.items():
            m_hat = self.momentum(
                param_name=key,
                gradient=grads.get("dfd{}".format(key)))
            v_hat = self.rmsprop(
                param_name=key,
                gradient=grads.get("dfd{}".format(key)))
            out[key] = value - ((self.alpha * m_hat)/(np.sqrt(v_hat) +
                                                      self.epsilon))
        return out
