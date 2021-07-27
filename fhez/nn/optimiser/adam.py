"""Adaptive moment optimiser, Adam."""
# @Author: George Onoufriou <archer>
# @Date:   2021-07-27T10:22:51+01:00
# @Last modified by:   archer
# @Last modified time: 2021-07-27T11:44:31+01:00

# SOURCES
# https://arxiv.org/abs/1412.6980
# https://openreview.net/pdf?id=ryQu7f-RZ
# https://www.youtube.com/watch?v=JXQT_vxqwIs&t=276s
# https://keras.io/api/optimizers/adam/


class Adam():
    """Adaptive moment optimiser abstraction.

    Sources:

    - https://arxiv.org/abs/1412.6980
    - https://openreview.net/pdf?id=ryQu7f-RZ
    - https://www.youtube.com/watch?v=JXQT_vxqwIs&t=276s
    - https://keras.io/api/optimizers/adam/
    """

    # HYPERPARAMETERS

    @property
    def alpha(self):
        """Get learning rate hyperparameter."""
        if self.__dict__.get("_alpha") is None:
            self._alpha = 0.001  # SHOULD BE TUNED BY USER
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float):
        """Set learning rate hyperparameter."""
        self._alpha = alpha

    @property
    def beta_1(self):
        """Set first order moment exponential decay rate."""
        if self.__dict__.get("_beta_1") is None:
            self._beta_1 = 0.9  # standard beta_1 default
        return self._beta_1

    @beta_1.setter
    def beta_1(self, beta_1: float):
        """Get first order moment exponential decay rate."""
        self._beta_1 = beta_1

    @property
    def beta_2(self):
        """Get second order moment exponential decay rate."""
        if self.__dict__.get("_beta_2") is None:
            self._beta_2 = 0.999  # standard beta_2 default
        return self._beta_2

    @beta_2.setter
    def beta_2(self, beta_2: float):
        """Set second order moment exponential decay rate."""
        self._beta_2 = beta_2

    @property
    def epsilon(self):
        """Get learning rate hyperparameter."""
        if self.__dict__.get("_epsilon") is None:
            self._epsilon = 1e-08  # standard epsilon default
        return self._epsilon

    @epsilon.setter
    def epsilon(self, epsilon: float):
        """Set learning rate hyperparameter."""
        self._epsilon = epsilon

    # Other Properties

    @property
    def V_d(self):
        """Get array of parameters."""
        if self.__dict__.get("_V_d") is None:
            self._V_d = []
        return self._V_d

    @V_d.setter
    def V_d(self, V_d):
        """."""
        self._V_d = V_d

    @property
    def S_d(self):
        """Get array of parameters."""
        if self.__dict__.get("_S_d") is None:
            self._S_d = []
        return self._S_d

    @S_d.setter
    def S_d(self, S_d):
        """."""
        self._S_d = S_d

    # CALCULATIONS

    def update(self):
        pass

    def updates(self):
        pass
