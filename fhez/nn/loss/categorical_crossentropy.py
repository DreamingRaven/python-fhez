# @Author: George Onoufriou <archer>
# @Date:   2021-08-02T22:04:55+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-02T22:07:16+01:00
from fhez.nn.loss.loss import Loss


class CategoricalCrossentropy(Loss):
    """Categorical Cross Entropy for Multi-Class Multi-Label problems."""
