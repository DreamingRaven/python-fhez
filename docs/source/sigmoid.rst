.. pyrtd documentation master file, created by
   sphinx-quickstart on Mon Aug 26 13:30:29 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Sigmoid Approximation
#####################

(Andrej Karpathy CS231n lecture https://youtu.be/i94OvYb6noo?t=1714)

.. math::
  :label: sigmoid

  \sigma{x}=\frac{1}{1+e^{-x}}

  \frac{d\sigma(x)}{dx} = \frac{e^{-x}}{(1+e^{-x})^2} = (\frac{1+e^{-x}-1}{1+e^{-x}})(\frac{1}{1+e^{-x}}) = (1-\sigma(x))\sigma(x)

https://eprint.iacr.org/2018/462

.. math::
  :label: sigmoid-approx

  0.5 + 0.197x + -0.004x^3 \approx \sigma(x)
