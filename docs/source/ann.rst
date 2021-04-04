.. include:: substitutions

.. |ann-fig| image:: img/sigmoid.png
  :width: 400
  :alt: Graph of sigmoid plotted on 2d axes

.. |ann-derivative-fig| image:: img/sigmoid.png
  :width: 400
  :alt: Graph of sigmoid plotted on 2d axes

Fully Connected Dense Net (ANN)
###############################

.. |ann| replace:: :eq:`ann`
.. |ann-deriv| replace:: :eq:`ann-derivative`

ANN Equations
+++++++++++++++++

ANN
-------

|ann| ANN

.. math::
  :label: ann

  a = \sigma(W_1x+b_1)

|ann-fig|

ANN Derivative
------------------

|ann-deriv| ANN derivative

.. math::
  :label: ann-derivative

  \frac{d\sigma(x)}{dx} = \frac{e^{-x}}{(1+e^{-x})^2} = (\frac{1+e^{-x}-1}{1+e^{-x}})(\frac{1}{1+e^{-x}}) = (1-\sigma(x))\sigma(x)

|ann-derivative-fig|

ANN
-------

|ann| ANN

.. math::
  :label: ann

  a = \sigma(W_1x+b_1)

|ann-fig|

ANN Derivative
------------------

|ann-deriv| ANN derivative

.. math::
  :label: ann-derivative

  \frac{d\sigma(x)}{dx} = \frac{e^{-x}}{(1+e^{-x})^2} = (\frac{1+e^{-x}-1}{1+e^{-x}})(\frac{1}{1+e^{-x}}) = (1-\sigma(x))\sigma(x)

|ann-derivative-fig|
