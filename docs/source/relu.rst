.. include:: substitutions

.. |relu-fig| image:: img/relu.png
  :width: 400
  :alt: Graph of relu plotted on 2d axes

.. |relu-approx-fig| image:: img/relu-approx.png
  :width: 400
  :alt: Graph of relu-approximation plotted on 2d axes, where the approximation range is -1,1

.. |relu-approx-2-fig| image:: img/relu-approx-2.png
  :width: 400
  :alt: Graph of relu-approximation plotted on 2d axes, where the approximation range is -2,2

.. |relu-derivative-fig| image:: img/relu-derivative.png
  :width: 400
  :alt: Graph of relus derivative plotted on 2d axes, showing a flat line at y=0 with a slight bump near x=0

.. |relu-approx-derivative-fig| image:: img/relu-approx-derivative.png
  :width: 400
  :alt: Graph of relu-apprximation derivative plotted on 2d axes showing significant overlap with the normal relu derivative, within the range -4 to 4 =x where the normal derivative "bumps", but this extends down to negative infinity quickly after on both sides

ReLU & Approximation
####################

.. warning::

  This activation function has an asymptote to negative :math:`y` infinity and positive y infinity outside of a very small *safe* band of input :math:`x` values. This **will** cause *nan* and extremely large numbers if you aren't especially careful and keep all values passed into this activation function within the range -4 to 4 which is its *golden* range. Think especially carefully of your *initial* weights, and whether or not they will exceed this band into the *danger* zone. See: |section_relu_approx|


.. |relu| replace:: :eq:`relu`
.. |relu-approx| replace:: :eq:`relu-approx`
.. |relu-deriv| replace:: :eq:`relu-derivative`
.. |relu-approx-deriv| replace:: :eq:`relu-approx-derivative`

To be able to use (fully homomorphically encrypted) cyphertexts with deep learning we need to ensure our activations functions are abelian compatible operations, polynomials. relu :eq:`relu` is not a polynomial, thus we approximate :eq:`relu-approx`. Similarly since we used an approximation for the forward activations we use a derivative of the relu approximation :eq:`relu-approx-derivative` for the backward pass to calculate the local gradient in hopes of descending towards the global optimum (gradient descent).


ReLU :math:`R(x)`
++++++++++++++++++++++

.. _section_relu:

:math:`\max(0,x)`
-----------------

|relu| Relu

.. math::
  :label: relu

  \sigma(x)=\frac{1}{1+e^{-x}}

|relu-fig|

.. _section_relu_derivative:

:math:`\frac{d\max(0,x)}{dx}`
-----------------------------

|relu-deriv|

.. math::
  :label: relu-derivative

  \frac{d\sigma(x)}{dx} = \frac{e^{-x}}{(1+e^{-x})^2} = (\frac{1+e^{-x}-1}{1+e^{-x}})(\frac{1}{1+e^{-x}}) = (1-\sigma(x))\sigma(x)

|relu-derivative-fig|


ReLU-Approximation :math:`R_a(x)`
+++++++++++++++++++++++++++++++++++++++++++

.. _section_relu_approximation:

:math:`\max_a(0,x)`
-------------------

|relu-approx| `relu-approximation <https://www.researchgate.net/publication/345756894_On_Polynomial_Approximations_for_Privacy-Preserving_and_Verifiable_ReLU_Networks>`_

.. math::
  :label: relu-approx

  \max(0,x) \approx \max_a(0,x) = \frac{4}{3\pi q}x^2 + \frac{1}{2}x + \frac{q}{3\pi}, where\ x \in \{q > x > -q \subset \R \}

where q is 1:

|relu-approx-fig|

where q is 2

|relu-approx-2-fig|

.. _section_relu_approximation_derivative:

:math:`\frac{d\max_a(0,x)}{dx}`
-------------------------------

|relu-approx-deriv| relu-approximation derivative

.. math::
  :label: relu-approx-derivative

  \frac{d\max(0,x)}{dx} \approx \frac{d\max_a(0,x)}{dx} = \frac{8}{3\pi q}x + \frac{1}{2}, where\ x \in \{q > x > -q \subset \R \}

|relu-approx-derivative-fig|
