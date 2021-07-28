.. include:: substitutions

.. _section_optimisers:

Optimisers
===============

.. note::

  |api-build-note|

Adam
++++

`Original algorithm <https://arxiv.org/abs/1412.6980>`_:

.. |adam-algo-fig| image:: img/adam.png
  :width: 400
  :alt: Adam optimiser algorithm from original paper.

|adam-algo-fig|

Our implementation API:

.. autoclass:: fhez.nn.optimiser.adam.Adam
  :members:

Gradient Descent
++++++++++++++++

.. autoclass:: fhez.nn.optimiser.gd.GD
  :members:
