
Categorical Cross Entropy (CCE)
###############################


.. |classification-network-fig| image:: /img/classification_network.svg
  :width: 800
  :alt: Neural network diagram showing the anatomy of a generic classification network

.. |cce-graph-fig| image:: /img/cce.svg
  :width: 800
  :alt: Graph showing categorical cross entropy plotted on graph. Where green is the target value, orange is the predicted value, red is the output of CCE, and blue is the local gradient of CCE.

Example Architecture
--------------------

This figure shows a generic classification network, and how the CCE is likeley to be used.

|classification-network-fig|

Graph
-----

The Graph here shows categorical cross entropy plotted on x and y axis. Where green is the target value, orange is the predicted value, red is the output of CCE, and blue is the local gradient of CCE.

See  https://www.desmos.com/calculator/q2dwniwjsp for an interactive version.

|cce-graph-fig|

CCE API
-------

.. autoclass:: fhez.nn.loss.categorical_crossentropy.CategoricalCrossentropy
  :members:
