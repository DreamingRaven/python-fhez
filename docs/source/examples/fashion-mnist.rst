.. include:: /substitutions

.. _section_fashion_mnist:

Fashion-MNIST
#########################

.. warning:

  This example is still in progress!

This doubles as an example and as reproducible code to get our results from one of our (soon to be) published papers. This example will download Fashion-MNIST (a drop in more complex replacement for standard MNIST) in CSV format, normalise it, and begin the training process.

.. warning:

  While we do our best to keep resource requirements low this Jupyter script is extremely RAM intensive, it will chew through less than 32GB of RAM like a meteor did to the dinosaurs.

Interactive Graph
-----------------

The following **interactive graph** represents the neural network used for MNIST prediction to exemplify this library in use:

.. raw:: html
  :file: ../img/mnist-nn-graph.html

Usage
-----

To run this example please use the generic docker script. Then the fashion-MNIST example is in the file fashion-mnist.ipynb which you can open in the browsers jupyter lab session (use the last of the three links).
