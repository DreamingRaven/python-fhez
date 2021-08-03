.. include:: substitutions

.. _section_activations:

Activations
===============

Neural network activation functions are the primary logic that dictates whether a given neuron should be activated or not, and by how much. Activation functions are best when they introduce some non-linearity to the neuron so that it can better model more complex behaviors.

In our case all activations are children of the Node abstraction to keep things consistent, and reduce the amount of redundant code.

.. toctree::
  :glob:
  :maxdepth: 3
  :caption: Activation Functions

  /activations/*
