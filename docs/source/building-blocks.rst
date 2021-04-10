.. include:: substitutions

Building Blocks
===============

.. note::

  |api-build-note|

There are several main types of neural network components we implement. This page serves as a quick reference to each of these.

|section_blocks|
++++++++++++++++

|section_blocks| are the base building blocks which all else inherits. These implement the most basic shared functionality between all neural network components.

|section_activations|
+++++++++++++++++++++

|section_activations| is the base for all activation functions.

|section_layers|
++++++++++++++++

|section_layers| is the base for all pure-network (non-activation) layers such as the core forward backward and update pass of ANNs, CNNs, and whatever other networks we create.

|section_optimizers|
++++++++++++++++++++

|section_optimizers| is the base for all more complex optimizers to update gradients in the desired/ expected manner.
