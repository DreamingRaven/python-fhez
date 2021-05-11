.. pyrtd documentation master file, created by
   sphinx-quickstart on Mon Aug 26 13:30:29 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. include:: substitutions

.. |neuron-cg-fig| image:: img/neuron-computational-graph.svg
  :width: 400
  :alt: Single neuron represented as computational graph of inputs and outputs

python-reseal
=============

Python-ReSeal is a privacy-preserving |fhe| (FHE) and deep learning library.

|neuron-cg-fig|

This library is capable of both fully homomorphically encrypting data and processing encrypted cyphertexts without the private keys, thus completely privately.

This library also supports:

- advanced serialization of cyphertext objects, for transmission over networks
- The |section_masquerade|
- The |section_commuted_sum|
- Single-cyphertext-multiple-values en-mass/ simultaneous processing via |section_hadmard_product|

Cite
++++

Either:

.. code-block:: latex

  @online{reseal,
    author = {George Onoufriou},
    title = {Python-Reseal Source Repository},
    year = {2021},
    url = {https://gitlab.com/GeorgeRaven/python-reseal},
  }

Or if you do not have @online support:


.. code-block:: latex

  @misc{reseal,
    author = {George Onoufriou},
    title = {Python-Reseal Source Repository},
    howpublished = {Github, GitLab},
    year = {2021},
    note = {\url{https://gitlab.com/GeorgeRaven/python-reseal}},
  }

.. toctree::
  :maxdepth: 1
  :caption: Table of Contents
  :numbered:

  license
  fhe
  install
  ann
  cnn
  sigmoid
  reseal
  rearray
  graph
  dll
  node
  building-blocks
  blocks
  layers
  activations
  optimizers
