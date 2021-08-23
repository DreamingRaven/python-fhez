.. pyrtd documentation master file, created by
   sphinx-quickstart on Mon Aug 26 13:30:29 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. include:: substitutions

.. |fhe-fig| image:: img/fhe.svg
  :width: 800
  :alt: Fully Homomorphic Encryption stages: encoding, encryption, computation, decryption and decoding.

.. |dl-fig| image:: img/neuron-computational-graph.svg
  :width: 400
  :alt: Example of artificial neural networks neuron, with inputs, parameters, and activations.

.. |graph| raw:: html
  :file: img/graph.html

Python-FHEz
===========

Python-FHEz is a privacy-preserving |fhe| (FHE) and deep learning library.

Interactive graph example of an FHE compatible neural network:

|graph|


|fhe|:

|fhe-fig|

Deep Learning:

|dl-fig|

This library is capable of both fully homomorphically encrypting data and processing encrypted cyphertexts without the private keys, thus completely privately.

This library also supports:

* advanced serialization of cyphertext objects, for transmission over networks using marshmallow
* |section_masquerade|\ ing
* |section_hadmard_product| cyphertext branching

Cite
++++

Either:

.. code-block:: latex

  @online{fhez,
    author = {George Onoufriou},
    title = {Python-FHEz Source Repository},
    year = {2021},
    url = {https://gitlab.com/DeepCypher/Python-FHEz},
  }

Or if you do not have @online support:


.. code-block:: latex

  @misc{fhez,
    author = {George Onoufriou},
    title = {Python-FHEz Source Repository},
    howpublished = {Github, GitLab},
    year = {2021},
    note = {\url{https://gitlab.com/DeepCypher/Python-FHEz}},
  }

.. toctree::
  :maxdepth: 1
  :caption: Table of Contents
  :numbered:

  documentation
  license
  fhe
  install
  examples
  graph
  traversers
  layers
  activations
  optimisers
  loss
  operations
