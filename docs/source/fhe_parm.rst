.. include:: substitutions

.. _section_fhe_parm:

FHE Parameterisation
====================

.. |fhe_param_problem| image:: /img/fhe_param_problem.svg
  :width: 800
  :alt: Neural network graph example showing source and sink nodes, and the computational depth from each to each that needs to be considered to parameterise the FHE nodes.

FHE nodes need to be parameterised to be properly capable of computing the source-to-sink computational distance in the network graph. If the FHE parameters do not match or at-least meet the minimum requirements, the cyphertext will be garbled. If no parameters are given plaintext is assumed, so please do parameterise the nodes.

|fhe_param_problem|

Sources are where data is encrypted. Sinks are where cyphertexts are consumed/ decrypted. Compute nodes are just generic data processors that work with both cyphertexts and plaintexts. Rotation nodes are either explicit bootstrapping nodes, or deliberate key-rotation operations to refresh the keys or reform how the data is encrypted e.g from a single cyphertext to split into many smaller cyphertexts.

To help automate parameterisation we have some useful utilities that you may be interested in:

.. automodule:: fhez.nn.parametrisation.autofhe
  :members:
