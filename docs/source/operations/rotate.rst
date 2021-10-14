.. include:: /substitutions

.. _section_rotate:

Rotate
#######

Rotate nodes have 2 modes of operation. Rotate.forward can, depending on configuration, transform data plaintext->cyphertext, cyphertext->cyphertext, cyphertext->plaintext, plaintext->plaintext. Rotate always keeps the original data shape, but can change which axis is encrypted. For instance a single cyphertext of shape (10,32,32) can come out as a list of (10,) cyphertexts of shape (32,32).

- node provider/ encryptor non-configured: input will be converted to plaintext, thus cyphertext->plaintext, and plaintext->plaintext mode
- node provider/ encryptor configured: input will be encrypted with new keys based on given axis, thus plaintext->cyphertext, cyphertext->cyphertext2

Rotate API
----------

.. automodule:: fhez.nn.operations.rotate
  :members:
