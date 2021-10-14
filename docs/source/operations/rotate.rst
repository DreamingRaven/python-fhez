.. include:: /substitutions

.. _section_rotate:

Rotate
#######

Rotate nodes have 3 modes of operation, an encryptor, decryptor, or rotator. Rotate.forward() can, depending on configuration, transform data plaintext->cyphertext, cyphertext->cyphertext, cyphertext->plaintext, plaintext->plaintext. Rotate always keeps the original data shape, but can change which axis is encrypted. For instance a single cyphertext of shape (10,32,32) can come out as a list of (10,) cyphertexts of shape (32,32).

- node provider/ encryptor non-configured: input will be converted to plaintext, thus cyphertext->plaintext, and plaintext->plaintext mode
- node provider/ encryptor configured: input will be encrypted with new keys based on given axis, thus plaintext->cyphertext, cyphertext->cyphertext2

This node can thus be used as a replacement for both encrypt and decrypt nodes. The only caveat being those nodes being more specialised give you slightly more warnings. For instance if an encryption node if left unconfigured it makes sense to warn you that it will be operating in plaintext. However if this node is left unconfigured it may be the case that you want it to be a decryption node, thus we don't warn you.

Rotate API
----------

.. automodule:: fhez.nn.operations.rotate
  :members:
