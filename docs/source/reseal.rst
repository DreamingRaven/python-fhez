.. include:: substitutions

ReSeal
######

.. note::

  You may see some or no content at all on this page that documents the API. If that is the case please build the documentation locally (|section_docker_build|), as we have yet to find a good way to build these complex libraries on RTDs servers.


The core ReSeal library consists primarily of two components, the cache (ReCache), and the Microsoft Simple Encrypted Arithmetic Library (SEAL) abstraction (ReSeal).

.. autoclass:: fhe.reseal.Reseal
  :members:

.. autoclass:: fhe.reseal.ReSeal
  :members:

.. autoclass:: fhe.recache.ReCache
  :members:

.. autoclass:: fhe.rescheme.ReScheme
  :members:
