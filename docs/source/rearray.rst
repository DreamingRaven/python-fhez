.. include:: substitutions

ReArray
#######

.. note::

  You may see some or no content at all on this page that documents the API. If that is the case please build the documentation locally (|section_docker_build|), as we have yet to find a good way to build these complex libraries on RTDs servers.

ReArray is a second level abstraction that uses some backend like ReSeal and packages it to be able to handle ndimensional operations, batches, and cross batch operations, while conforming to a custom numpy container implementation, making FHE compatible with numpy.

.. autoclass:: fhe.rearray.ReArray
  :members:
