.. include:: substitutions

.. _section_docs:

Documentation Variations
========================

This documentation is hosted using different branches/ versions, and to different extents depending on the provider, to ensure maximum availability, and specificity so you are never left without documentation.

.. note::

  We use sphinx autodoc to document our API, however because some base libraries are still new and are only really available in dockerland this auto-documentation is not easily implementable on all providers. If you would like to have this auto-documentation please choose a version of the docs with **autodoc**.

The current variations of the documentation are as follows:

.. |pages-latest| image:: https://readthedocs.org/projects/python-fhez/badge/?version=latest
  :target: https://deepcypher.gitlab.io/python-fhez
  :alt: GitLab Pages Documentation Status (Latest/Master)

.. |rtd-latest| image:: https://readthedocs.org/projects/python-fhez/badge/?version=latest
  :target: https://python-fhez.readthedocs.io/en/latest/?badge=latest
  :alt: RTD Documentation Status (Latest/Master)

.. |rtd-stable| image:: https://readthedocs.org/projects/python-fhez/badge/?version=stable
  :target: https://python-fhez.readthedocs.io/en/stable/?badge=stable
  :alt: RTD Documentation Status (Latest Stable Release)

.. |rtd-staging| image:: https://readthedocs.org/projects/python-fhez/badge/?version=staging
  :target: https://python-fhez.readthedocs.io/en/staging/?badge=staging
  :alt: RTD Documentation Status (Staging)

.. |rtd-dev| image:: https://readthedocs.org/projects/python-fhez/badge/?version=dev
  :target: https://python-fhez.readthedocs.io/en/dev/?badge=dev
  :alt: RTD Documentation Status (Staging)

.. _latest: https://gitlab.com/deepcypher/python-fhez
.. |latest| replace:: Master

.. _staging: https://gitlab.com/deepcypher/python-fhez/-/tree/staging
.. |staging| replace:: Staging

.. _dev: https://gitlab.com/deepcypher/python-fhez/-/tree/dev
.. |dev| replace:: Dev

.. _stable: https://gitlab.com/deepcypher/python-fhez/-/tags
.. |stable| replace:: Stable

.. list-table::
    :widths: 25 25 30
    :header-rows: 1

    * - Version
      - Provider
      - Link to Documentation
    * - |latest|_
      - |rtd|_
      - |rtd-latest|
    * - |stable|_
      - |rtd|_
      - |rtd-stable|
    * - |staging|_
      - |rtd|_
      - |rtd-staging|
    * - |dev|_
      - |rtd|_
      - |rtd-dev|
    * - |staging|_ (Autodoc)
      - `GitLab Pages <https://docs.gitlab.com/ee/user/project/pages/>`_
      - |pages-latest|
