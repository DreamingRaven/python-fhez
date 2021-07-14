Python-FHEz
###########

(Formerly Python-ReSeal)

Python-FHEz is a fully homomorphic encryption, abstraction library, primarily focused on enabling encrypted deep learning. This library will be able to use various back-ends and streamline them into a single generic API to try to reduce both programmatic and conceptual complexity.

Cypherpunks please read the docs:

.. _docker: https://www.docker.com/
.. |docker| replace:: Docker

.. _rtd: https://readthedocs.org/
.. |rtd| replace:: RTD

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


.. list-table:: Docs
    :widths: 25 25 30
    :header-rows: 1

    * - Version
      - Provider
      - Link to Documentation
    * - Master (Mirror)
      - `GitLab Pages <https://docs.gitlab.com/ee/user/project/pages/>`_
      - |pages-latest|
    * - Master
      - |rtd|_
      - |rtd-latest|
    * - Stable
      - |rtd|_
      - |rtd-stable|
    * - Staging
      - |rtd|_
      - |rtd-stable|
    * - Dev
      - |rtd|_
      - |rtd-dev|


Python-FHEz will start to transition to gitlab at: https://gitlab.com/DeepCypher/python-fhez for a plethora of reasons including Ci/CD, however github will remain an up-to-date mirror.

Modules
+++++++

This library supports extensions that expose a uniform API which we call "errays" or encrypted-arrays much like numpy custom containers, of which as simple implementation can be achieved by inheriting from our erray class. We hope to add more modules over time and would welcome others implementations too. We ourselves support the following modules:

- [Microsoft SEAL](https://github.com/Microsoft/SEAL) ([beta](https://gitlab.com/deepcypher/python-fhez-seal))\*
- [py-fhe](https://github.com/sarojaerabelli/py-fhe) ([Saroja Erabelli](https://github.com/sarojaerabelli)) ([alpha](https://gitlab.com/deepcypher/python-fhez-erabelli))

\* Currently built in but we are in the process of separating into its own module so excuse the potentially empty repository, this will be its final destination.

Build The Docs
++++++++++++++

If you would like to build the documentation manually, for example to auto-doc the API (which is not easy in |rtd|_ ) then from this directory:

- build the |docker|_ container (docker must be installed, repository must be cloned, and you must be in this directory)

.. code-block:: bash

   sudo docker build -t archer/fhe -f Dockerfile_archlinux .  || exit 1

- run the docker container with volume-mount, and trigger documentation build

.. code-block:: bash

   sudo docker run -v ${PWD}/docs/build/:/python-fhe/docs/build -it archer/fhe make -C /python-fhe/docs html

- you can then find the documentation in: ``${PWD}/docs/build/html/``

The docs will walk you through the rest. Enjoy.

Cite
++++

Either:

.. code-block:: latex

  @online{reseal,
    author = {George Onoufriou},
    title = {Python-FHEz Source Repository},
    howpublished = {GitLab},
    year = {2021},
    url = {https://gitlab.com/DeepCypher/python-fhez},
  }

Or if you do not have @online support:

.. code-block:: latex

  @misc{reseal,
    author = {George Onoufriou},
    title = {Python-FHEz Source Repository},
    howpublished = {GitLab},
    year = {2021},
    note = {\url{https://gitlab.com/DeepCypher/python-fhez}},
  }
