.. include:: substitutions

Installation
############

Currently due to a certain amount of awkwardness of the python to Microsoft-SEAL bindings, the only "supported" installation method is by docker container.
However we are working on fixing some of these issues in the near future, or at the very least supporting a second pure-python back-end so that installation is easier as a native library if that is desired.

Docker Registry
+++++++++++++++

Dockerised images that you can play around with are available at: https://gitlab.com/deepcypher/python-fhez/container_registry/2063426

E.G, download and run interactively:

.. code-block:: bash

  docker run -it registry.gitlab.com/deepcypher/python-fhez:master

.. _section_docker_build:

Docker Build
++++++++++++

If you would rather build this library locally/ for security reasons, you can issue the following build command from inside the projects root directory:

.. code-block:: bash

  docker build -t archer/fhe -f Dockerfile_archlinux .

.. note::

  The build process could take several minutes to build, as it will compile, bind, and package the Microsoft-SEAL library in with itself.

Docker Documentation Build
--------------------------

To build the detailed documentation with all the autodoc functionality for the core classes, you can use your now built container to create these docs for you by:

.. code-block:: bash

  docker run -v ${PWD}/docs/build/:/python-fhe/docs/build -it archer/fhe make -C /python-fhe/docs html

The docs will then be available in ``${PWD}/docs/build/html`` for your viewing pleasure

Locally-Built Docker-Image Interactive Usage
--------------------------------------------

To run the now locally built container you can issue the following command to gain interactive access, by selecting the tag (-t) that you named it previously (here it is archer/fhe):

.. code-block:: bash

  docker run -it archer/fhe

FHE Modules/ Plugins
++++++++++++++++++++

This library will support extensions that expose a uniform API which we call "errays" or encrypted-arrays much like numpy custom containers, of which as simple implementation can be achieved by inheriting from our erray class. We hope to add more modules over time and would welcome others implementations too. We ourselves support the following modules:

- `Microsoft SEAL <https://github.com/Microsoft/SEAL>`_ (`beta <https://gitlab.com/deepcypher/python-fhez-seal>`_)\*
- `py-fhe <https://github.com/sarojaerabelli/py-fhe>`_ (`Saroja Erabelli <https://github.com/sarojaerabelli>`_) (`alpha <https://gitlab.com/deepcypher/python-fhez-erabelli>`_)

\* Currently built in but being separated.

.. note::

  More instructions to follow on specific installation of modules once implementation is complete for them.

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
