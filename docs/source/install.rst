.. pyrtd documentation master file, created by
   sphinx-quickstart on Mon Aug 26 13:30:29 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Installation
############

Currently due to a certain amount of awkwardness of the python to Microsoft-SEAL bindings, the only "supported" installation method is by docker container.
However we are working on fixing some of these issues in the near future, or at the very least supporting a second pure-python backend so that installation is easier as a native library if that is desired.

Docker Registry
+++++++++++++++

Dockerised images that you can play around with are available at: https://gitlab.com/GeorgeRaven/python-reseal/container_registry/1756780

E.G, download and run interactively:

.. code-block:: bash

  docker run -it registry.gitlab.com/georgeraven/python-reseal:master

.. _section_docker_build:

Docker Build
++++++++++++

If you would rather build this library locally/ for security reasons, you can issue the following build command from inside the projects root directory:

.. code-block:: bash

  docker build -t archer/fhe -f Dockerfile_archlinux .

.. note::

  The build process could take several minutes to build, as it will compile, bind, and package the Microsoft-SEAL library in with itself.

To build the detailed documentation with all the autodoc functionality for the core classes, you can use your now build container to create these docs for you by:

.. code-block:: bash

  docker run -v ${PWD}/docs/build/:/python-fhe/docs/build -it archer/fhe make -C /python-fhe/docs html

The docs will then be available in ``${PWD}/docs/build/html`` for your viewing pleasure

To run the now locally built container you can issue the following command to gain interactive access:

.. code-block:: bash

  docker run -it archer/fhe
