Python-ReSeal
#############

Python-ReSeal is a fully homomorphic encryption, abstraction library, primarily focused on enabling encrypted deep learning. This library for now supports the CKKS scheme through Microsoft-SEAL being bound to python using pybind11, then we abstract all of MS-SEAL's objects into one single meta object that facilitates serialization, deserialization, encryption, addition, multiplication, etc, in an attempt to make it as streamlined and as easy to use as possible.

Cypherpunks please read the docs:

.. image:: https://readthedocs.org/projects/python-reseal/badge/?version=latest
  :target: https://python-reseal.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status,

Python-ReSeal will start to transition to gitlab at: https://gitlab.com/GeorgeRaven/python-reseal for a plethora of reasons including Ci/CD, however github will remain an up-to-date mirror.

Build It Yourself
+++++++++++++++++

If you would like to build the documentation manually, for example to auto-doc the API (which read-the-docs cannot do) then from this directory:

- build the docker container (docker must be installed, repository must be cloned, and you must be in this directory)

.. code-block:: bash

   sudo docker build -t archer/fhe -f Dockerfile_archlinux .  || exit 1

- run the docker container with volume-mount, and trigger documentation build

.. code-block:: bash

   sudo docker run -v ${PWD}/docs/build/:/python-fhe/docs/build -it archer/fhe make -C /python-fhe/docs html

- you can then find the documentation in: ``${PWD}/docs/build/html/``

The docs will walk you through the rest. Enjoy.
