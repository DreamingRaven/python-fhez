Python-ReSeal
#############

Python-ReSeal is a fully homomorphic encryption, abstraction library, primarily focused on enabling encrypted deep learning. This library for now supports the CKKS scheme through Microsoft-SEAL being bound to python using pybind11, then we abstract all of MS-SEAL's objects into one single meta object that facilitates serialization, deserialization, encryption, addition, multiplication, etc, in an attempt to make it as streamlined and as easy to use as possible.

Please read the docs:

.. image:: https://readthedocs.org/projects/pyrtd/badge/?version=latest
  :target: https://pyrtd.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status,

Python-ReSeal will start to transition to gitlab at: https://gitlab.com/GeorgeRaven/python-reseal for a plethora of reasons including Ci/CD, however github will remain an up-to-date mirror.
