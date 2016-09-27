.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :target: https://github.com/vicolab/ml-pyxis/blob/master/LICENSE

========
ml-pyxis
========

Tool for reading and writing datasets of tensors in a Lightning Memory-Mapped Database (LMDB).


Installation
============

The installation instructions are generic and should work on most operating systems that support the prerequisites.

``ml-pyxis`` requires Python version 2.7, 3.4, or above. We recommend installing ``ml-pyxis``, as well as all prerequisites, in a virtual environment via `virtualenv`_.


-------------
Prerequisites
-------------

The following Python packages are required to run ``ml-pyxis``:

* `NumPy`_ - N-dimensional array object and tools for operating on them
* `lmdb`_ - Universal Python binding for the `LMDB 'Lightning' Database`_

Please refer to the individual packages for more information about additional dependencies and how to install them for your operating system.


--------------------------
Bleeding-edge installation
--------------------------

To install the latest version of ``ml-pyxis``, run the following command:

.. code-block:: bash

  pip install --upgrade https://github.com/vicolab/ml-pyxis/archive/master.zip

Add the ``--user`` tag if you want to install the package in your home directory.


------------------------
Development installation
------------------------

``ml-pyxis`` can be installed from source in such a way that any changes to your local copy will take effect without having to reinstall the package. Start by making a copy of the repository:

.. code-block:: bash

  git clone https://github.com/vicolab/ml-pyxis.git

Next, enter the directory and install ``ml-pyxis`` in development mode by issuing the following command:

.. code-block:: bash

  cd ml-pyxis
  python setup.py develop


.. Links

.. _virtualenv: https://virtualenv.pypa.io/en/stable/
.. _NumPy: http://www.numpy.org/
.. _lmdb: http://lmdb.readthedocs.io/en/release/
.. _LMDB 'Lightning' Database: https://symas.com/products/lightning-memory-mapped-database/
