.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :target: https://github.com/vicolab/ml-pyxis/blob/master/LICENSE

========
ml-pyxis
========

Tool for reading and writing datasets of tensors (``numpy.ndarray``) with
MessagePack and Lightning Memory-Mapped Database (LMDB).


Example
=======

.. code-block:: python

  import numpy as np
  import pyxis as px

  # Create data
  nb_samples = 10
  X = np.ones((nb_samples, 2, 2), dtype=np.float32)
  y = np.arange(nb_samples, dtype=np.uint8)

  # Write
  db = px.Writer(dirpath='data', map_size_limit=1)
  db.put_samples('input', X, 'target', y)
  db.close()

  # Read
  db = px.Reader(dirpath='data')
  sample = db[0]
  db.close()

  print(sample)

.. code-block:: python

  {'input': array([[ 1.,  1.], [ 1.,  1.]], dtype=float32), 'target': array(0, dtype=uint8)}

More examples can be found in the ``examples/`` directory.


Installation
============

The installation instructions are generic and should work on most operating
systems that support the prerequisites.

``ml-pyxis`` requires Python version 2.7, 3.4, 3.5, or 3.6. We recommend
installing ``ml-pyxis``, as well as all prerequisites, in a virtual environment
via `virtualenv`_.


-------------
Prerequisites
-------------

The following Python packages are required to use ``ml-pyxis``:

* `lmdb`_ - Universal Python binding for the `LMDB 'Lightning' Database`_
* `msgpack`_ - `MessagePack`_ implementation for Python (binary serialisation)
* `NumPy`_ - N-dimensional array object and tools for operating on them
* `six`_ - A Python 2 and 3 compatibility library

Please refer to the individual packages for more information about additional
dependencies and how to install them for your operating system.


--------------------------
Bleeding-edge installation
--------------------------

To install the latest version of ``ml-pyxis``, use the following command:

.. code-block:: bash

  pip install --upgrade https://github.com/vicolab/ml-pyxis/archive/master.zip

Add the ``--user`` tag if you want to install the package in your home
directory.


Notice
------

The previous LMDB-only API has been deprecated in favour of a combination
between LMDB and msgpack. The old version can be installed by using the
following commit hash with pip:

.. code-block:: bash

  pip install --upgrade git+git://github.com/vicolab/ml-pyxis.git@787c3484e3121f2767b254fc41be091d0a3e0cf0


------------------------
Development installation
------------------------

``ml-pyxis`` can be installed from source in such a way that any changes to
your local copy will take effect without having to reinstall the package.
Start by making a copy of the repository:

.. code-block:: bash

  git clone https://github.com/vicolab/ml-pyxis.git

Next, enter the directory and install ``ml-pyxis`` in development mode by
issuing the following command:

.. code-block:: bash

  cd ml-pyxis
  python setup.py develop


.. Links

.. _virtualenv: https://virtualenv.pypa.io/en/stable/
.. _lmdb: http://lmdb.readthedocs.io/en/release/
.. _LMDB 'Lightning' Database: https://symas.com/products/lightning-memory-mapped-database/
.. _msgpack: https://github.com/msgpack/msgpack-python
.. _MessagePack: http://msgpack.org/
.. _NumPy: http://www.numpy.org/
.. _six: https://github.com/benjaminp/six
