#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# pyxis.py: Tool for reading and writing datasets of tensors in a Lightning
#           Memory-Mapped Database (LMDB).
#
from __future__ import division

import lmdb
import numpy as np

__all__ = [
    "Reader",
    "Writer",
]


def encode_str(string, encoding='utf-8', errors='strict'):
    """Return an encoded byte object of the input string.

    Parameters
    ----------
    string : string
    encoding : string
        Default is `utf-8`.
    errors : string
        Specifies how encoding errors should be handled. Default is
        `strict`.
    """
    return str(string).encode(encoding=encoding, errors=errors)


def decode_bytes(byte_obj, encoding='utf-8', errors='strict'):
    """Decodes the input byte object to a string.

    Parameter
    ---------
    byte_obj : byte object
    encoding : string
        Default is `utf-8`.
    errors : string
        Specifies how encoding errors should be handled. Default is
        `strict`.
    """
    return byte_obj.decode(encoding=encoding, errors=errors)


# Three databases: (i) inputs, (ii) targets, and (iii) metadata
NB_DBS       = 3

# Name of the three databases
INPUT_DB     = encode_str('INPUT_DB')
TARGET_DB    = encode_str('TARGET_DB')
METADATA_DB  = encode_str('METADATA_DB')

# Keys for the metadata
NB_SAMPLES   = encode_str('NB_SAMPLES')
INPUT_DTYPE  = encode_str('INPUT_DTYPE')
TARGET_DTYPE = encode_str('TARGET_DTYPE')
INPUT_SHAPE  = encode_str('INPUT_SHAPE')
TARGET_SHAPE = encode_str('TARGET_SHAPE')


class Reader(object):
    """Object for reading a dataset of tensors in a Lightning Memory-Mapped
    Database (LMDB).

    Parameter
    ---------
    dirpath : string
        Path to the directory containing the LMDB.
    """

    def __init__(self, dirpath):
        # Open LMDB environment in read-only mode
        self._lmdb_env = lmdb.open(dirpath, readonly=True, max_dbs=NB_DBS)

        # Open the three databases associated with the environment
        self.input_db = self._lmdb_env.open_db(INPUT_DB)
        self.target_db = self._lmdb_env.open_db(TARGET_DB)
        self.metadata_db = self._lmdb_env.open_db(METADATA_DB)

        # Begin transaction
        self._txn = self._lmdb_env.begin()

        # Read metadata
        # Number of samples in the dataset
        byte_obj = self._txn.get(NB_SAMPLES, db=self.metadata_db)
        self.nb_samples = int(decode_bytes(byte_obj))

        # Read data types
        # Input
        byte_obj = self._txn.get(INPUT_DTYPE, db=self.metadata_db)
        self.input_dtype = np.dtype(decode_bytes(byte_obj))
        # Target
        byte_obj = self._txn.get(TARGET_DTYPE, db=self.metadata_db)
        self.target_dtype = np.dtype(decode_bytes(byte_obj))

        # Read data shapes
        # Input
        byte_obj = self._txn.get(INPUT_SHAPE, db=self.metadata_db)
        self.input_shape = tuple(np.fromstring(byte_obj, dtype=np.uint8))
        # Target
        byte_obj = self._txn.get(TARGET_SHAPE, db=self.metadata_db)
        self.target_shape = tuple(np.fromstring(byte_obj, dtype=np.uint8))

    def batch_generator(self, batch_size, shuffle=False, endless_mode=True):
        """Return a batch of samples from `input_db` and `target_db`.

        Parameters
        ----------
        batch_size : int
            Number of samples that should make up a batch.
        shuffle : boolean
            When set to `True` the samples are shuffled before being split
            up into batches, otherwise the samples er kept in the order they
            were written. Default is `False`.
        endless_mode : boolean
            Indicates whether or not the batch generator should yield the whole
            dataset only once (`False`) or until the user stops using the
            function (`True`). `Default is `True`.

        Returns
        -------
        (numpy.array, numpy.array)
            The batch generator returns two values packed in a tuple. The
            first is a batch of inputs and the second is a batch of targets.
        """
        # Keep track of whether or not this is the end of the dataset
        self.end_of_dataset = False

        # Generate indices for the data
        idxs = np.arange(self.nb_samples, dtype=np.uint64)

        # Compute how many calls it will take to go through the whole dataset
        nb_calls = self.nb_samples / batch_size

        while True:
            # Shuffle indices
            if shuffle:
                np.random.shuffle(idxs)

            # Generate batches
            for call in np.arange(np.ceil(nb_calls)):
                # Check and see if we have enough data to fill a whole batch
                if call < np.floor(nb_calls):  # Whole batch
                    size = batch_size
                    self.end_of_dataset = False
                else:                          # Remainder
                    size = self.nb_samples % batch_size
                    self.end_of_dataset = True

                # Check and see if we have gone through the dataset
                if call + 1 == np.ceil(nb_calls):
                    self.end_of_dataset = True

                # Create a batch
                xs = np.zeros((size,) + self.input_shape,
                              dtype=self.input_dtype)
                ys = np.zeros((size,) + self.target_shape,
                              dtype=self.target_dtype)

                batch_idxs = np.arange(size) + batch_size * call
                batch_idxs = batch_idxs.astype(np.int)

                for i, v in enumerate(batch_idxs):
                    xs[i] = self.get_input(idxs[v])
                    ys[i] = self.get_target(idxs[v])

                yield xs, ys

            if not endless_mode:
                break

    def stochastic_batch_generator(self, batch_size):
        """Return a batch of samples uniformly sampled from `input_db` and
        `target_db`.

        Parameters
        ----------
        batch_size : int
            Number of samples that should make up a batch.

        Returns
        -------
        (numpy.array, numpy.array)
            The batch generator returns two values packed in a tuple. The
            first is a batch of inputs and the second is a batch of targets.
        """
        xs = np.zeros((batch_size,) + self.input_shape, dtype=self.input_dtype)
        ys = np.zeros((batch_size,) + self.target_shape,
                      dtype=self.target_dtype)

        while True:
            # Randomly sample indices
            batch_idxs = np.random.randint(self.nb_samples, size=batch_size,
                                           dtype=np.uint64)

            for i, v in enumerate(batch_idxs):
                xs[i] = self.get_input(v)
                ys[i] = self.get_target(v)

            yield xs, ys

    def get_sample(self, i):
        """Return the ith sample from `input_db` and `target_db`.

        Parameter
        ---------
        i : int
        """
        x = self.get_input(i)
        y = self.get_target(i)
        return x, y

    def get_input(self, i):
        """Return the ith input tensor from `input_db`.

        The tensor is a `numpy.array` reshaped with respect to `input_shape`.

        Parameter
        ---------
        i : int
        """
        _input = self._get_array(i, is_inputs=True)
        return np.reshape(_input, self.input_shape)

    def get_target(self, i):
        """Return the ith label from `target_db`.

        The tensor is a `numpy.array` reshaped with respect to `target_shape`.

        Parameter
        ---------
        i : int
        """
        _target = self._get_array(i, is_inputs=False)
        return np.reshape(_target, self.target_shape)

    def _get_array(self, i, is_inputs=True):
        """Returns the ith array from either `input_db` or `target_db`.

        Parameters
        ----------
        i : int
        is_inputs : bool
            When set to `True` the `input_db` is used, otherwise the `target_db`
            is used instead. Default is `True`.
        """
        if i >= self.nb_samples:
            raise ValueError('The selected sample number `i` is larger than '
                             'the number of samples in the database: %d' % i)

        # Convert `i` to a string with trailing zeros
        key = '{:010}'.format(i)

        if is_inputs:
            byte_obj = self._txn.get(encode_str(key), db=self.input_db)
            array = np.fromstring(byte_obj, dtype=self.input_dtype)
        else:
            byte_obj = self._txn.get(encode_str(key), db=self.target_db)
            array = np.fromstring(byte_obj, dtype=self.target_dtype)

        return np.copy(array)

    def close(self):
        """Close the environment.

        Invalidates any open iterators, cursors, and transactions.
        """
        self._lmdb_env.close()


class Writer(object):
    """Object for writing a dataset of tensors to a Lightning Memory-Mapped
    Database (LMDB).

    Parameters
    ----------
    dirpath : string
        Path to the directory where the LMDB should be written.
    input_shape : tuple of ints
        Shape of an input tensor, e.g. `(254, 254, 3)` for a colour image with
        254 rows and columns.
    target_shape : tuple of ints
        Shape of a target tensor, e.g. `()` for an 1-d array. Default is `()`.
    input_dtype : `numpy.dtype` or string
        Data type of the input data. Default is `numpy.uint8`.
    target_dtype : `numpy.dtype` or string
        Data type of the target data. Default is `numpy.uint8`.
    ram_gb_limit : int
        The maximum size of data (inputs and targets) that can be put on the
        RAM at the same time. The size of the data input to `put_samples` can
        not exceed this number. Default is `2` GB.
    map_size_limit : int
        Map size for LMDB in MB. Default is `1000` MB.
    """

    def __init__(self, dirpath, input_shape, target_shape=(),
                 input_dtype=np.uint8, target_dtype=np.uint8, ram_gb_limit=2,
                 map_size_limit=1000):
        self.ram_gb_limit = ram_gb_limit
        self.map_size_limit = int(map_size_limit)
        self.input_dtype = np.dtype(input_dtype)
        self.target_dtype = np.dtype(target_dtype)
        self.nb_samples = 0

        # Convert `map_size_limit` from MB to B
        map_size_limit <<= 20

        # Open LMDB environment
        self._lmdb_env = lmdb.open(dirpath, map_size=map_size_limit,
                                   max_dbs=NB_DBS)

        # Open the three databases associated with the environment
        self.input_db = self._lmdb_env.open_db(INPUT_DB)
        self.target_db = self._lmdb_env.open_db(TARGET_DB)
        self.metadata_db = self._lmdb_env.open_db(METADATA_DB)

        # Write the metadata we already have to `metadata_db`
        with self._lmdb_env.begin(write=True) as txn:
            # Data types
            # Input
            txn.put(INPUT_DTYPE, encode_str(self.input_dtype.str),
                    db=self.metadata_db)
            # Target
            txn.put(TARGET_DTYPE, encode_str(self.target_dtype.str),
                    db=self.metadata_db)

            # Data shapes
            # Input
            txn.put(INPUT_SHAPE, self._pack_array(np.array(input_shape),
                                                  is_data=False),
                    db=self.metadata_db)
            # Target
            txn.put(TARGET_SHAPE, self._pack_array(np.array(target_shape),
                                                   is_data=False),
                    db=self.metadata_db)

    def put_samples(self, inputs, targets):
        """Puts the inputs and targets into the `input_db` and `target_db`,
        respectively.

        Parameters
        ----------
        inputs : numpy.array
            A `numpy.array` where the first axis is used to select an input.
        targets : numpy.array
            A `numpy.array` where the first axis is used to select a target.
        """
        # Ensure that a hypothetical RAM can handle the number of samples being
        # stored
        gb_in_used = np.uint64(inputs.nbytes) + np.uint64(targets.nbytes)
        gb_in_used = float(gb_in_used / 10**9)
        if self.ram_gb_limit < gb_in_used:
            raise ValueError('The size of the data that are to be stored is '
                             'bigger than `ram_gb_limit`: '
                             '%d < %f' % (self.ram_gb_limit, gb_in_used))

        # Ensure that the number of inputs and targets are equal
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError('The number of inputs must equal the number of '
                             'targets: %d inputs vs. %d '
                             'targets' % (inputs.shape[0], targets.shape[0]))

        try:
            # Attempt to write the inputs and targets
            with self._lmdb_env.begin(write=True) as txn:
                for i, _input in enumerate(inputs):
                    # Find associated target
                    _target = targets[i]
                    if _target is not np.ndarray:
                        _target = np.array(_target)

                    # Key :: sample number as a string with trailing zeros
                    key = encode_str('{:010}'.format(self.nb_samples))

                    # Put the input in `input_db`
                    txn.put(key, self._pack_array(_input, is_inputs=True),
                            db=self.input_db)
                    # Put the target in `target_db`
                    txn.put(key, self._pack_array(_target, is_inputs=False),
                            db=self.target_db)

                    # Increase sample counter
                    self.nb_samples += 1
        except lmdb.MapFullError as e:
            raise AttributeError('The LMDB `map_size` is too small: '
                                 '%s MB, %s' % (self.map_size_limit, e))

    def _pack_array(self, array, is_data=True, is_inputs=True):
        """Returns a flattened byte object version of the incoming array.

        Parameter
        ---------
        array : numpy.array
        is_data : bool
            When set to `True` the array will be cast to either `input_dtype`
            or `target_dtype`, otherwise `numpy.uint8` will be used. Default is
            `True`.
        is_inputs : bool
            When set to `True` the array will be cast to `input_dtype`,
            otherwise `target_dtype` is used instead. Default is `True`.
        """
        if is_data:
            if is_inputs:
                arr = array.astype(self.input_dtype)
            else:
                arr = array.astype(self.target_dtype)
        else:
            arr = array.astype(np.uint8)
        return arr.flatten().tostring()

    def close(self):
        """Close the environment. Before closing, the number of samples is
        written to `metadata_db`.

        Invalidates any open iterators, cursors, and transactions.
        """
        with self._lmdb_env.begin(write=True) as txn:
            txn.put(NB_SAMPLES, encode_str(self.nb_samples),
                    db=self.metadata_db)

        self._lmdb_env.close()
