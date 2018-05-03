# -*- coding: utf-8 -*-
#
# iterators.py: Implements a set of generic Python iterators for reading data
# from pyxis.
#
import abc
import six
import threading

import numpy as np

__all__ = [
    "Iterator",
    "DataIterator",
    "SimpleBatch",
    "StochasticBatch",
    "SequentialBatch",
    "SimpleBatchThreadSafe",
    "StochasticBatchThreadSafe",
    "SequentialBatchThreadSafe",
]


@six.add_metaclass(abc.ABCMeta)
class Iterator(six.Iterator):
    """Abstract (thread-safe) Python iterator.

    By thread-safe we mean that when more than one thread makes use of the
    iterator it will not raise an exception. It is up to the user to make use
    of the lock object.
    """

    def __init__(self):
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    @abc.abstractmethod
    def __next__(self):
        """Return the next element that is iterated over.

        This is the only method that *must* be implemented in any object that
        inherits from this iterator.
        """


class DataIterator(Iterator):
    """Abstract iterator that keeps track of the pyxis LMDB and data keys.

    This is an intermediate object to link the abstract iterator object with a
    pyxis LMDB.

    Parameters
    ----------
    db : `pyxis.Reader`
        This is the database that the iterator will iterate over.
    keys: tuple or list
        Tuple of (dictionary) keys. The data returned by the iterator will be
        the values for which these keys point to. The order of the keys matter.
        For example, when using the keys `('a', 'b')` the iterator will return
        `(a_val, b_val)`, where `a_val` and `b_val` are the values associated
        with the keys `'a'` and `'b'`, respectively.
    rng : `numpy.random` or a :class:`numpy.random.RandomState` instance
        The random number generator to use. Default is `numpy.random`.
    """

    def __init__(self, db, keys, rng=np.random):
        super(DataIterator, self).__init__()
        self.db = db
        self.keys = keys
        self.rng = rng

        # If there is only one key, wrap it in a list
        if isinstance(self.keys, str):
            self.keys = [self.keys]

        # Retrieve the data specification (shape & dtype) for all data objects
        # Assumes that all samples have the same shape and data type
        self.spec = db.get_data_specification(0)


class SimpleBatch(DataIterator):
    """A simple batch iterator.

    Parameters
    ----------
    db : `pyxis.Reader`
        This is the database that the iterator will iterate over.
    keys: tuple or list
        Tuple of (dictionary) keys. The data returned by the iterator will be
        the values for which these keys point to. The order of the keys matter.
        For example, when using the keys `('a', 'b')` the iterator will return
        `(a_val, b_val)`, where `a_val` and `b_val` are the values associated
        with the keys `'a'` and `'b'`, respectively.
    batch_size : int
        Number of samples that should make up a batch.
    shuffle : boolean
        When set to `True` the samples are shuffled before being split
        up into batches, otherwise the samples er kept in the order they
        were written. Default is `False`.
    endless : boolean
        Indicates whether or not the batch generator should yield the whole
        dataset only once (`False`) or until the user stops using the
        function (`True`). `Default is `True`.
    rng : `numpy.random` or a :class:`numpy.random.RandomState` instance
        The random number generator to use. Default is `numpy.random`.
    """

    def __init__(self, db, keys, batch_size, shuffle=False, endless=True,
                 rng=np.random):
        super(SimpleBatch, self).__init__(db, keys, rng)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.endless = endless

        # Set up Python generator
        self.gen = self.batch()

    def __next__(self):
        return next(self.gen)

    def batch(self):
        """Return a batch of samples.

        Returns
        -------
        (numpy.ndarray, ...)
            The sample values are returned in a tuple in the order of the
            `keys` specified by the user.
        """
        # Count the number of keys (i.e. data objects)
        nb_keys = len(self.keys)

        # Keep track of whether or not this is the end of the dataset
        self.end_of_dataset = False

        # Generate indices for the data
        idxs = np.arange(len(self.db), dtype=np.uint64)

        # Compute how many calls it will take to go through the whole dataset
        nb_calls = len(self.db) / self.batch_size

        while True:
            # Shuffle indices
            if self.shuffle:
                self.rng.shuffle(idxs)

            # Generate batches
            for call in np.arange(np.ceil(nb_calls)):
                # Check and see if we have enough data to fill a whole batch
                if call < np.floor(nb_calls):  # Whole batch
                    size = self.batch_size
                    self.end_of_dataset = False
                else:                          # Remainder
                    size = len(self.db) % self.batch_size
                    self.end_of_dataset = True

                # Check and see if we have gone through the dataset
                if call + 1 == np.ceil(nb_calls):
                    self.end_of_dataset = True

                # Create a batch for each key
                data = []
                for key in self.keys:
                    data.append(np.zeros((size,) + self.spec[key]['shape'],
                                dtype=self.spec[key]['dtype']))

                batch_idxs = np.arange(size) + self.batch_size * call
                batch_idxs = batch_idxs.astype(np.uint64)

                for i, v in enumerate(batch_idxs):
                    sample = self.db[idxs[v]]
                    for k in range(nb_keys):
                        data[k][i] = sample[self.keys[k]]

                # Account for batches with only one key
                if 1 == len(data):
                    yield tuple(data)[0]
                else:
                    yield tuple(data)

            if not self.endless:
                break


class SimpleBatchThreadSafe(SimpleBatch):
    """A simple, thread-safe batch iterator.

    Parameters
    ----------
    db : `pyxis.Reader`
        This is the database that the iterator will iterate over.
    keys: tuple or list
        Tuple of (dictionary) keys. The data returned by the iterator will be
        the values for which these keys point to. The order of the keys matter.
        For example, when using the keys `('a', 'b')` the iterator will return
        `(a_val, b_val)`, where `a_val` and `b_val` are the values associated
        with the keys `'a'` and `'b'`, respectively.
    batch_size : int
        Number of samples that should make up a batch.
    shuffle : boolean
        When set to `True` the samples are shuffled before being split
        up into batches, otherwise the samples er kept in the order they
        were written. Default is `False`.
    endless : boolean
        Indicates whether or not the batch generator should yield the whole
        dataset only once (`False`) or until the user stops using the
        function (`True`). `Default is `True`.
    rng : `numpy.random` or a :class:`numpy.random.RandomState` instance
        The random number generator to use. Default is `numpy.random`.
    """

    def __init__(self, db, keys, batch_size, shuffle=False, endless=True,
                 rng=np.random):
        super(SimpleBatchThreadSafe, self).__init__(db, keys, batch_size,
                                                    shuffle, endless, rng)

    def __next__(self):
        with self.lock:
            data = next(self.gen)

        return data


class StochasticBatch(DataIterator):
    """A stochastic batch iterator.

    By stochastic we mean that the samples in a batch are gathered by uniform
    sampling from the database.

    Parameters
    ----------
    db : `pyxis.Reader`
        This is the database that the iterator will iterate over.
    keys: tuple or list
        Tuple of (dictionary) keys. The data returned by the iterator will be
        the values for which these keys point to. The order of the keys matter.
        For example, when using the keys `('a', 'b')` the iterator will return
        `(a_val, b_val)`, where `a_val` and `b_val` are the values associated
        with the keys `'a'` and `'b'`, respectively.
    batch_size : int
        Number of samples that should make up a batch.
    rng : `numpy.random` or a :class:`numpy.random.RandomState` instance
        The random number generator to use. Default is `numpy.random`.
    """

    def __init__(self, db, keys, batch_size, rng=np.random):
        super(StochasticBatch, self).__init__(db, keys, rng)
        self.batch_size = batch_size

        # Set up Python generator
        self.gen = self.batch()

    def __next__(self):
        return next(self.gen)

    def batch(self):
        """Return a batch of samples sampled uniformly from the database.

        Returns
        -------
        (numpy.ndarray, ...)
            The sample values are returned in a tuple in the order of the
            `keys` specified by the user.
        """
        # Count the number of keys (i.e. data objects)
        nb_keys = len(self.keys)

        data = []
        for key in self.keys:
            data.append(np.zeros((self.batch_size,) + self.spec[key]['shape'],
                        dtype=self.spec[key]['dtype']))

        while True:
            # Sample indices uniformly
            batch_idxs = self.rng.randint(len(self.db),
                                          size=self.batch_size,
                                          dtype=np.uint64)

            for i, v in enumerate(batch_idxs):
                sample = self.db[int(v)]
                for k in range(nb_keys):
                    data[k][i] = sample[self.keys[k]]

            # Account for batches with only one key
            if 1 == len(data):
                yield tuple(data)[0]
            else:
                yield tuple(data)


class StochasticBatchThreadSafe(StochasticBatch):
    """A stochastic, thread-safe batch iterator.

    By stochastic we mean that the samples in a batch are gathered by uniform
    sampling from the database.

    Parameters
    ----------
    db : `pyxis.Reader`
        This is the database that the iterator will iterate over.
    keys: tuple or list
        Tuple of (dictionary) keys. The data returned by the iterator will be
        the values for which these keys point to. The order of the keys matter.
        For example, when using the keys `('a', 'b')` the iterator will return
        `(a_val, b_val)`, where `a_val` and `b_val` are the values associated
        with the keys `'a'` and `'b'`, respectively.
    batch_size : int
        Number of samples that should make up a batch.
    rng : `numpy.random` or a :class:`numpy.random.RandomState` instance
        The random number generator to use. Default is `numpy.random`.
    """

    def __init__(self, db, keys, batch_size, rng=np.random):
        super(StochasticBatchThreadSafe, self).__init__(db, keys, batch_size,
                                                        rng)

    def __next__(self):
        with self.lock:
            data = next(self.gen)

        return data


class SequentialBatch(DataIterator):
    """A sequential batch iterator.

    By sequential we mean that the samples in a batch are gathered exactly in
    the way it is laid out in the LMDB.

    Parameters
    ----------
    db : `pyxis.Reader`
        This is the database that the iterator will iterate over.
    keys: tuple or list
        Tuple of (dictionary) keys. The data returned by the iterator will be
        the values for which these keys point to. The order of the keys matter.
        For example, when using the keys `('a', 'b')` the iterator will return
        `(a_val, b_val)`, where `a_val` and `b_val` are the values associated
        with the keys `'a'` and `'b'`, respectively.
    batch_size : int
        Number of samples that should make up a batch.
    endless : boolean
        Indicates whether or not the batch generator should yield the whole
        dataset only once (`False`) or until the user stops using the
        function (`True`). `Default is `True`.
    """

    def __init__(self, db, keys, batch_size, endless=True):
        super(SequentialBatch, self).__init__(db, keys)
        self.batch_size = batch_size
        self.endless = endless

        # Set up Python generator
        self.gen = self.batch()

    def __next__(self):
        return next(self.gen)

    def batch(self):
        """Return a batch of samples as they are laid out in the LMDB.

        Returns
        -------
        (numpy.ndarray, ...)
            The sample values are returned in a tuple in the order of the
            `keys` specified by the user.
        """
        # Count the number of keys (i.e. data objects)
        nb_keys = len(self.keys)

        # Keep track of whether or not this is the end of the dataset
        self.end_of_dataset = False

        # Compute how many calls it will take to go through the whole dataset
        nb_calls = len(self.db) / self.batch_size

        data = []
        for key in self.keys:
            data.append(np.zeros((self.batch_size,) + self.spec[key]['shape'],
                        dtype=self.spec[key]['dtype']))

        while True:
            # Generate batches
            for call in np.arange(np.ceil(nb_calls)):
                # Check and see if we have enough data to fill a whole batch
                if call < np.floor(nb_calls):  # Whole batch
                    size = self.batch_size
                    self.end_of_dataset = False
                else:                          # Remainder
                    size = len(self.db) % self.batch_size
                    self.end_of_dataset = True

                # Check and see if we have gone through the dataset
                if call + 1 == np.ceil(nb_calls):
                    self.end_of_dataset = True

                # Create a batch for each key
                data = []
                for key in self.keys:
                    data.append(np.zeros((size,) + self.spec[key]['shape'],
                                dtype=self.spec[key]['dtype']))

                start_idx = int(self.batch_size * (call))

                samples = self.db.get_samples(start_idx, size)

                for i in range(size):
                    for k in range(nb_keys):
                        data[k][i] = samples[self.keys[k]][i]

                # Account for batches with only one key
                if 1 == len(data):
                    yield tuple(data)[0]
                else:
                    yield tuple(data)

            if not self.endless:
                break


class SequentialBatchThreadSafe(SequentialBatch):
    """A sequential, thread-safe batch iterator.

    By sequential we mean that the samples in a batch are gathered exactly in
    the way it is laid out in the LMDB.

    Parameters
    ----------
    db : `pyxis.Reader`
        This is the database that the iterator will iterate over.
    keys: tuple or list
        Tuple of (dictionary) keys. The data returned by the iterator will be
        the values for which these keys point to. The order of the keys matter.
        For example, when using the keys `('a', 'b')` the iterator will return
        `(a_val, b_val)`, where `a_val` and `b_val` are the values associated
        with the keys `'a'` and `'b'`, respectively.
    batch_size : int
        Number of samples that should make up a batch.
    endless : boolean
        Indicates whether or not the batch generator should yield the whole
        dataset only once (`False`) or until the user stops using the
        function (`True`). `Default is `True`.
    """

    def __init__(self, db, keys, batch_size, endless=True):
        super(SequentialBatchThreadSafe, self).__init__(db, keys, batch_size,
                                                        endless)

    def __next__(self):
        with self.lock:
            data = next(self.gen)

        return data
