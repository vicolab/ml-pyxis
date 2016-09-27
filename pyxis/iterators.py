# -*- coding: utf-8 -*-
#
# iterators.py: Implements a generic thread-safe Python iterator and a couple of
# examples that allows the user to further customise data read from a LMDB.
#
import abc
import threading

__all__ = [
    "Iterator",
    "SimpleBatchIterator",
    "StochasticBatchIterator",
]


class Iterator(object):
    """Abstract thread-safe Python iterator.

    By thread-safe we mean that when more than one thread makes use of the
    iterator it will not raise an exception.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    @abc.abstractmethod
    def __next__(self):
        """Returns the next element that is iterated over.

        This is the only method that *must* be implemented in any object that
        inherits from this iterator.
        """

    next = __next__  # Alias for Python 2.*


class SimpleBatchIterator(Iterator):
    """A rudimentary batch iterator that wraps the ``batch_generator`` method.

    Parameters
    ----------
    db_reader : ``pyxis.Reader``
        A ``pyxis.Reader`` object which is used to access a LMDB.
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
    """

    def __init__(self, db_reader, batch_size, shuffle=False, endless_mode=True):
        self.batch_gen = db_reader.batch_generator(batch_size=batch_size,
                                                   shuffle=shuffle,
                                                   endless_mode=endless_mode)
        super(SimpleBatchIterator, self).__init__()

    def __next__(self):
        with self.lock:
            X, y = next(self.batch_gen)

        return X, y


class StochasticBatchIterator(Iterator):
    """A rudimentary batch iterator that wraps the
    ``stochastic_batch_generator`` method.

    Parameters
    ----------
    db_reader : ``pyxis.Reader``
        A ``pyxis.Reader`` object which is used to access a LMDB.
    batch_size : int
        Number of samples that should make up a batch.
    """

    def __init__(self, db_reader, batch_size):
        self.batch_gen = db_reader.stochastic_batch_generator(
            batch_size=batch_size)
        super(StochasticBatchIterator, self).__init__()

    def __next__(self):
        with self.lock:
            X, y = next(self.batch_gen)

        return X, y
