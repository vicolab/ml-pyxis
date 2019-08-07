# -*- coding: utf-8 -*-
"""pyxis.py: Tool for reading and writing datasets of tensors (`numpy.ndarray`) with
MessagePack and Lightning Memory-Mapped Database (LMDB).
"""
import abc
import collections
import contextlib
import io
import itertools
import os

import lmdb
import msgpack
import msgpack_numpy
import numpy as np

msgpack_numpy.patch()


def serialise(value):
    if isinstance(value, bytes):
        return value

    if isinstance(value, dict):
        obj = {k: msgpack.packb(value[k], use_bin_type=True) for k in value}
    else:
        obj = value

    return msgpack.packb(obj, use_bin_type=True)


def deserialise(value, is_sample=False):
    if not isinstance(value, bytes):
        return value

    if is_sample:
        obj = {}
        _obj = msgpack.unpackb(value, raw=False, use_list=True)
        for k in _obj:
            _k = k.decode() if isinstance(k, bytes) else str(k)
            obj[_k] = msgpack.unpackb(_obj[_k], raw=False, use_list=False)
    else:
        obj = msgpack.unpackb(value, raw=False, use_list=True)

    return obj


class AbstractPyxis(abc.ABC):  # pragma: no cover
    @abc.abstractmethod
    def get(self, i):
        raise NotImplementedError

    @abc.abstractmethod
    def put(self, *args):
        raise NotImplementedError

    @abc.abstractmethod
    def replace(self, i, *args):
        raise NotImplementedError

    @abc.abstractmethod
    def close(self):
        raise NotImplementedError

    @abc.abstractmethod
    def render(self):
        raise NotImplementedError

    def __repr__(self):
        return str(self)


class LMDBPyxis(AbstractPyxis):
    BYTES_IN_MB = 1048576
    EXT = "mdb"
    MAIN_DB_KEY = serialise("main")
    METADATA_DB_KEY = serialise("metadata")
    DB_KEYS = [MAIN_DB_KEY, METADATA_DB_KEY]
    METADATA_KEYS = {"indices": serialise("__indices")}

    def __init__(
        self,
        path,
        map_size=1,
        max_map_size=8192,
        iter_shuffle=False,
        iter_rng=np.random,
    ):
        self.path = path
        self.default_map_size = map_size
        self.max_map_size = max_map_size
        self.iter_shuffle = iter_shuffle
        self.iter_rng = iter_rng

        # Convert MB to bytes
        self.default_map_size *= LMDBPyxis.BYTES_IN_MB
        self.max_map_size *= LMDBPyxis.BYTES_IN_MB

        self._env = None
        self._dbs = {LMDBPyxis.MAIN_DB_KEY: None, LMDBPyxis.METADATA_DB_KEY: None}
        self._isdir = os.path.isdir(self.path)

        # Touch the LMDB if necessary, but keep it read-only afterwards
        if not (
            os.path.isfile(self.path)
            or os.path.isfile(os.path.join(self.path, f"data.{LMDBPyxis.EXT}"))
        ):
            self._setup(readonly=False, lock=True)

        self._setup(readonly=True, lock=False)

    @property
    def is_open(self):
        if self._env is not None and isinstance(self._env, lmdb.Environment):
            try:
                self._env.flags()  # Dummy check
                return True
            except lmdb.Error:
                pass

        return False

    @property
    def is_readonly(self):
        if self.is_open:
            return self._env.flags()["readonly"]
        else:
            raise lmdb.Error("LMDB environment is not open")

    @property
    def is_locked(self):
        if self.is_open:
            return self._env.flags()["lock"]
        else:
            raise lmdb.Error("LMDB environment is not open")

    @property
    def map_size(self):
        if self.is_open:
            return self._env.info()["map_size"]
        else:
            return self.default_map_size

    @property
    def indices(self):
        self._open(readonly=True, lock=False)

        with self._env.begin(db=self._dbs[LMDBPyxis.METADATA_DB_KEY]) as txn:
            ind = deserialise(
                txn.get(LMDBPyxis.METADATA_KEYS["indices"]), is_sample=False
            )

        return [] if ind is None else ind

    @property
    def empty(self):
        return len(self) == 0

    def get(self, i):
        if self.empty:
            raise IndexError("The database is empty")
        elif not 0 <= i < len(self):
            raise IndexError(
                f" The selected sample, `{i}`, is out of bounds. There are "
                f"`{len(self)}` samples in total. Indices are zero-based"
            )

        self._open(readonly=True, lock=False)
        with self._env.begin(db=self._dbs[LMDBPyxis.MAIN_DB_KEY]) as txn:
            obj = deserialise(txn.get(serialise(i)), is_sample=True)

        return obj

    def put(self, *args):
        data = LMDBPyxis._unpack_key_value_from_args(*args)

        # Must be a set of samples, `__len__` must be implemented and cannot be a
        # dictionary for serialisation reasons: `TypeError`
        n_samples = []
        for k in data:
            if isinstance(data[k], dict):
                raise TypeError(f"Dictionary data is not supported: `{data[k]}`")

            n_samples.append(len(data[k]))

        # The number of elements must be the same over all values
        if len(n_samples) != n_samples.count(n_samples[0]):
            raise ValueError("The number of samples must be the same for all values")

        # Create indices
        start = 0 if len(self.indices) == 0 else sorted(self.indices)[-1] + 1
        new_indices = list(range(start, start + n_samples[0]))

        # Insert the data
        self._put_txn(new_indices, data, db_key=LMDBPyxis.MAIN_DB_KEY, n_samples=True)

        # Update the database with the new indices
        self._put_txn(
            LMDBPyxis.METADATA_KEYS["indices"],
            self.indices + new_indices,
            db_key=LMDBPyxis.METADATA_DB_KEY,
            n_samples=False,
        )

    def replace(self, i, *args):
        data = LMDBPyxis._unpack_key_value_from_args(*args)

        # Verify that the value to replaced already exists
        _ = self.get(i)

        # Insert the data
        self._put_txn(i, data, db_key=LMDBPyxis.MAIN_DB_KEY, n_samples=False)

    def batch(self, n=1):
        it = iter(self)
        while True:
            batch = tuple(itertools.islice(it, n))
            if not batch:
                return
            else:
                yield batch

    def close(self):
        if self._env is not None:
            self._env.close()
            self._dbs = {LMDBPyxis.MAIN_DB_KEY: None, LMDBPyxis.METADATA_DB_KEY: None}
            self._env = None

    def render(self):
        _vars = {k: vars(self)[k] for k in vars(self) if not k.startswith("_")}
        out = str(collections.namedtuple(self.__class__.__name__, _vars)(**_vars))
        return io.StringIO(out)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            i = int(key)
            i += len(self) if i < 0 else 0
            return self.get(i)
        elif isinstance(key, slice):
            return [self[i] for i in range(*key.indices(len(self)))]
        else:
            raise TypeError(f"Invalid argument type: `{type(key)}`")

    def __setitem__(self, key, value):
        if not isinstance(value, dict):
            raise ValueError(
                "The value must be a dictionary in the following format: "
                "`{'key1': value1, 'key2': value2, ...}`"
            )

        if isinstance(key, (int, np.integer)):
            i = int(key)
            i += len(self) if i < 0 else 0
            self.replace(i, value)
        elif isinstance(key, slice):
            for i in range(*key.indices(len(self))):
                self.replace(i, value)
        else:
            raise TypeError(f"Invalid argument type: `{type(key)}`")

    def __iter__(self):
        if self.iter_shuffle:
            indices = self.indices
            self.iter_rng.shuffle(indices)
            for key in indices:
                yield self.get(key)
        else:
            self._open(readonly=True, lock=False)
            with self._env.begin(db=self._dbs[LMDBPyxis.MAIN_DB_KEY]) as txn:
                cursor = txn.cursor()
                for _, value in cursor.iternext():
                    yield deserialise(value, is_sample=True)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __str__(self):
        out = self.render()
        with contextlib.closing(out):
            return out.getvalue()

    def _setup(self, readonly, lock):
        # Close previous environment
        self.close()

        # Open new environment
        self._env = lmdb.open(
            self.path,
            map_size=self.map_size,
            subdir=self._isdir,
            readonly=readonly,
            create=self._isdir,
            max_dbs=len(LMDBPyxis.DB_KEYS),
            lock=lock,
        )
        assert self.is_open

        # Open reference to databases
        self._dbs[LMDBPyxis.MAIN_DB_KEY] = self._env.open_db(LMDBPyxis.MAIN_DB_KEY)
        self._dbs[LMDBPyxis.METADATA_DB_KEY] = self._env.open_db(
            LMDBPyxis.METADATA_DB_KEY
        )

    def _open(self, readonly, lock):
        # If for some reason, the environment is closed, then open
        if not self.is_open:
            self._setup(readonly, lock)

        # Check if, given the arguments, it is necessary to open another environment
        if self.is_readonly == readonly and self.is_locked == lock:
            return False

        # Open new environment and run sanity check
        self._setup(readonly, lock)

        return True

    def _put_txn(self, ind, data, db_key, n_samples):
        self._open(readonly=False, lock=True)

        # Try to write, if map size is recognised as full, attempt to double it
        try:
            with self._env.begin(write=True, db=self._dbs[db_key]) as txn:
                if n_samples:
                    for i in range(len(ind)):
                        txn.put(
                            serialise(ind[i]), serialise({k: data[k][i] for k in data})
                        )
                else:
                    txn.put(serialise(ind), serialise(data))
        except lmdb.MapFullError:
            # Use NVIDIA DIGITS' method, where the map size is doubled on failure
            # see PR: https://github.com/NVIDIA/DIGITS/pull/209
            new_map_size = self.map_size * 2
            if new_map_size > self.max_map_size:
                raise RuntimeError(
                    "Unable to grow the LMDB map size any further: "
                    f"`{new_map_size / LMDBPyxis.BYTES_IN_MB}` MB > "
                    f"`{self.max_map_size / LMDBPyxis.BYTES_IN_MB}` MB. Please "
                    "increase `max_map_size` to allow a larger map size"
                )

            self._env.set_mapsize(self.map_size * 2)

            # Attempt the write again
            self._put_txn(ind, data, db_key, n_samples)

    @staticmethod
    def _unpack_key_value_from_args(*args):
        # Select `*args` style
        if len(args) == 1 and isinstance(args[0], dict):
            data = args[0]
        else:
            if not len(args) % 2 == 0:
                raise ValueError(
                    "Each value must be associated with a key: "
                    "`put(key1, value1, key2, value2, ...)`"
                )

            # Convert to dictionary format: `{key1: value1, key2: value2, ...}`
            data = dict((a, b) for a, b in zip(args[0::2], args[1::2]))

        return data


Pyxis = LMDBPyxis
