"""Tool for reading and writing datasets of tensors (`numpy.ndarray`) with MessagePack
and Lightning Memory-Mapped Database (LMDB).
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
    def has(self, i):
        raise NotImplementedError

    @abc.abstractmethod
    def get(self, ind):
        raise NotImplementedError

    @abc.abstractmethod
    def put(self, ind, *args):
        raise NotImplementedError

    @abc.abstractmethod
    def delete(self, ind, *args):
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

        self._indices = []
        self._dirty_indices = True

        self._env = None
        self._dbs = {LMDBPyxis.MAIN_DB_KEY: None, LMDBPyxis.METADATA_DB_KEY: None}
        self._isdir = os.path.isdir(self.path)

        # Touch the LMDB if necessary, but keep it read-only afterwards
        if not self.exists:
            self._setup(readonly=False, lock=True)

        self._setup(readonly=True, lock=False)

    @property
    def exists(self):
        return os.path.isfile(self.path) or os.path.isfile(
            os.path.join(self.path, f"data.{LMDBPyxis.EXT}")
        )

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
        # Only touch database when necessary
        if self._dirty_indices:
            self._open(readonly=True, lock=False)

            with self._env.begin(db=self._dbs[LMDBPyxis.METADATA_DB_KEY]) as txn:
                ind = deserialise(
                    txn.get(LMDBPyxis.METADATA_KEYS["indices"]), is_sample=False
                )

            self._indices = [] if ind is None else ind
            self._dirty_indices = False

        return self._indices

    @property
    def empty(self):
        return len(self) == 0

    @staticmethod
    def is_index(i):
        if isinstance(i, (int, np.integer)) and i >= 0:
            return True
        else:
            return False

    def has(self, i):
        if LMDBPyxis.is_index(i):
            return i in self.indices
        else:
            raise IndexError(f"An index must be a non-negative integer: `{i}`")

    def get(self, ind):
        if self.empty:
            raise IndexError("The database is empty")

        ind = self._verify_ind(ind)

        # Get all elements from database
        self._open(readonly=True, lock=False)
        objs = []
        with self._env.begin(db=self._dbs[LMDBPyxis.MAIN_DB_KEY]) as txn:
            for i in ind:
                objs.append(deserialise(txn.get(serialise(i)), is_sample=True))

        return objs[0] if len(ind) == 1 else objs

    def put(self, ind, *args, **kwargs):
        overwrite = kwargs.get("overwrite", True)
        ind = [ind] if not isinstance(ind, list) else ind
        ind = list(set(ind))
        data = LMDBPyxis._unpack_key_value_from_args(*args)

        if len(ind) > 1:
            n_elem = []
            # 1. ensure that data can be indexed (`__len__` and `__getitem__`)
            for k in data:
                # Dictionaries are currently explicitly not supported
                if isinstance(data[k], dict):
                    raise TypeError(f"Dictionary data is not supported: `{data[k]}`")

                if not (
                    hasattr(data[k], "__len__") and hasattr(data[k], "__getitem__")
                ):
                    raise TypeError(
                        f"Object associated with `{k}` does not have `__len__` and "
                        "`__getitem__` attributes"
                    )

                n_elem.append(len(data[k]))

            # 2. ensure the same number of elements over all data values
            if len(n_elem) != n_elem.count(n_elem[0]):
                raise ValueError(
                    "The number of elements must be the same for all values"
                )

        # Put data in database
        self._put_txn(ind, data, db_key=LMDBPyxis.MAIN_DB_KEY, overwrite=overwrite)

        # Update indices in database
        indices = self.indices
        indices.extend(ind)
        indices = sorted(list(set(indices)))
        self._put_txn(
            LMDBPyxis.METADATA_KEYS["indices"],
            indices,
            db_key=LMDBPyxis.METADATA_DB_KEY,
            overwrite=True,
        )
        self._dirty_indices = True

    def delete(self, ind, *args):
        ind = list(set(self._verify_ind(ind)))

        # Delete from database
        self._del_txn(ind, db_key=LMDBPyxis.MAIN_DB_KEY)

        # Update indices in database
        indices = self.indices
        for i in ind:
            indices.remove(i)

        self._put_txn(
            LMDBPyxis.METADATA_KEYS["indices"],
            indices,
            db_key=LMDBPyxis.METADATA_DB_KEY,
            overwrite=True,
        )
        self._dirty_indices = True

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
            return self.get(self.indices[i])
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
            self.put(self.indices[i], value)
        elif isinstance(key, slice):
            for i in range(*key.indices(len(self))):
                self.put(self.indices[i], value)
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

    def _verify_ind(self, ind):
        ind = [ind] if not isinstance(ind, list) else ind
        invalid_ind = list(itertools.compress(ind, [not self.has(i) for i in ind]))
        if len(invalid_ind) > 0:
            raise IndexError(
                f"Non-existent {'indices' if len(invalid_ind) > 1 else 'index'}: "
                f"`{invalid_ind}`"
            )

        return ind

    def _put_txn(self, ind, data, db_key, overwrite=True):
        self._open(readonly=False, lock=True)

        # Try to write, if map size is recognised as full, attempt to double it
        try:
            with self._env.begin(write=True, db=self._dbs[db_key]) as txn:
                if isinstance(ind, list):
                    if len(ind) > 1:
                        for i in range(len(ind)):
                            txn.put(
                                serialise(ind[i]),
                                serialise({k: data[k][i] for k in data}),
                                overwrite=overwrite,
                            )
                    else:
                        txn.put(serialise(ind[0]), serialise(data), overwrite=overwrite)
                else:
                    txn.put(serialise(ind), serialise(data), overwrite=overwrite)
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
            self._put_txn(ind, data, db_key)

    def _del_txn(self, ind, db_key):
        self._open(readonly=False, lock=True)

        with self._env.begin(write=True, db=self._dbs[db_key]) as txn:
            for i in ind:
                txn.delete(serialise(i))

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
