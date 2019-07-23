# -*- coding: utf-8 -*-
#
# pyxis.py: Tool for reading and writing datasets of tensors (`numpy.ndarray`)
#           with MessagePack and Lightning Memory-Mapped Database (LMDB).
#
from __future__ import division, print_function

import numpy as np

try:
    import lmdb
except ImportError:
    raise ImportError(
        "Could not import the LMDB library `lmdb`. Please refer "
        "to https://github.com/dw/py-lmdb/ for installation "
        "instructions."
    )
try:
    import msgpack
except ImportError:
    raise ImportError(
        "Could not import the MessagePack library `msgpack`. "
        "Please refer to "
        "https://github.com/msgpack/msgpack-python for "
        "installation instructions."
    )

__all__ = ["Reader", "Writer"]


def encode_str(string, encoding="utf-8", errors="strict"):
    """Return an encoded byte object of the input string.

    Parameters
    ----------
    string : string
    encoding : string
        Default is `utf-8`.
    errors : string
        Specifies how encoding errors should be handled. Default is `strict`.
    """
    return str(string).encode(encoding=encoding, errors=errors)


def decode_str(obj, encoding="utf-8", errors="strict"):
    """Decode the input byte object to a string.

    Parameters
    ----------
    obj : byte object
    encoding : string
        Default is `utf-8`.
    errors : string
        Specifies how encoding errors should be handled. Default is `strict`.
    """
    return obj.decode(encoding=encoding, errors=errors)


# Supported types for serialisation
TYPES = {"str": 1, "ndarray": 2}

# Default number of databases
NB_DBS = 2

# Name of the default database(s)
DATA_DB = encode_str("data_db")
META_DB = encode_str("meta_db")

# Default key(s) for metadata
NB_SAMPLES = encode_str("nb_samples")


def encode_data(obj):
    """Return a dictionary with information encoding the input data object.

    Parameter
    ---------
    obj : data object
        If the incoming data object is neither a string nor an ordinary NumPy
        array, then the object will simply be returned as is.
    """
    if isinstance(obj, str):
        return {b"type": TYPES["str"], b"data": obj}
    elif isinstance(obj, np.ndarray):
        return {
            b"type": TYPES["ndarray"],
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
            b"data": obj.tobytes(),
        }
    else:
        # Assume the user know what they are doing
        return obj


def decode_data(obj):
    """Decode a serialised data object.

    Parameter
    ---------
    obj : Python dictionary
        A dictionary describing a serialised data object.
    """
    try:
        if TYPES["str"] == obj[b"type"]:
            return obj[b"data"]
        elif TYPES["ndarray"] == obj[b"type"]:
            return np.fromstring(obj[b"data"], dtype=np.dtype(obj[b"dtype"])).reshape(obj[b"shape"])
        else:
            # Assume the user know what they are doing
            return obj
    except KeyError:
        # Assume the user know what they are doing
        return obj


class Reader(object):
    """Object for reading a dataset of tensors (`numpy.ndarray`).

    The tensors are read from a Lightning Memory-Mapped Database (LMDB) with
    the help of MessagePack.

    Parameter
    ---------
    dirpath : string
        Path to the directory containing the LMDB.

    lock : bool
        Either to use lock blocking methods on the reader.
        If False make sure you do not have co-occurent write to the dataset while reading it.

    """

    def __init__(self, dirpath, lock=True):
        self.dirpath = dirpath

        # Open LMDB environment in read-only mode
        self._lmdb_env = lmdb.open(dirpath, readonly=True, max_dbs=NB_DBS, lock=lock)

        # Open the default database(s) associated with the environment
        self.data_db = self._lmdb_env.open_db(DATA_DB)
        self.meta_db = self._lmdb_env.open_db(META_DB)

        # Read the metadata
        self.nb_samples = int(self.get_meta_str(NB_SAMPLES))

    def get_meta_str(self, key):
        """Return the value associated with the input key in as a string.

        The value is retrieved from `meta_db`.

        Parameter
        ---------
        key : string or bytestring
        """
        if isinstance(key, str):
            _key = encode_str(key)
        else:
            _key = key

        with self._lmdb_env.begin(db=self.meta_db) as txn:
            _k = txn.get(_key)
            if isinstance(_k, bytes):
                return decode_str(_k)
            else:
                return str(_k)

    def get_data_keys(self, i=0):
        """Return a list of the keys for the ith sample in `data_db`.

        If all samples contain the same keys, then we only need to check
        the first sample, hence the `i=0` default value.

        Parameter
        ---------
        i : int, optional
        """
        return list(self[i].keys())

    def get_data_value(self, i, key):
        """Return the value associated with the input key for the ith sample.

        The value is retrieved from `data_db`.

        Because each sample is stored in a msgpack, we will need to read the
        whole msgpack before returning the value.

        Parameters
        ----------
        i : int
        key : string
        """
        try:
            return self[i][key]
        except KeyError:
            raise KeyError("Key does not exist: {}".format(key))

    def get_data_specification(self, i):
        """Return the specification of all data objects for the ith sample.

        The specification includes the shape and data type. This assumes each
        data object is a `numpy.ndarray`.

        Parameter
        ---------
        i : int
        """
        spec = {}
        sample = self[i]
        for key in sample.keys():
            spec[key] = {}
            try:
                spec[key]["dtype"] = sample[key].dtype
                spec[key]["shape"] = sample[key].shape
            except KeyError:
                raise KeyError("Key does not exist: {}".format(key))

        return spec

    def get_sample(self, i):
        """Return the ith sample from `data_db`.

        Parameter
        ---------
        i : int
        """
        if 0 > i or self.nb_samples <= i:
            raise IndexError("The selected sample number is out of range: %d" % i)

        # Convert the sample number to a string with trailing zeros
        key = encode_str("{:010}".format(i))

        obj = {}
        with self._lmdb_env.begin(db=self.data_db) as txn:
            # Read msgpack from LMDB and decode each value in it
            _obj = msgpack.unpackb(txn.get(key), raw=False, use_list=True)
            for k in _obj:
                # Keys must be decoded if they are stored as byte objects
                if isinstance(k, bytes):
                    _k = decode_str(k)
                else:
                    _k = str(k)
                obj[_k] = msgpack.unpackb(
                    _obj[_k], raw=False, use_list=False, object_hook=decode_data
                )

        return obj

    def get_samples(self, i, size):
        """Return all consecutive samples from `i` to `i + size`.

        Assumptions:
        * Every sample from `i` to `i + size` have the same set of keys.
        * All data objects in a sample are of the type `numpy.ndarray`.
        * Values associated with the same key have the same tensor shape and
          data type.

        Parameters
        ----------
        i : int
        size : int
        """
        if 0 > i or self.nb_samples <= i + size - 1:
            raise IndexError(
                "The selected sample number is out of range: %d "
                " to %d (size: %d)" % (i, i + size, size)
            )

        # The assumptions about the data will be made based on the ith sample
        samples = {}
        _sample = self[i]
        for key in _sample:
            samples[key] = np.zeros((size,) + _sample[key].shape, dtype=_sample[key].dtype)
            samples[key][0] = _sample[key]

        with self._lmdb_env.begin(db=self.data_db) as txn:
            pos = 1  # The first position was filled above
            for _i in range(i + 1, i + size):
                # Convert the sample number to a string with trailing zeros
                key = encode_str("{:010}".format(_i))

                # Read msgpack from LMDB, decode each value in it, and add it
                # to the set of retrieved samples
                obj = msgpack.unpackb(txn.get(key), raw=False, use_list=True)
                for k in obj:
                    # Keys must be decoded if they are stored as byte objects
                    if isinstance(k, bytes):
                        _k = decode_str(k)
                    else:
                        _k = str(k)
                    samples[_k][pos] = msgpack.unpackb(
                        obj[_k], raw=False, use_list=False, object_hook=decode_data
                    )

                pos += 1

        return samples

    def __getitem__(self, key):
        """Return sample(s) from `data_db` using `get_sample()`.

        Parameter
        ---------
        key : int or slice object
        """
        if isinstance(key, (int, np.integer)):
            _key = int(key)
            if 0 > _key:
                _key += len(self)
            if 0 > _key or len(self) <= _key:
                raise IndexError("The selected sample is out of range: `{}`".format(key))
            return self.get_sample(_key)
        elif isinstance(key, slice):
            return [self[i] for i in range(*key.indices(len(self)))]
        else:
            raise TypeError("Invalid argument type: `{}`".format(type(key)))

    def __getslice__(self, i, j):
        """Python 2.* slicing compatibility: delegate to `__getitem__`.

        This method is deprecated by Python, so functionality may vary
        compared with `__getitem__`. Use with caution.
        """
        return self.__getitem__(slice(i, j, None))

    def __len__(self):
        """Return the number of samples in the dataset.
        """
        return self.nb_samples

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __repr__(self):
        spec = self.get_data_specification(0)
        out = "pyxis.{}\n".format(self.__class__.__name__)
        out += "Location:\t\t'{}'\n".format(self.dirpath)
        out += "Number of samples:\t{}\n".format(len(self))
        out += "Data keys (0th sample):"
        for key in self.get_data_keys():
            out += "\n\t'{}' <- dtype: {}, shape: {}".format(
                key, spec[key]["dtype"], spec[key]["shape"]
            )
        return out

    def close(self):
        """Close the environment.

        Invalidates any open iterators, cursors, and transactions.
        """
        self._lmdb_env.close()


class Writer(object):
    """Object for writing a dataset of tensors (`numpy.ndarray`).

    The tensors are written to a Lightning Memory-Mapped Database (LMDB) with
    the help of MessagePack.

    Parameters
    ----------
    dirpath : string
        Path to the directory where the LMDB should be written.
    map_size_limit : int
        Map size for LMDB in MB. Must be big enough to capture all of the data
        intended to be stored in the LMDB.
    ram_gb_limit : float
        The maximum size of the data that be put in RAM at the same time. The
        size of the data this object tries to write cannot exceed this number.
        Default is `2` GB.
    """

    def __init__(self, dirpath, map_size_limit, ram_gb_limit=2):
        self.dirpath = dirpath
        self.map_size_limit = int(map_size_limit)  # Megabytes (MB)
        self.ram_gb_limit = float(ram_gb_limit)  # Gigabytes (GB)
        self.keys = []
        self.nb_samples = 0

        # Minor sanity checks
        if self.map_size_limit <= 0:
            raise ValueError(
                "The LMDB map size must be positive: " "{}".format(self.map_size_limit)
            )
        if self.ram_gb_limit <= 0:
            raise ValueError(
                "The RAM limit (GB) per write must be " "positive: {}".format(self.ram_gb_limit)
            )

        # Convert `map_size_limit` from MB to B
        map_size_limit <<= 20

        # Open LMDB environment
        self._lmdb_env = lmdb.open(dirpath, map_size=map_size_limit, max_dbs=NB_DBS)

        # Open the default database(s) associated with the environment
        self.data_db = self._lmdb_env.open_db(DATA_DB)
        self.meta_db = self._lmdb_env.open_db(META_DB)

    def put_samples(self, *args):
        """Put the incoming argument keys and values in the `data_db` LMDB.

        The user can input their data in two different ways:
        * As a Python dictionary:
            * `put_samples({'key1': value1, 'key2': value2, ...})`
        * Alternating keys and values:
            * `put_samples('key1', value1, 'key2', value2, ...)`

        The function assumes that the first axis in all values represent the
        sample number. For that reason, a single sample must be prepended with
        a `numpy.newaxis`.

        Parameter
        ---------
        *args: see above
        """
        # Select `*args` style
        if len(args) == 1 and isinstance(args[0], dict):
            samples = args[0]
        else:
            if not len(args) % 2 == 0:
                raise ValueError(
                    "Each data object must be associated with a "
                    "key, e.g. `put_samples(key1, value1, key2, "
                    "value2, ...)`"
                )
            # Convert to `{key1: value1, key2: value2, ...}` format
            samples = dict((a, b) for a, b in zip(args[0::2], args[1::2]))

        # Sanity checks #
        gb_required = 0
        nb_elems = []
        for key in samples:
            # All data objects must have the type: `numpy.ndarray`
            if not isinstance(samples[key], np.ndarray):
                raise ValueError(
                    "Data object type not supported: " "`numpy.ndarray` != %s" % type(samples[key])
                )
            else:
                gb_required += np.uint64(samples[key].nbytes)
                nb_elems.append(samples[key].shape[0])

        # Ensure that the hypothetical RAM size specified by the user can
        # handle the number of samples being stored
        gb_required = float(gb_required / 10 ** 9)
        if self.ram_gb_limit < gb_required:
            raise ValueError(
                "The size of the data being written is larger "
                "than `ram_gb_limit`: %d < %f" % (self.ram_gb_limit, gb_required)
            )

        # The number of data elements must be the same over all data objects
        if len(nb_elems) != nb_elems.count(nb_elems[0]):
            raise ValueError(
                "The number of data elements must be the same " "over all data objects."
            )

        try:
            # For each sample, build a msgpack and store it in the LMDB
            with self._lmdb_env.begin(write=True, db=self.data_db) as txn:
                for i in range(nb_elems[0]):
                    # Build a msgpack for each data object
                    msg_pkgs = {}
                    for key in samples:
                        # Ensure the current sample is a `numpy.ndarray`
                        obj = samples[key][i]
                        if not isinstance(obj, np.ndarray):
                            obj = np.array(obj)

                        # Create msgpack
                        msg_pkgs[key] = msgpack.packb(obj, use_bin_type=True, default=encode_data)

                    # LMDB key: sample number as a string with trailing zeros
                    key = encode_str("{:010}".format(self.nb_samples))

                    # Construct final msgpack and store it in the LMDB
                    pkg = msgpack.packb(msg_pkgs, use_bin_type=True)
                    txn.put(key, pkg)

                    # Increase global sample counter
                    self.nb_samples += 1
        except lmdb.MapFullError as e:
            raise AttributeError(
                "The LMDB `map_size` is too small: " "%s MB, %s" % (self.map_size_limit, e)
            )

        # Write the current number of samples to `meta_db` just in case
        self.set_meta_str(NB_SAMPLES, self.nb_samples)

    def set_meta_str(self, key, string):
        """Write the input string to the input key in `meta_db`.

        Parameters
        ----------
        key : string or bytestring
        string : string
        """
        if isinstance(key, str):
            _key = encode_str(key)
        else:
            _key = key

        with self._lmdb_env.begin(write=True, db=self.meta_db) as txn:
            txn.put(_key, encode_str(string))

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __repr__(self):
        out = "pyxis.{}\n".format(self.__class__.__name__)
        out += "Location:\t\t'{}'\n".format(self.dirpath)
        out += "LMDB map size (MB):\t{}\n".format(self.map_size_limit)
        out += "RAM limit (GB):\t\t{}".format(self.ram_gb_limit)
        return out

    def close(self):
        """Close the environment.

        Before closing, the number of samples is written to `meta_db`.

        Invalidates any open iterators, cursors, and transactions.
        """
        self.set_meta_str(NB_SAMPLES, self.nb_samples)
        self._lmdb_env.close()
