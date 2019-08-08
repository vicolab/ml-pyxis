# -*- coding: utf-8 -*-
"""test_pyxis.py: Testing of pyxis.py.
"""
import pytest


def test_serialise():
    import msgpack
    import numpy as np
    from pyxis import serialise

    # bytes -> bytes
    assert b"abc" == serialise(b"abc")
    assert bytes([1, 2, 3]) == serialise(bytes([1, 2, 3]))

    # dict -> data
    data = {"X": np.ones((3, 4, 5)), "y": "testing"}
    data_pkg = {k: msgpack.packb(data[k], use_bin_type=True) for k in data}
    assert msgpack.packb(data_pkg, use_bin_type=True) == serialise(data)

    # other -> value
    assert msgpack.packb([1, 2, 3], use_bin_type=True) == serialise([1, 2, 3])
    arr = np.array([[1, 2], [3, 4]])
    assert msgpack.packb(arr, use_bin_type=True) == serialise(arr)


def test_deserialise():
    import msgpack
    import numpy as np
    from pyxis import deserialise

    # value -> value
    assert "abc" == deserialise("abc")
    assert [1, 2, 3] == deserialise([1, 2, 3])

    # sample -> data
    data = {"X": np.ones((3, 4, 5)), "y": "testing"}
    data_pkg = {k: msgpack.packb(data[k], use_bin_type=True) for k in data}
    data_pkg = msgpack.packb(data_pkg, use_bin_type=True)
    out = deserialise(data_pkg, is_sample=True)
    assert data.keys() == out.keys()
    assert np.array_equal(data["X"], out["X"])
    assert data["y"] == out["y"]

    # bytes -> value
    value_pkg = msgpack.packb([1, 2, 3], use_bin_type=True)
    assert [1, 2, 3] == deserialise(value_pkg, is_sample=False)
    arr = np.array([[1, 2], [3, 4]])
    value_pkg = msgpack.packb(arr, use_bin_type=True)
    assert np.array_equal(arr, deserialise(value_pkg, is_sample=False))


def test_abstractpyxis():
    from pyxis import AbstractPyxis

    with pytest.raises(TypeError):
        AbstractPyxis()


def test_alias():
    from pyxis import LMDBPyxis, Pyxis

    assert LMDBPyxis == Pyxis


def test_lmdbpyxis_constructor(temp_dir):
    import lmdb
    import numpy as np
    from pyxis import LMDBPyxis

    db = LMDBPyxis(temp_dir)

    assert temp_dir == db.path
    assert 1 * LMDBPyxis.BYTES_IN_MB == db.default_map_size
    assert 8192 * LMDBPyxis.BYTES_IN_MB == db.max_map_size
    assert db.iter_shuffle is False
    assert np.random == db.iter_rng
    assert isinstance(db._env, lmdb.Environment)
    assert isinstance(db._dbs, dict)
    assert db._isdir is True
    assert LMDBPyxis.DB_KEYS == list(db._dbs.keys())

    db.close()


def test_lmdbpyxis_is_open(temp_dir):
    from pyxis import LMDBPyxis

    db = LMDBPyxis(temp_dir)

    assert db.is_open is True

    db.close()
    assert db.is_open is False

    db._open(readonly=True, lock=False)
    assert db.is_open is True

    # Precaution: this cannot happen during normal use
    db._env.close()
    assert db.is_open is False

    db.close()


def test_lmdbpyxis_is_readonly(temp_dir):
    import lmdb
    from pyxis import LMDBPyxis

    db = LMDBPyxis(temp_dir)

    db._open(readonly=True, lock=False)
    assert db.is_readonly is True

    db._open(readonly=False, lock=False)
    assert db.is_readonly is False

    db.close()

    with pytest.raises(lmdb.Error):
        db.is_readonly


def test_lmdbpyxis_is_locked(temp_dir):
    import lmdb
    from pyxis import LMDBPyxis

    db = LMDBPyxis(temp_dir)

    db._open(readonly=True, lock=True)
    assert db.is_locked is True

    db._open(readonly=False, lock=False)
    assert db.is_locked is False

    db.close()

    with pytest.raises(lmdb.Error):
        db.is_locked


def test_lmdbpyxis_map_size(temp_dir):
    from pyxis import LMDBPyxis

    db = LMDBPyxis(temp_dir)

    db._open(readonly=True, lock=True)
    assert 1 * LMDBPyxis.BYTES_IN_MB == db.map_size

    db.close()
    db.default_map_size *= 2
    assert 2 * LMDBPyxis.BYTES_IN_MB == db.map_size


def test_lmdbpyxis_indices(temp_dir):
    import numpy as np
    from pyxis import LMDBPyxis

    db = LMDBPyxis(temp_dir)

    # Empty db
    assert [] == db.indices

    # A few elements
    db.put("a", np.zeros((5, 1)))
    assert [0, 1, 2, 3, 4] == db.indices

    db.close()


def test_lmdbpyxis_empty(temp_dir):
    import numpy as np
    from pyxis import LMDBPyxis

    db = LMDBPyxis(temp_dir)

    # Empty db
    assert db.empty is True

    # A few elements
    db.put("a", np.zeros((1, 1)))
    assert db.empty is False

    db.close()


@pytest.mark.parametrize("val", [-1, 5, 10])
def test_lmdbpyxis_get(temp_dir, val):
    import numpy as np
    from pyxis import LMDBPyxis

    db = LMDBPyxis(temp_dir)

    # Empty
    with pytest.raises(IndexError):
        db.get(0)

    db.put("a", np.ones((5, 1)) * val, "b", np.ones((5, 1)) * val * -1j)

    # Out of bounds
    with pytest.raises(IndexError):
        db.get(val)

    # Regular `get`
    for i in db.indices:
        out = db.get(i)
        assert np.array_equal(["a", "b"], list(out.keys()))
        assert np.array_equal(out["a"], np.asarray([val]))
        assert np.array_equal(out["b"], np.asarray([val]) * -1j)

    db.close()


@pytest.mark.parametrize("val", [1, 2, -0.234])
def test_lmdbpyxis_put(temp_dir, val):
    import numpy as np
    from pyxis import LMDBPyxis

    db = LMDBPyxis(temp_dir)

    # Ensure `TypeError` on data without `__len__`
    with pytest.raises(TypeError):
        db.put("abc", 123)

    # Ensure `TypeError` on dictionary data
    with pytest.raises(TypeError):
        db.put("abc", {"b1": "test1", "b2": "test2"})

    # Varying number of samples per value
    with pytest.raises(ValueError):
        db.put("a", np.ones((5, 1, 2, 3, 7)), "b", np.zeros((3, 5)))

    # Regular `put`
    db.put("a", np.ones((5, 1)) * val, "b", np.ones((5, 1)) * val * -1j)
    for i in db.indices:
        out = db.get(i)
        assert np.array_equal(["a", "b"], list(out.keys()))
        assert np.array_equal(out["a"], np.asarray([val]))
        assert np.array_equal(out["b"], np.asarray([val]) * -1j)

    db.close()


def test_lmdbpyxis_replace(temp_dir):
    import numpy as np
    from pyxis import LMDBPyxis

    db = LMDBPyxis(temp_dir)
    db.put("a", [1, 2], "b", [3, 4])

    # Sanity check: internal get failure
    with pytest.raises(IndexError):
        db.replace(2, "b", [1, 2, 3])

    # Successful replace operation
    db.replace(0, "a", np.ones((2, 2)))
    assert [0, 1] == db.indices
    assert ["a"] == list(db.get(0).keys())
    assert ["a", "b"] == list(db.get(1).keys())
    assert np.array_equal(np.ones((2, 2)), db.get(0)["a"])
    assert 2 == db.get(1)["a"] and 4 == db.get(1)["b"]

    db.close()


def test_lmdbpyxis_batch(temp_dir):
    import collections
    from pyxis import LMDBPyxis

    db = LMDBPyxis(temp_dir)

    # Sanity check with no data
    it = db.batch()
    assert isinstance(it, collections.abc.Iterable)
    with pytest.raises(StopIteration):
        next(it)

    db.put("a", [1, 2, 3, 4], "b", [4, 3, 2, 1])

    # Group has only a single value
    it = db.batch(1)
    for b in it:
        assert 1 == len(b)

    # The batch size must be a positive integer, we rely upon `itertools.islice` to
    # check for this
    with pytest.raises(StopIteration):
        it = db.batch(0)
        next(it)

    with pytest.raises(ValueError):
        it = db.batch(-4)
        next(it)

    with pytest.raises(ValueError):
        it = db.batch(1.5)
        next(it)

    # Group has two values (a multiple of the number of samples)
    it = db.batch(2)
    for b in it:
        assert 2 == len(b)

    # Group has three values (*not* a multiple of the number of samples)
    it = db.batch(3)
    assert 3 == len(next(it))
    assert 1 == len(next(it))

    # Group has more elements than the number of samples in the database
    it = db.batch(5)
    assert 4 == len(next(it))

    db.close()


def test_lmdbpyxis_close(temp_dir):
    from pyxis import LMDBPyxis

    db = LMDBPyxis(temp_dir)
    db.close()
    assert db._env is None
    assert db.is_open is False


def test_lmdbpyxis_render(temp_dir):
    import io
    from pyxis import LMDBPyxis

    db = LMDBPyxis(temp_dir)
    assert isinstance(db.render(), io.StringIO)
    db.close()


def test_lmdbpyxis__len__(temp_dir):
    from pyxis import LMDBPyxis

    db = LMDBPyxis(temp_dir)

    assert len(db) == 0
    db.put("a", [1])
    assert len(db) == 1
    db.put("b", [2])
    assert len(db) == 2

    db.close()


def test_lmdbpyxis__getitem__(temp_dir):
    from pyxis import LMDBPyxis

    db = LMDBPyxis(temp_dir)
    db.put("test", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    # Single key
    assert 0 == db[0]["test"]
    assert 9 == db[9]["test"]
    assert 5 == db[5]["test"]
    assert 9 == db[-1]["test"]
    assert 0 == db[-10]["test"]

    # Slice object
    assert [] == db[0:0]
    assert [] == db[-1:1]
    assert 1 == len(db[0:1]) and 0 == db[0:1][0]["test"]
    assert 1 == len(db[-1:]) and 9 == db[-1:][0]["test"]
    assert 4 == len(db[1:5])
    out = db[::-1]
    for i, v in enumerate(range(9, 0, -1)):
        assert v == out[i]["test"]

    # Erroneous type
    with pytest.raises(TypeError):
        db["test"]


def test_lmdbpyxis__setitem__(temp_dir):
    # TODO
    pass


def test_lmdbpyxis__iter__(temp_dir):
    # TODO
    pass


def test_lmdbpyxis__enter__exit__(temp_dir):
    from pyxis import LMDBPyxis

    with LMDBPyxis(temp_dir) as db:
        assert db.is_open is True

    assert db.is_open is False


def test_lmdbpyxis__str__(temp_dir):
    from pyxis import LMDBPyxis

    db = LMDBPyxis(temp_dir)
    assert isinstance(str(db), str)
    db.close()


@pytest.mark.parametrize(
    "state", [(True, True), (True, False), (False, True, False, False)]
)
def test_lmdbpyxis__setup(temp_dir, state):
    from pyxis import LMDBPyxis

    db = LMDBPyxis(temp_dir)
    db.close()
    assert db._env is None
    assert db._dbs[LMDBPyxis.MAIN_DB_KEY] is None
    assert db._dbs[LMDBPyxis.METADATA_DB_KEY] is None

    db._setup(readonly=state[0], lock=state[1])
    assert db._env is not None
    assert db._dbs[LMDBPyxis.MAIN_DB_KEY] is not None
    assert db._dbs[LMDBPyxis.METADATA_DB_KEY] is not None
    assert db.is_readonly is state[0]
    assert db.is_locked is state[1]

    db.close()


def test_lmdbpyxis__open(temp_dir):
    from pyxis import LMDBPyxis

    db = LMDBPyxis(temp_dir)
    db._setup(readonly=True, lock=False)
    assert db._open(readonly=True, lock=False) is False
    assert db._open(readonly=False, lock=True) is True

    db.close()


def test_lmdbpyxis__put_txn(temp_dir):
    # TODO
    pass


def test_lmdbpyxis__unpack_key_value_from_args(temp_dir):
    # TODO
    pass
