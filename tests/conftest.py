# -*- coding: utf-8 -*-
"""conftest.py

The function `cleanup` and the `temp_dir` fixture is highly inspired by the `testlib`
file in https://github.com/jnwatson/py-lmdb .

The full license in `tests/testlib.py` can be seen below:
"""
#
# Copyright 2013 The py-lmdb authors, all rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted only as authorized by the OpenLDAP
# Public License.
#
# A copy of this license is available in the file LICENSE in the
# top-level directory of the distribution or, alternatively, at
# <http://www.OpenLDAP.org/license.html>.
#
# OpenLDAP is a registered trademark of the OpenLDAP Foundation.
#
# Individual files and/or contributed packages may be copyright by
# other parties and/or subject to additional restrictions.
#
# This work also contains materials derived from public sources.
#
# Additional information about OpenLDAP can be obtained at
# <http://www.openldap.org/>.
#
import atexit
import shutil
import traceback

import pytest

_cleanups = []


def cleanup():
    while _cleanups:
        func = _cleanups.pop()
        try:
            func()
        except Exception:
            traceback.print_exc()


atexit.register(cleanup)


@pytest.fixture(scope="function")
def temp_dir():
    import tempfile
    import sys

    path = tempfile.mkdtemp(prefix="pyxis")

    _cleanups.append(lambda: shutil.rmtree(path, ignore_errors=True))
    if hasattr(path, "decode"):
        path = path.decode(sys.getfilesystemencoding())

    return path
