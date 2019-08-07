#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re
from setuptools import find_packages, setup


# Derived definitions
root = os.path.abspath(os.path.dirname(__file__))

try:
    with open(os.path.join(root, "README.md"), "r", encoding="utf-8") as f:
        README = f.read()
except IOError:
    README = ""

try:
    with open(os.path.join(root, "pyxis", "__init__.py"), "r") as f:
        contents = f.read()
    version = re.search('__version__ = "(.*)"', contents).groups()[0]
except Exception:
    version = ""

# Setup specification
setup(
    name="ml-pyxis",
    version=version,
    description="Tool for reading and writing datasets of tensors with "
    "MessagePack and Lightning Memory-Mapped Database (LMDB)",
    long_description=README,
    author="Igor Barros Barbosa and Aleksander Rognhaugen",
    author_email="",
    license="MIT",
    url="https://github.com/vicolab/ml-pyxis",
    package_dir={"": "src"},
    include_package_data=False,
    zip_safe=True,
    packages=find_packages("src"),
    install_requires=[
        "lmdb>=0.95",
        "msgpack>=0.5.2",
        "msgpack-numpy>=0.4.4.3",
        "numpy>=1.9.0",
    ],
    extras_require={
        "test": ["pytest", "pytest-cov"],
        "docs": ["sphinx>=2.0.0", "sphinx_rtd_theme", "sphinx-autodoc-typehints"],
    },
    python_requires=">=3.6",
)
