# -*- coding: utf-8 -*-
import os
import re
from setuptools import (find_packages, setup)


path = os.path.abspath(os.path.dirname(__file__))
readme = open(os.path.join(path, 'README.rst')).read()

try:
    with open(os.path.join(path, 'pyxis', '__init__.py'), 'r') as f:
        contents = f.read()
    version = re.search('__version__ = "(.*)"', contents).groups()[0]
except Exception:
    version = ''

setup(
    name='ml-pyxis',
    version=version,
    description='Tool for reading and writing datasets of tensors with '
                'MessagePack and Lightning Memory-Mapped Database (LMDB)',
    long_description='\n\n'.join(readme),
    author='Igor Barros Barbosa and Aleksander Rognhaugen',
    author_email='',
    url='https://github.com/vicolab/ml-pyxis',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    install_requires=['lmdb', 'msgpack-python>=0.4.0', 'numpy', 'six'],
)
