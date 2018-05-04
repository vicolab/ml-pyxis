# -*- coding: utf-8 -*-
#
# torch.py: Implements pyxis support for PyTorch: `torch.utils.data.Dataset` and
# `torch.utils.data.DataLoader`.
#
from .pyxis import Reader

try:
    import torch
    import torch.utils.data
except ImportError:
    raise ImportError('Could not import the PyTorch library `torch` or '
                      '`torch.utils.data`. Please refer to '
                      'https://pytorch.org/ for installation instructions.')

__all__ = [
    "TorchDataset",
]


class TorchDataset(torch.utils.data.Dataset):
    """Object for interfacing with `torch.utils.data.Dataset`.

    This object allows you to wrap a pyxis LMDB as a PyTorch
    `torch.utils.data.Dataset`. The main benefit of doing so is to utilise
    the PyTorch iterator: `torch.utils.data.DataLoader`.

    Please note that all data values are converted to `torch.Tensor` type using
    the `torch.from_numpy` function in `self.__getitem__`.

    Parameter
    ---------
    dirpath : string
        Path to the directory containing the LMDB.
    """

    def __init__(self, dirpath):
        self.dirpath = dirpath

        with Reader(self.dirpath) as db:
            self.nb_samples = len(db)
            self.repr = db.__repr__()

    def __len__(self):
        return self.nb_samples

    def __getitem__(self, key):
        with Reader(self.dirpath) as db:
            data = db[key]

        for k in data.keys():
            data[k] = torch.from_numpy(data[k])

        return data

    def __repr__(self):
        return self.repr
