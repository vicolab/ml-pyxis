# -*- coding: utf-8 -*-
#
# torch.py: Implements pyxis support for PyTorch: `torch.utils.data.Dataset` and
# `torch.utils.data.DataLoader`.
#
import copy
import multiprocessing as mp
import queue
import threading
from typing import Tuple

import numpy as np

from .iterators import DataIterator, StochasticBatch
from .pyxis import Reader

try:
    import torch
    import torch.utils.data
except ImportError:
    raise ImportError(
        "Could not import the PyTorch library `torch` or "
        "`torch.utils.data`. Please refer to "
        "https://pytorch.org/ for installation instructions."
    )

__all__ = ["TorchDataset"]


class TorchDataset(torch.utils.data.Dataset):
    """Object for interfacing with `torch.utils.data.Dataset`.

    This object allows you to wrap a pyxis LMDB as a PyTorch
    `torch.utils.data.Dataset`. The main benefit of doing so is to utilise
    the PyTorch iterator: `torch.utils.data.DataLoader`.

    Please note that all data values are converted to `torch.Tensor` type using
    the `torch.from_numpy` function in `self.__getitem__`.

    It is not safe to use this dataset along with a dataset writer.
    Make sure you are only reading from the dataset while using the class.

    Parameter
    ---------
    dirpath : string
        Path to the directory containing the LMDB.
    """

    def __init__(self, dirpath):
        self.dirpath = dirpath
        self.db = Reader(self.dirpath, lock=False)

    def __len__(self):
        return len(self.db)

    def __getitem__(self, key):
        data = self.db[key]
        for k in data.keys():
            data[k] = torch.from_numpy(data[k])

        return data

    def __repr__(self):
        return str(self.db)


exit_signal = mp.Event()


class TorchIterator:
    """A cuda safe and scalable iterator for Pyxis/pytorch.
    """

    def __init__(
        self,
        device: torch.device,
        dir_path: str,
        keys: tuple,
        batch_size: int = 32,
        multiplier: int = 10,
        num_worker: int = 2,
        pre_fetcher_queue: int = 1000,
        device_transfer_queue: int = 1,
    ):
        """
        A cuda safe and scalable iterator for Pyxis/pytorch.

        Parameters
        ----------
        device : torch.device
            To which torch device should the data be sent to.
        dir_path : str
            String with path to dataset
        keys : tuple
            Which dataset keys should be fetched
        batch_size : int, optional
            batch size, by default 32
        multiplier : int, optional
            Multiplier for batch, by default 10.
            How many times bigger should the pre-fetcher and device transfer `batch fetch`.
        num_worker : int, optional
            Number of workers used to fetch batchs*multiplier, by default 2
        pre_fetcher_queue : int, optional
            Number of batched*multiliers that can be pre-stored into ram, by default 1000
        device_transfer_queue : int, optional
            How many batch*multipliers can be transfered to the `device` at a time, by default 1

        Returns
        -------
        PyxisTorchIterator
            A Pyxis dataset iterator.

        Raises
        ------
        RuntimeError
            Error is raised if the dataset has been signaled to stop pre-fetching and it is tried to be used.
        StopIteration
            Default error needed to stop dataset iteration once the whole dataset has been processed.
        """
        self.multiplier = multiplier
        db = Reader(dir_path, False)
        self.nb_samples = db.nb_samples
        self.batches_per_epoch = np.ceil(db.nb_samples / batch_size)
        self.info = db.__repr__()
        db.close()
        self.batch_size = batch_size
        self.exit = exit_signal
        print(self.exit)
        self.batch_counter = 0
        self.processes = []
        self.device_transfer_queue = queue.Queue(maxsize=device_transfer_queue)
        self.device = device

        ctx = mp.get_context("spawn")
        self.pre_fetcher_queue = ctx.Queue(maxsize=pre_fetcher_queue)
        for _ in range(num_worker):
            p = ctx.Process(
                target=self.cpu_fetch_batches,
                args=(self.pre_fetcher_queue, dir_path, keys, batch_size * self.multiplier,),
            )
            p.start()
            self.processes.append(p)

        self.cuda_thread = threading.Thread(target=self.from_pre_fetch_to_device)
        self.cuda_thread.start()

    def __len__(self):
        return int(self.batches_per_epoch)

    def __iter__(self):
        return self

    @staticmethod
    def cpu_fetch_batches(local_queue, dir_path, keys, batch_size):
        db = Reader(dir_path, False)
        gen = StochasticBatch(db, keys=keys, batch_size=batch_size)
        while exit_signal.is_set() == False:
            data = next(gen)
            data_ = copy.deepcopy(data)
            try:
                local_queue.put(data_, block=True, timeout=2)
            except queue.Full:
                if exit_signal.is_set():
                    return

    def from_pre_fetch_to_device(self):
        "Tansfer from pre_fetcher_queue to device_transfer_queue"
        while self.exit.is_set() == False:
            # get from  device
            if not self.exit.is_set():
                data = self.pre_fetcher_queue.get(block=True)
            else:
                return
            # to torch and then to device
            if isinstance(data, np.ndarray):
                tensor = torch.from_numpy(data).to(self.device)
                # divide the by pre_fetcher multiplier
                for chunk in torch.chunk(tensor, self.multiplier):
                    self.device_transfer_queue.put(chunk, block=True)

            elif isinstance(data, tuple):
                holder = {}
                for nb_keys, item in enumerate(data):
                    # send to device
                    tensor = torch.from_numpy(item).to(self.device)
                    for nb_chunks, chunk in enumerate(torch.chunk(tensor, self.multiplier)):
                        # slice into chunks
                        holder[(nb_keys, nb_chunks)] = torch.from_numpy(item).to(self.device)
                nb_keys = nb_keys + 1
                nb_chunks = nb_chunks + 1

                # post tuple of chunk
                for chunk in range(nb_chunks):
                    sample = []
                    for key in range(nb_keys):
                        sample.append(holder[key, chunk])
                    try:
                        self.device_transfer_queue.put(tuple(sample), block=True, timeout=2)
                    except queue.Full:
                        if self.exit.is_set():
                            return

            else:
                raise ValueError("Can't process iterator of type" + type(data))

    def __next__(self):
        data = None
        if self.exit.is_set():
            raise RuntimeError("Processing has finished. (done = True)")

        if self.batch_counter < self.batches_per_epoch:
            # get data that was sent to device
            data = self.device_transfer_queue.get()
            self.batch_counter += 1
        else:
            self.batch_counter = 0
            raise StopIteration
        return data

    def close(self):
        """Close iterator threads
        """
        self.exit.set()

    def __del__(self):
        self.close()
        for process in self.processes:
            process.terminate()
        self.pre_fetcher_queue.close()

        while not self.device_transfer_queue.empty():
            self.device_transfer_queue.get(False)
        self.device_transfer_queue.task_done()
