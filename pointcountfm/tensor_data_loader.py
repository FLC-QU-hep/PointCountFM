from collections.abc import Iterable, Iterator

import torch
from torch import Tensor


class TensorDataloader(Iterable):
    class TensorIterator(Iterator):
        def __init__(self, dataloader: "TensorDataloader") -> None:
            self.dataloader = dataloader
            self.i = 0
            if self.dataloader.shuffle:
                self.idx = torch.randperm(dataloader.n)
            else:
                self.idx = torch.arange(dataloader.n)

        def __iter__(self) -> "TensorDataloader.TensorIterator":
            return self

        def __next__(self) -> tuple[Tensor, ...]:
            if self.i >= self.dataloader.num_batches:
                raise StopIteration
            start = self.i * self.dataloader.batch_size
            end = min((self.i + 1) * self.dataloader.batch_size, self.dataloader.n)
            idx = self.idx[start:end]
            self.i += 1
            return tuple(tensor[idx].detach() for tensor in self.dataloader.tensors)

    def __init__(
        self,
        tensors: tuple[Tensor, ...],
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        r"""
        Initialize the dataloader.

        Args:
            tensors (tuple[Tensor, ...]): The tensors to be loaded.
            batch_size (int): The batch size.
            shuffle (bool): Whether to shuffle the data.
            drop_last (bool): Whether to drop the last batch if it is smaller than `batch_size`.
        """
        if not all(len(tensors[0]) == len(tensor) for tensor in tensors):
            raise ValueError("All tensors must have the same length")
        self.tensors = tensors
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.n = len(tensors[0])
        if drop_last:
            self.num_batches = self.n // batch_size
        else:
            self.num_batches = (self.n + batch_size - 1) // batch_size

    def __iter__(self) -> TensorIterator:
        return self.TensorIterator(self)

    def __len__(self) -> int:
        return self.num_batches
