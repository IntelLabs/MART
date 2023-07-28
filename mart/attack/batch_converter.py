#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import abc
from typing import Callable

# TODO: Do we need to copy batch?

__all__ = [
    "TensorBatchConverter",
    "DictBatchConverter",
    "ListBatchConverter",
    "TupleBatchConverter",
]


class BatchConverter(abc.ABC):
    def __init__(
        self,
        *,
        transform: Callable = None,
        untransform: Callable = None,
        target_transform: Callable = None,
        target_untransform: Callable = None,
    ):
        """Convert batch into (input, target), and vice versa.

        Args:
            transform (Callable): Transform input into a convenient format, e.g. [0,1]->[0.255].
            untransform (Callable): Transform adversarial input in the convenient format back into the original format of input, e.g. [0,255]->[0,1].
            target_transform (Callable): Transform target.
            target_untransform (Callable): Untransform target.
        """
        self.transform = transform if isinstance(transform, Callable) else lambda x: x
        self.untransform = untransform if isinstance(untransform, Callable) else lambda x: x

        self.target_transform = (
            target_transform if isinstance(target_transform, Callable) else lambda x: x
        )
        self.target_untransform = (
            target_untransform if isinstance(target_untransform, Callable) else lambda x: x
        )

    def __call__(self, batch):
        input, target = self._convert(batch)
        input_transformed = self.transform(input)
        target_transformed = self.target_transform(target)
        return input_transformed, target_transformed

    def revert(self, input_transformed, target_transformed):
        input = self.untransform(input_transformed)
        target = self.target_untransform(target_transformed)
        batch = self._revert(input, target)
        return batch

    @abc.abstractclassmethod
    def _revert(self, input, target):
        pass

    @abc.abstractclassmethod
    def _convert(self, batch):
        pass


class TensorBatchConverter(BatchConverter):
    def _convert(self, batch):
        input = batch
        target = None
        return input, target

    def _revert(self, input, target):
        batch = input
        return batch


class DictBatchConverter(BatchConverter):
    def __init__(self, input_key: str = "input", **kwargs):
        """_summary_

        Args:
            input_key (str): Input locator in a batch. Defaults to "input".
        """
        super().__init__(**kwargs)

        self.input_key = input_key
        self.rest = {}

    def _convert(self, batch):
        input = batch.pop(self.input_key)
        if "target" in batch:
            target = batch["target"]
            self.rest = batch
        else:
            target = batch
        return input, target

    def _revert(self, input, target):
        if self.rest == {}:
            batch = target
        else:
            batch = self.rest

        # Input may have been changed.
        batch[self.input_key] = input

        return batch


class ListBatchConverter(BatchConverter):
    def __init__(self, input_key: int = 0, **kwargs):
        super().__init__(**kwargs)

        self.input_key = input_key
        self.target_size = None

    def _convert(self, batch: list):
        input = batch.pop(self.input_key)
        self.target_size = len(batch)

        if self.target_size == 1:
            target = batch[0]
        else:
            target = batch

        return input, target

    def _revert(self, input, target):
        if self.target_size == 1:
            batch = [target]
            batch.insert(self.input_key, input)
        else:
            batch = target
            batch.insert(self.input_key, input)
        return batch


class TupleBatchConverter(ListBatchConverter):
    def _convert(self, batch: tuple):
        batch_list = list(batch)
        input, target = super()._convert(batch_list)
        return input, target

    def _revert(self, input, target):
        batch_list = super()._revert(input, target)
        batch = tuple(batch_list)
        return batch
