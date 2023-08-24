#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import abc
from typing import Callable

# TODO: Do we need to copy batch?

__all__ = [
    "InputOnlyBatchConverter",
    "DictBatchConverter",
    "ListBatchConverter",
    "TupleBatchConverter",
]


class BatchConverter(abc.ABC):
    def __init__(
        self,
        *,
        transform: Callable | None = None,
        untransform: Callable | None = None,
        target_transform: Callable | None = None,
        target_untransform: Callable | None = None,
        batch_transform: Callable | None = None,
        batch_untransform: Callable | None = None,
    ):
        """Convert batch into (input, target), and vice versa.

        Args:
            transform (Callable): Transform input into a convenient format, e.g. normalized_input->[0, 255].
            untransform (Callable): Transform adversarial input in the convenient format back into the original format of input, e.g. [0,255]->normalized_input.
            target_transform (Callable): Transform target.
            target_untransform (Callable): Untransform target.
            batch_transform (Callable): Transform batch before converting the batch.
            batch_untransform (callable): Untransform batch after reverting the batch.
        """

        self.transform = transform
        self.untransform = untransform

        self.target_transform = target_transform
        self.target_untransform = target_untransform

        self.batch_transform = batch_transform
        self.batch_untransform = batch_untransform

    def __call__(self, batch, device=None):
        if self.batch_transform is not None:
            batch = self.batch_transform(batch, device=device)

        input, target = self._convert(batch)

        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return input, target

    def revert(self, input, target):
        if self.untransform is not None:
            input = self.untransform(input)
        if self.target_untransform is not None:
            target = self.target_untransform(target)

        batch = self._revert(input, target)

        if self.batch_untransform is not None:
            batch = self.batch_untransform(batch)

        return batch

    @abc.abstractclassmethod
    def _revert(self, input, target):
        pass

    @abc.abstractclassmethod
    def _convert(self, batch):
        pass


class InputOnlyBatchConverter(BatchConverter):
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
        # Make a copy because we don't want to break the original batch.
        batch = batch.copy()
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
    def __init__(self, input_key: int = 0, target_size: int | None = None, **kwargs):
        super().__init__(**kwargs)

        self.input_key = input_key
        self.target_size = target_size

    def _convert(self, batch: list):
        # Make a copy because we don't want to break the original batch.
        batch = batch.copy()
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
