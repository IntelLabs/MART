#
# Copyright (C) 2022 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

import logging

logger = logging.getLogger(__name__)

import pytorch_lightning as pl  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from torch.utils.data import DataLoader, Dataset, Sampler  # noqa: E402

__all__ = ["LitDataModule"]


class LitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        test_dataset=None,
        train_sampler=None,
        val_sampler=None,
        test_sampler=None,
        num_workers=0,
        collate_fn=None,
        ims_per_batch=1,
        world_size=1,
        pin_memory=False,
    ):
        super().__init__()

        if test_dataset is None:
            test_dataset = val_dataset

        self.train_dataset = train_dataset
        self.train_sampler = train_sampler

        self.val_dataset = val_dataset
        self.val_sampler = val_sampler

        self.test_dataset = test_dataset
        self.test_sampler = test_sampler

        self.num_workers = num_workers
        self.collate_fn = collate_fn
        if not callable(self.collate_fn):
            self.collate_fn = instantiate(self.collate_fn)

        world_size = max(world_size, 1)
        if ims_per_batch % world_size != 0:
            raise AttributeError(
                f"world_size ({world_size}) must be multiple of ims_per_batch ({ims_per_batch})"
            )

        self.batch_size = ims_per_batch // world_size

        logger.info(
            "Effective batch size is %d so batch size per GPU is %d",
            ims_per_batch,
            self.batch_size,
        )

        self.pin_memory = pin_memory

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            if not isinstance(self.train_dataset, Dataset):
                self.train_dataset = instantiate(self.train_dataset)
            if not isinstance(self.val_dataset, Dataset):
                self.val_dataset = instantiate(self.val_dataset)

        if stage == "test" or stage is None:
            if not isinstance(self.test_dataset, (Dataset, type(None))):
                self.test_dataset = instantiate(self.test_dataset)

    def train_dataloader(self):
        batch_sampler = self.train_sampler
        if not isinstance(batch_sampler, (Sampler, type(None))):
            batch_sampler = instantiate(batch_sampler, self.train_dataset)

        kwargs = {"batch_sampler": batch_sampler, "pin_memory": self.pin_memory}

        if batch_sampler is None:
            kwargs["batch_size"] = self.batch_size
            kwargs["shuffle"] = True
            kwargs["drop_last"] = True

        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            **kwargs,
        )

    def val_dataloader(self):
        batch_sampler = self.val_sampler
        if not isinstance(batch_sampler, (Sampler, type(None))):
            batch_sampler = instantiate(batch_sampler, self.val_dataset)

        kwargs = {"batch_sampler": batch_sampler, "pin_memory": self.pin_memory}

        if batch_sampler is None:
            kwargs["batch_size"] = self.batch_size
            kwargs["shuffle"] = False

        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            **kwargs,
        )

    def test_dataloader(self):
        batch_sampler = self.test_sampler
        if not isinstance(batch_sampler, (Sampler, type(None))):
            batch_sampler = instantiate(batch_sampler, self.test_dataset)

        kwargs = {"batch_sampler": batch_sampler, "pin_memory": self.pin_memory}

        if batch_sampler is None:
            kwargs["batch_size"] = self.batch_size
            kwargs["shuffle"] = False

        return DataLoader(
            self.test_dataset,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            **kwargs,
        )
