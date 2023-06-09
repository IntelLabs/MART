#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from operator import attrgetter

from pytorch_lightning.callbacks import Callback

__all__ = ["ImageVisualizer"]


class ImageVisualizer(Callback):
    def __init__(self, frequency: int = 100, **tag_paths):
        self.frequency = frequency
        self.tag_paths = tag_paths

    def log_image(self, trainer, tag, image):
        # Add image to each logger
        for logger in trainer.loggers:
            # FIXME: Should we just use isinstance(logger.experiment, SummaryWriter)?
            if not hasattr(logger.experiment, "add_image"):
                continue

            logger.experiment.add_image(tag, image, global_step=trainer.global_step)

    def log_images(self, trainer, pl_module):
        for tag, path in self.tag_paths.items():
            image = attrgetter(path)(pl_module)
            self.log_image(trainer, tag, image)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.frequency != 0:
            return

        self.log_images(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        self.log_images(trainer, pl_module)
