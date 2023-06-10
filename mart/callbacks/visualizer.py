#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from operator import attrgetter

from pytorch_lightning.callbacks import Callback

__all__ = ["ImageVisualizer"]


class ImageVisualizer(Callback):
    def __init__(self, frequency: int = 100, **tag_keys):
        self.frequency = frequency
        self.tag_keys = tag_keys

    def log_image(self, trainer, tag, image):
        # Add image to each logger
        for logger in trainer.loggers:
            # FIXME: Should we just use isinstance(logger.experiment, SummaryWriter)?
            if not hasattr(logger.experiment, "add_image"):
                continue

            if len(image.shape) == 4:
                logger.experiment.add_images(tag, image, global_step=trainer.global_step)
            elif len(image.shape) == 3:
                logger.experiment.add_image(tag, image, global_step=trainer.global_step)
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.frequency != 0:
            return

        for tag, output_key in self.tag_keys.items():
            image = outputs[output_key]
            self.log_image(trainer, tag, image)
