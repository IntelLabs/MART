#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from ..utils import MonkeyPatch


class LightningModuleAsTarget:
    """Prepare a LightningModule as a target model for Adversary,
    such that `output = model(batch)`.
    """

    def __call__(self, model):
        # Generate a pseudo dataloader_idx.
        dataloader_idx = 1

        if hasattr(model, "attack_step"):

            def model_forward(batch):
                output = model.attack_step(batch, dataloader_idx)
                return output

        elif hasattr(model, "training_step"):
            # Monkey-patch model.log to avoid spamming.
            def model_forward(batch):
                with MonkeyPatch(model, "log", lambda *args, **kwargs: None):
                    output = model.training_step(batch, dataloader_idx)
                return output

        else:
            raise ValueError("Model does not have `attack_step()` or `training_step()`.")

        return model_forward


# TODO: We may need to do model.eval() if there's BN-like layers in the model.
