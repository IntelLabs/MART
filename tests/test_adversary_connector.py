#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from itertools import cycle
from unittest.mock import Mock

from lightning.pytorch import LightningModule, Trainer

from mart.callbacks import AdversaryConnector
from mart.transforms import TupleBatchC15n


def test_adversary_connector_callback(input_data, target_data, perturbation):
    batch = (input_data, target_data)
    input_adv = input_data + perturbation

    def adversary_fit(input, target, *, model):
        model(input, target)

    adversary = Mock(return_value=(input_adv, target_data), fit=adversary_fit)
    batch_c15n = TupleBatchC15n()
    callback = AdversaryConnector(test_adversary=adversary, batch_c15n=batch_c15n)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        limit_test_batches=1,
        callbacks=callback,
        num_sanity_val_steps=0,
        logger=[],
        enable_model_summary=False,
        enable_checkpointing=False,
    )

    # Call attack_step()
    # `model` must be a `LightningModule`
    model_attack = LightningModule()
    # Trick PL that test_step is overridden.
    model_attack.test_step = Mock(wraps=lambda *args: None)
    model_attack.attack_step = Mock(wraps=lambda *args: None)
    trainer.test(model_attack, dataloaders=cycle([batch]))
    model_attack.attack_step.assert_called_once()

    # Call training_step()
    model_training = LightningModule()
    model_training.test_step = Mock(wraps=lambda *args: None)
    model_training.training_step = Mock(wraps=lambda *args: None)
    trainer.test(model_training, dataloaders=cycle([batch]))
    model_training.training_step.assert_called_once()
