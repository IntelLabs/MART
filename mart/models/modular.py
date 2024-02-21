#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import logging
from operator import attrgetter

import torch
from lightning.pytorch import LightningModule

from ..nn import SequentialDict
from ..optim import OptimizerFactory
from ..utils import flatten_dict
from ..utils.optimization import configure_optimizers

logger = logging.getLogger(__name__)

__all__ = ["LitModular"]


class LitModular(LightningModule):
    def __init__(
        self,
        modules,
        optimizer,
        lr_scheduler=None,
        training_sequence=None,
        training_step_log=None,
        training_metrics=None,
        validation_sequence=None,
        validation_step_log=None,
        validation_metrics=None,
        test_sequence=None,
        test_step_log=None,
        test_metrics=None,
        predict_sequence=None,
        predict_step_log=None,
        load_state_dict=None,
        output_loss_key="loss",
        output_preds_key="preds",
        output_target_key="target",
    ):
        super().__init__()

        # FIXME: Why not just make these required arguments? Update: these are allowed to be None
        #        in which case they default to the order of the modules. I think this is bad behavior.
        # assert training_sequence is not None
        # assert validation_sequence is not None
        # assert test_sequence is not None

        # Convert dict sequences to list sequences by sorting keys
        if isinstance(training_sequence, dict):
            training_sequence = [training_sequence[key] for key in sorted(training_sequence)]
        if isinstance(validation_sequence, dict):
            validation_sequence = [validation_sequence[key] for key in sorted(validation_sequence)]
        if isinstance(test_sequence, dict):
            test_sequence = [test_sequence[key] for key in sorted(test_sequence)]
        if isinstance(predict_sequence, dict):
            predict_sequence = [predict_sequence[key] for key in sorted(predict_sequence)]

        # *_step() functions make some assumptions about the type of Module it can call.
        # That is, injecting a nn.Module generally won't work, so better to hardcode SequentialDict.
        # It also gets rid of an indentation level in the configs.
        sequences = {
            "training": training_sequence,
            "validation": validation_sequence,
            "test": test_sequence,
            "predict": predict_sequence,
        }
        self.model = SequentialDict(modules, sequences)

        self.optimizer_fn = optimizer
        if not isinstance(self.optimizer_fn, OptimizerFactory):
            # Set bias_decay and norm_decay to 0.
            self.optimizer_fn = OptimizerFactory(self.optimizer_fn)

        self.lr_scheduler = lr_scheduler

        # Be backwards compatible by turning list into dict where each item is its own key-value
        if isinstance(training_step_log, (list, tuple)):
            training_step_log = {item: item for item in training_step_log}
        self.training_step_log = training_step_log or {}
        self.training_metrics = training_metrics

        # Be backwards compatible by turning list into dict where each item is its own key-value
        if isinstance(validation_step_log, (list, tuple)):
            validation_step_log = {item: item for item in validation_step_log}
        self.validation_step_log = validation_step_log or {}
        self.validation_metrics = validation_metrics

        # Be backwards compatible by turning list into dict where each item is its own key-value
        if isinstance(test_step_log, (list, tuple)):
            test_step_log = {item: item for item in test_step_log}
        self.test_step_log = test_step_log or {}
        self.test_metrics = test_metrics

        # Be backwards compatible by turning list into dict where each item is its own key-value
        if isinstance(predict_step_log, (list, tuple)):
            predict_step_log = {item: item for item in predict_step_log}
        self.predict_step_log = predict_step_log or {}

        # Load state dict for specified modules. We flatten it because Hydra
        # commandlines converts dotted paths to nested dictionaries.
        if isinstance(load_state_dict, str):
            load_state_dict = {None: load_state_dict}
        load_state_dict = flatten_dict(load_state_dict or {})

        for name, path in load_state_dict.items():
            module = self.model
            if name is not None:
                module = attrgetter(name)(module)
            logger.info(f"Loading state_dict {path} for {module.__class__.__name__}...")
            module.load_state_dict(torch.load(path, map_location="cpu"))

        self.output_loss_key = output_loss_key
        self.output_preds_key = output_preds_key
        self.output_target_key = output_target_key

    def configure_optimizers(self):

        return configure_optimizers(self.model, self.optimizer_fn, self.lr_scheduler)

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def attack_step(self, batch, batch_idx):
        # Use the training sequence in attack.
        input, target = batch
        output = self(input=input, target=target, model=self.model, step="training")
        return output

    #
    # Training
    #
    def training_step(self, batch, batch_idx):
        # FIXME: Would be much nicer if batch was a dict!
        input, target = batch
        output = self(input=input, target=target, model=self.model, step="training")

        for log_name, output_key in self.training_step_log.items():
            self.log(f"training/{log_name}", output[output_key])

        if self.training_metrics is not None:
            # Some models only return loss in the training mode.
            if self.output_preds_key not in output or self.output_target_key not in output:
                raise ValueError(
                    f"You have specified training_metrics, but the model does not return {self.output_preds_key} or {self.output_target_key} during training. You can either nullify training_metrics or configure the model to return {self.output_preds_key} and {self.output_target_key} in the training output."
                )
            self.training_metrics(output[self.output_preds_key], output[self.output_target_key])

        loss = output[self.output_loss_key]
        # We need to manually log loss on the progress bar in newer PL.
        self.log("loss", loss, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        if self.training_metrics is not None:
            # Some models only return loss in the training mode.
            metrics = self.training_metrics.compute()
            metrics = self.flatten_metrics(metrics)
            self.training_metrics.reset()

            self.log_metrics(metrics, prefix="training_metrics")

    #
    # Validation
    #
    def validation_step(self, batch, batch_idx):
        # FIXME: Would be much nicer if batch was a dict!
        input, target = batch
        output = self(input=input, target=target, model=self.model, step="validation")

        for log_name, output_key in self.validation_step_log.items():
            self.log(f"validation/{log_name}", output[output_key])

        self.validation_metrics(output[self.output_preds_key], output[self.output_target_key])

        return None

    def on_validation_epoch_end(self):
        metrics = self.validation_metrics.compute()
        metrics = self.flatten_metrics(metrics)
        self.validation_metrics.reset()

        self.log_metrics(metrics, prefix="validation_metrics")

    #
    # Test
    #
    def test_step(self, batch, batch_idx):
        # FIXME: Would be much nicer if batch was a dict!
        input, target = batch
        output = self(input=input, target=target, model=self.model, step="test")

        for log_name, output_key in self.test_step_log.items():
            self.log(f"test/{log_name}", output[output_key])

        self.test_metrics(output[self.output_preds_key], output[self.output_target_key])

        return None

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        metrics = self.flatten_metrics(metrics)
        self.test_metrics.reset()

        self.log_metrics(metrics, prefix="test_metrics")

    #
    # Predict
    #
    def predict_step(self, batch, batch_idx):
        input, target = batch
        pred = self(input=input, target=target, model=self.model, step="predict")

        for log_name, output_key in self.predict_step_log.items():
            self.log(f"predict/{log_name}", pred[output_key])

        return pred

    #
    # Utilities
    #
    def flatten_metrics(self, metrics):
        # torchmetrics==0.6.0 does not flatten group metrics such as mAP (which includes mAP and mAP-50, etc),
        # while later versions do. We add this for forward compatibility while we downgrade to 0.6.0.
        flat_metrics = {}

        for k, v in metrics.items():
            if isinstance(v, dict):
                # recursively flatten metrics
                v = self.flatten_metrics(v)
                for k2, v2 in v.items():
                    if k2 in flat_metrics:
                        logger.warn(f"{k}/{k2} overrides existing metric!")

                    flat_metrics[k2] = v2
            else:
                # assume raw metric
                if k in flat_metrics:
                    logger.warn(f"{k} overrides existing metric!")

                flat_metrics[k] = v

        return flat_metrics

    def log_metrics(self, metrics, prefix="", prog_bar=False):
        metrics_dict = {}

        def enumerate_metric(metric, name):
            # Metrics can have arbitrary depth.
            if isinstance(metric, torch.Tensor):
                # Ignore non-scalar results generated by Metrics, such as list of classes from MAP.
                if metric.shape == torch.Size([]):
                    metrics_dict[name] = metric
            else:
                for k, v in metric.items():
                    enumerate_metric(v, f"{name}/{k}")

        enumerate_metric(metrics, prefix)

        # sync_dist is not necessary for torchmetrics: https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html
        self.log_dict(metrics_dict, prog_bar=prog_bar)
