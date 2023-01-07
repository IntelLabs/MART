#
# Copyright (C) 2022 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

import logging

logger = logging.getLogger(__name__)

import torch  # noqa: E402
from pytorch_lightning import LightningModule  # noqa: E402

from ..nn import SequentialDict  # noqa: E402

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
        loss_sequence=None,
        prediction_sequence=None,
        weights_fpath=None,
        strict=True,
    ):
        super().__init__()

        # *_step() functions make some assumptions about the type of Module it can call.
        # That is, injecting a nn.Module generally won't work, so better to hardcode ModuleDict.
        # It also gets rid of an indentation level in the configs.
        sequences = {
            "training": training_sequence,
            "validation": validation_sequence,
            "test": test_sequence,
            "loss": loss_sequence,
            "prediction": prediction_sequence,
        }
        self.model = SequentialDict(modules, sequences)

        if weights_fpath is not None:
            self.model.load_state_dict(torch.load(weights_fpath), strict=strict)

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.training_step_log = training_step_log or ["loss"]
        self.training_metrics = training_metrics

        self.validation_step_log = validation_step_log or []
        self.validation_metrics = validation_metrics

        self.test_step_log = test_step_log or []
        self.test_metrics = test_metrics

    def configure_optimizers(self):
        config = {}
        config["optimizer"] = self.optimizer(self.model)

        if self.lr_scheduler is not None:
            # FIXME: I don't think this actually work correctly, but we don't have an example of an lr_scheduler that is not a DictConfig
            if "scheduler" in self.lr_scheduler:
                config["lr_scheduler"] = dict(self.lr_scheduler)
                config["lr_scheduler"]["scheduler"] = config["lr_scheduler"]["scheduler"](
                    config["optimizer"]
                )
            else:
                config["lr_scheduler"] = self.lr_scheduler(config["optimizer"])

        return config

    def forward(self, **kwargs):
        return self.model(**kwargs)

    #
    # Training
    #
    def training_step(self, batch, batch_idx):
        # FIXME: Would be much nicer if batch was a dict!
        input, target = batch
        output = self(input=input, target=target, model=self.model, step="training")

        for name in self.training_step_log:
            self.log(f"training/{name}", output[name])

        assert "loss" in output
        return output

    def training_step_end(self, output):
        if self.training_metrics is not None:
            # Some models only return loss in the training mode.
            if "preds" not in output or "target" not in output:
                raise ValueError(
                    "You have specified training_metrics, but the model does not return preds and target during training. You can either nullify training_metrics or configure the model to return preds and target in the training output."
                )
            self.training_metrics(output["preds"], output["target"])
        loss = output.pop("loss")
        return loss

    def training_epoch_end(self, outputs):
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

        for name in self.validation_step_log:
            self.log(f"validation/{name}", output[name])

        return output

    def validation_step_end(self, output):
        self.validation_metrics(output["preds"], output["target"])

        # I don't know why this is required to prevent CUDA memory leak in validaiton and test. (Not required in training.)
        output.clear()

    def validation_epoch_end(self, outputs):
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

        for name in self.test_step_log:
            self.log(f"test/{name}", output[name])

        return output

    def test_step_end(self, output):
        self.test_metrics(output["preds"], output["target"])

        # I don't know why this is required to prevent CUDA memory leak in validaiton and test. (Not required in training.)
        output.clear()

    def test_epoch_end(self, outputs):
        metrics = self.test_metrics.compute()
        metrics = self.flatten_metrics(metrics)
        self.test_metrics.reset()

        self.log_metrics(metrics, prefix="test_metrics")

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
                metrics_dict[name] = metric
            else:
                for k, v in metric.items():
                    enumerate_metric(v, f"{name}/{k}")

        enumerate_metric(metrics, prefix)

        self.log_dict(metrics_dict, prog_bar=prog_bar)
