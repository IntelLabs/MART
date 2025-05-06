#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import logging
from typing import Sequence

import torch
from lightning.pytorch.callbacks import Callback
from torchmetrics import Metric

from ..nn.nn import DotDict

logger = logging.getLogger(__name__)


class LogMetrics(Callback):
    """For models returning a dictionary, we can configure the callback to log scalars from the
    outputs, calculate and log metrics."""

    def __init__(
        self,
        train_step_log: Sequence | dict = None,
        val_step_log: Sequence | dict = None,
        test_step_log: Sequence | dict = None,
        train_metrics: Metric = None,
        val_metrics: Metric = None,
        test_metrics: Metric = None,
        output_preds_key: str = "preds",
        output_target_key: str = "target",
        # We may display only some of the metrics on the progress bar, if there are too many.
        metrics_on_train_prog_bar: bool | Sequence[str] = True,
        metrics_on_val_prog_bar: bool | Sequence[str] = True,
        metrics_on_test_prog_bar: bool | Sequence[str] = True,
    ):
        super().__init__()

        # Be backwards compatible by turning list into dict where each item is its own key-value
        if isinstance(train_step_log, (list, tuple)):
            train_step_log = {item: {"key": item, "prog_bar": True} for item in train_step_log}
        train_step_log = train_step_log or {}

        # Be backwards compatible by turning list into dict where each item is its own key-value
        if isinstance(val_step_log, (list, tuple)):
            val_step_log = {item: {"key": item, "prog_bar": True} for item in val_step_log}
        val_step_log = val_step_log or {}

        # Be backwards compatible by turning list into dict where each item is its own key-value
        if isinstance(test_step_log, (list, tuple)):
            test_step_log = {item: {"key": item, "prog_bar": True} for item in test_step_log}
        test_step_log = test_step_log or {}

        self.step_log = {
            "train": train_step_log,
            "val": val_step_log,
            "test": test_step_log,
        }
        self.metrics = {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        }
        self.metrics_on_prog_bar = {
            "train": metrics_on_train_prog_bar,
            "val": metrics_on_val_prog_bar,
            "test": metrics_on_test_prog_bar,
        }

        self.output_preds_key = output_preds_key
        self.output_target_key = output_target_key

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        return self.on_batch_end(outputs, prefix="train")

    def on_train_epoch_end(self, trainer, pl_module):
        return self.on_epoch_end(pl_module, prefix="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        return self.on_batch_end(outputs, prefix="val")

    def on_validation_epoch_end(self, trainer, pl_module):
        return self.on_epoch_end(pl_module, prefix="val")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        return self.on_batch_end(outputs, prefix="test")

    def on_test_epoch_end(self, trainer, pl_module):
        return self.on_epoch_end(pl_module, prefix="test")

    #
    # Utilities
    #
    def on_batch_end(self, outputs, *, prefix: str):
        # Convert to DotDict, so that we can use a dot-connected string as a key to find a value deep in the dictionary.
        outputs = DotDict(outputs)

        step_log = self.step_log[prefix]
        for log_name, cfg in step_log.items():
            key, prog_bar = cfg["key"], cfg["prog_bar"]
            self.log(f"{prefix}/{log_name}", outputs[key], prog_bar=prog_bar)

        metric = self.metrics[prefix]
        if metric is not None:
            metric(outputs[self.output_preds_key], outputs[self.output_target_key])

    def on_epoch_end(self, pl_module, *, prefix: str):
        metric = self.metrics[prefix]
        if metric is not None:
            # Some models only return loss in the train mode.
            results = metric.compute()
            results = self.flatten_metrics(results)
            metric.reset()

            self.log_metrics(pl_module, results, prefix=prefix)

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
                        logger.warning(f"{k}/{k2} overrides existing metric!")

                    flat_metrics[k2] = v2
            else:
                # assume raw metric
                if k in flat_metrics:
                    logger.warning(f"{k} overrides existing metric!")

                flat_metrics[k] = v

        return flat_metrics

    def log_metrics(self, pl_module, metrics, prefix=""):
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
        on_prog_bar = self.metrics_on_prog_bar[prefix]
        if isinstance(on_prog_bar, bool):
            pl_module.log_dict(metrics_dict, prog_bar=on_prog_bar)
        elif isinstance(on_prog_bar, Sequence):
            for metric_key in on_prog_bar:
                metric_value = metrics_dict.pop(metric_key)
                pl_module.log(f"{prefix}/{metric_key}", metric_value, prog_bar=on_prog_bar)
        else:
            raise ValueError(f"Unknown type: {type(self.metrics_on_prog_bar[prefix])=}")
