#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import logging
from typing import List

from lightning.pytorch.callbacks import BasePredictionWriter, Callback

from ..datamodules import FiftyOneDataset

logger = logging.getLogger(__name__)
try:
    import fiftyone as fo
except ImportError:
    logger.debug("fiftyone module is not installed!")

__all__ = ["FiftyOneEvaluateDetections", "FiftyOnePredictionAdder"]


class FiftyOneEvaluateDetections(Callback):
    def __init__(self, run_id: str, gt_field: str = "ground_truth_detections") -> None:
        self.run_id = run_id
        self.gt_field = gt_field

    def on_predict_end(self, trainer, pl_module):
        predict_dataset = trainer.datamodule.predict_dataset
        assert isinstance(predict_dataset, FiftyOneDataset)

        eval_key = f"eval_{self.run_id}".replace("-", "")
        eval_key = eval_key.replace("_", "")
        results = predict_dataset.filtered_dataset.evaluate_detections(
            f"prediction_{self.run_id}",
            gt_field=self.gt_field,
            eval_key=eval_key,
            compute_mAP=True,
        )

        logger.info(f"Prediction mAP={results.mAP()}")

        # Get the 10 most common classes in the dataset
        counts = predict_dataset.filtered_dataset.count_values(f"{self.gt_field}.detections.label")
        classes_top10 = sorted(counts, key=counts.get, reverse=True)[:10]

        # Print a classification report for the top-10 classes
        results.print_report(classes=classes_top10)


class FiftyOnePredictionAdder(BasePredictionWriter):
    def __init__(self, output_dir: str, write_interval: List[str]) -> None:
        super().__init__(write_interval)
        self.run_id = f"prediction_{output_dir}"

    def _write_predictions(self, predictions, groundtruth_preds, dataset):
        for pred, gt_pred in zip(predictions, groundtruth_preds):
            filename = gt_pred["file_name"]
            dataset.add_predictions(filename, pred, self.run_id)

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        predict_dataset = trainer.datamodule.predict_dataset
        assert isinstance(predict_dataset, FiftyOneDataset)

        self._write_predictions(
            prediction[pl_module.output_preds_key],
            prediction[pl_module.output_target_key],
            predict_dataset,
        )

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        predict_dataset = trainer.datamodule.predict_dataset
        assert isinstance(predict_dataset, FiftyOneDataset)

        for output in predictions:
            self._write_predictions(
                output[pl_module.output_preds_key],
                output[pl_module.output_target_key],
                predict_dataset,
            )
