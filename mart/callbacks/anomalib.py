#
# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import types

import torch
from lightning.pytorch.callbacks import Callback

from torch_rotation import rotate_three_pass
from kornia.color import rgb_to_hsv, hsv_to_rgb
from kornia.geometry.transform import rotate
from tqdm import trange
from anomalib.metrics import create_metric_collection

from ..utils import get_pylogger

logger = get_pylogger(__name__)


__all__ = ["SemanticAdversary"]


class SemanticAdversary(Callback):
    """Perturbs inputs to be adversarial under semantic contraints."""

    def __init__(
        self,
        lr: float = 5.0,
        steps: int = 100,
        restarts: int = 5,
        angle_init: float = 0,
        angle_bound: float = 90.0,
        angle_lr_mult: float = 1,
        hue_init: float = 0,
        hue_bound: float = torch.pi,
        hue_lr_mult: float = 0.02,
        sat_init: float = 0,
        sat_bound: float = 0.5,
        sat_lr_mult: float = 0.02,
        metrics: list[str] = None,
    ):
        super().__init__()

        self.steps = steps
        self.restart_every = steps // (restarts or 1)

        self.angle_init = angle_init
        self.angle_bound = [-angle_bound, angle_bound]
        self.angle_lr = lr * angle_lr_mult

        self.hue_init = hue_init
        self.hue_bound = [-hue_bound, hue_bound]
        self.hue_lr = lr * hue_lr_mult

        self.sat_init = sat_init
        self.sat_bound = [-sat_bound, sat_bound]
        self.sat_lr = lr * sat_lr_mult

        self.metrics = metrics
        if self.metrics is None:
            self.metrics = ["F1Score", "AUROC"]

    def setup(self, trainer, pl_module, stage=None):
        self._on_after_batch_transfer = pl_module.on_after_batch_transfer
        pl_module.on_after_batch_transfer = types.MethodType(
            self.on_after_batch_transfer, pl_module
        )

    def teardown(self, trainer, pl_module, stage=None):
        pl_module.on_after_batch_transfer = self._on_after_batch_transfer

    @torch.inference_mode(False)
    def on_after_batch_transfer(self, pl_module, batch, dataloader_idx):
        batch = self._on_after_batch_transfer(batch, dataloader_idx)

        # FIXME: Dispatch to on_after_train/val/test_batch_transfer
        if not pl_module.trainer.testing:
            return batch

        device = pl_module.device

        # Create optimization variables for angle, hue, and saturation
        batch_size = batch["image"].shape[0]
        angle = torch.tensor(
            [self.angle_init] * batch_size,
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )

        hue = torch.tensor(
            [self.hue_init] * batch_size,
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )

        sat = torch.tensor(
            [self.sat_init] * batch_size,
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )

        # Metrics to save
        per_example_metric = torch.tensor(
            [torch.inf] * batch_size, device=device, dtype=torch.float32
        )

        # FIXME: F1Score seems like it needs an adaptive threshold!
        image_metrics = create_metric_collection(self.metrics, "image_").to(device)
        pixel_metrics = create_metric_collection(self.metrics, "pixel_").to(device)

        metrics = {
            "angle": angle.detach().clone(),
            "hue": hue.detach().clone(),
            "sat": sat.detach().clone(),
            "step": per_example_metric.clone(),
            "loss": per_example_metric.clone(),
        }

        for name in image_metrics.keys():
            metrics[name] = per_example_metric.clone()
        for name in pixel_metrics.keys():
            metrics[name] = per_example_metric.clone()

        # FIXME: It would be nice to extract these from the datamodule transform
        image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device)
        image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)

        # FIXME: Ideally we would clone the trainer and run fit instead of creating our own optimization loop
        # Run optimization
        for step in (pbar := trange(self.steps, desc="Attack", position=1)):
            if step % self.restart_every == 0:
                optimizer = torch.optim.Adam(
                    [
                        {"params": angle, "lr": self.angle_lr},
                        {"params": hue, "lr": self.hue_lr},
                        {"params": sat, "lr": self.sat_lr},
                    ],
                )

                # Randomly reinitialize parameters after restart
                if step > 0:
                    angle.data.uniform_(*self.angle_bound)
                    hue.data.uniform_(*self.hue_bound)
                    sat.data.uniform_(*self.sat_bound)

            # Clip parameters to valid bounds using straight-through estimator
            adv_batch = batch | {
                "angle": angle
                + (torch.clip(angle, *self.angle_bound) - angle).detach(),
                "hue": hue + (torch.clip(hue, *self.hue_bound) - hue).detach(),
                "sat": sat + (torch.clip(sat, *self.sat_bound) - sat).detach(),
            }

            # Perturb image and get outputs from model on perturbed image
            adv_image = perturb_image(
                **adv_batch, image_mean=image_mean, image_std=image_std
            )
            adv_batch = pl_module.validation_step(adv_batch | adv_image)
            del adv_image

            # Compute adversarial loss from model outputs and perturbed mask
            adv_mask = perturb_mask(**adv_batch)
            adv_batch = adv_batch | compute_loss(**(adv_batch | adv_mask))
            breakpoint()
            del adv_mask

            # Unrotate anomaly_maps for metric computations
            unrotated_anomaly_maps = perturb_mask(
                adv_batch["anomaly_maps"].detach(),
                angle=-adv_batch["angle"].detach(),
                mode="three_pass",
                padding_mode="constant",
            )
            adv_batch["anomaly_maps"] = unrotated_anomaly_maps["mask"]
            adv_batch["orig_anomaly_maps"] = unrotated_anomaly_maps["benign_mask"]
            del unrotated_anomaly_maps

            # Compute per-example and batch metrics
            adv_batch = adv_batch | compute_metrics(
                image_metrics,
                inputs=adv_batch["pred_scores"],
                targets=adv_batch["label"].long(),
            )
            adv_batch = adv_batch | compute_metrics(
                pixel_metrics,
                inputs=adv_batch["anomaly_maps"],
                targets=adv_batch["mask"].long(),
            )

            # Save metrics with lowest loss
            better = torch.where(adv_batch["loss"] < metrics["loss"])
            for key in metrics:
                # FIXME: remove this by comprehending scalars instead of key names
                if key == "step":
                    metrics[key][better] = step
                else:
                    metrics[key][better] = adv_batch[key][better].detach()

            pbar.set_postfix(
                {
                    "loss": adv_batch["batch_loss"].item(),
                    "iAUROC": adv_batch["batch_image_AUROC"].item(),
                    "pAUROC": adv_batch["batch_pixel_AUROC"].item(),
                    "↓loss": metrics["loss"].sum().item(),
                    "↓iAUROC": metrics["image_AUROC"].mean().item(),
                    "↓pAUROC": metrics["pixel_AUROC"].mean().item(),
                }
            )

            # Take optimization step
            optimizer.zero_grad()
            adv_batch["batch_loss"].backward()
            optimizer.step()

        print(f"{metrics = }")
        # NOTE: mask is now rotated and will be used to compute metrics!!!
        return adv_batch


# FIXME: Make this a @staticmethod
def perturb_image(
    image,
    *,
    angle=None,
    hue=None,
    sat=None,
    image_mean=None,
    image_std=None,
    mode="three_pass",  # bilinear | nearest | three_pass
    padding_mode="replicate",  # constant | replicate | reflect | circular | zeros | border | reflection
    **kwargs,
):
    image_adv = image.clone()

    # Return image to [0, 1] space
    if image_mean is not None and image_std is not None:
        image_adv = image_adv * image_std[..., None, None] + image_mean[..., None, None]

    # Rotate image by angle degrees
    if angle is not None:
        if mode == "three_pass":
            theta = torch.deg2rad(angle)
            image_adv = rotate_three_pass(
                image_adv, theta, N=-1, padding_mode=padding_mode
            )
        else:
            image_adv = rotate(image_adv, angle, mode=mode, padding_mode=padding_mode)

    # Convert image from RGB to HSV space
    if hue is not None or sat is not None:
        image_hsv = rgb_to_hsv(image_adv)
        image_hue, image_sat, image_val = torch.unbind(image_hsv, dim=-3)

    # Additive hue perturbation with STE clipping
    if hue is not None:
        image_hue = image_hue + hue[:, None, None]
        image_hue = torch.remainder(image_hue, 2 * torch.pi)

    # Additive saturation perturbation with STE clipping
    if sat is not None:
        image_sat = image_sat + sat[:, None, None]
        image_sat = image_sat + (torch.clip(image_sat, 0.0, 1.0) - image_sat).detach()

    # Convert image fro HSV to RGB space
    if hue is not None or sat is not None:
        image_hsv = torch.stack([image_hue, image_sat, image_val], dim=-3)
        image_adv = hsv_to_rgb(image_hsv)

    # Re-normalize image to image_mean and image_std
    if image_mean is not None and image_std is not None:
        image_adv = (image_adv - image_mean[..., None, None]) / image_std[
            ..., None, None
        ]

    return {"benign_image": image, "image": image_adv}


# FIXME: Make this a @staticmethod
@torch.no_grad()
def perturb_mask(
    mask,
    *,
    angle,
    mode="nearest",  # bilinear | nearest | three_pass
    padding_mode="zeros",  # constant | replicate | reflect | circular | zeros | border | reflection
    **kwargs,
):
    mask_rot = mask[..., None, :, :].clone()

    if mode == "three_pass":
        theta = torch.deg2rad(angle)
        mask_rot = rotate_three_pass(mask_rot, theta, N=-1, padding_mode=padding_mode)
    else:
        mask_rot = rotate(mask_rot, angle, mode=mode, padding_mode=padding_mode)

    mask_rot = mask_rot[..., 0, :, :]

    return {
        "benign_mask": mask,
        "mask": mask_rot,
    }


# FIXME: Make this a @staticmethod
def compute_loss(anomaly_maps, mask, **kwargs):
    true_negatives = mask == 0
    negative_loss = 1 - torch.sum(anomaly_maps * true_negatives, dim=(1, 2)) / (
        torch.sum(true_negatives, dim=(1, 2)) + 1e-8
    )

    true_positives = mask == 1
    positive_loss = torch.sum(anomaly_maps * true_positives, dim=(1, 2)) / (
        torch.sum(true_positives, dim=(1, 2)) + 1e-8
    )

    loss = negative_loss + positive_loss

    # decrease negatives and increase positives
    return {
        "negative_loss": negative_loss,
        "positive_loss": positive_loss,
        "loss": loss,
        "batch_loss": loss.sum(),
    }


# FIXME: Make this a @staticmethod
def compute_metrics(
    metric_collection,
    *,
    inputs,
    targets,
):
    ret = {}

    for name in metric_collection.keys():
        ret[name] = []

    for input, target in zip(inputs, targets):
        for name, value in metric_collection(input[None], target[None]).items():
            ret[name].append(value)

    for name in metric_collection.keys():
        ret[name] = torch.stack(ret[name])

    for name, value in metric_collection.compute().items():
        ret[f"batch_{name}"] = value

    return ret
