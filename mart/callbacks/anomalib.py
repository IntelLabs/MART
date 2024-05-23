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
        steps = torch.tensor(
            [0] * batch_size,
            device=device,
            dtype=torch.int32,
        )

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
        metrics = {
            "angle": angle.detach().clone(),
            "hue": hue.detach().clone(),
            "sat": sat.detach().clone(),
            "gain": torch.tensor(
                [-torch.inf] * batch_size,
                device=device,
                dtype=torch.float32,
            ),
            "step": torch.tensor(
                [0] * batch_size,
                device=device,
                dtype=torch.int32,
            ),
        }

        # Run optimization
        for step in (pbar := trange(self.steps, desc="Attack", position=1)):
            if step % self.restart_every == 0:
                optimizer = torch.optim.Adam(
                    [
                        {"params": angle, "lr": self.angle_lr},
                        {"params": hue, "lr": self.hue_lr},
                        {"params": sat, "lr": self.sat_lr},
                    ],
                    maximize=True,
                )

                # Randomly reinitialize parameters after restart
                if step > 0:
                    angle.data.uniform_(*self.angle_bound)
                    hue.data.uniform_(*self.hue_bound)
                    sat.data.uniform_(*self.sat_bound)

            # Clip parameters to valid bounds using straight-through estimator
            batch["angle"] = (
                angle + (torch.clip(angle, *self.angle_bound) - angle).detach()
            )
            batch["hue"] = hue + (torch.clip(hue, *self.hue_bound) - hue).detach()
            batch["sat"] = sat + (torch.clip(sat, *self.sat_bound) - sat).detach()
            batch["step"] = steps + step

            adv_batch = perturb(**batch)
            adv_batch = pl_module.validation_step(adv_batch)
            adv_batch["gain"] = compute_gain(**adv_batch)

            # Save examples with highest gain
            better = torch.where(adv_batch["gain"] > metrics["gain"])
            for key in metrics:
                metrics[key][better] = adv_batch[key][better].detach()

            optimizer.zero_grad()
            adv_batch["gain"].sum().backward()
            optimizer.step()

        # NOTE: mask is now rotated and will be used to compute metrics!!!
        return adv_batch


def perturb(*, image, mask, angle=None, hue=None, sat=None, **kwargs):
    # Return image to [0, 1] space
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=image.device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=image.device)
    image_adv = image * std[..., None, None] + mean[..., None, None]

    # Rotate image by angle degrees
    if angle is not None:
        theta = torch.deg2rad(angle)
        image_adv = rotate_three_pass(image_adv, theta, N=-1, padding_mode="replicate")

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

    if hue is not None or sat is not None:
        image_hsv = torch.stack([image_hue, image_sat, image_val], dim=-3)
        image_adv = hsv_to_rgb(image_hsv)

    # Re-normalize image
    image_adv = (image_adv - mean[..., None, None]) / std[..., None, None]

    # Rotate mask to match image
    mask_rot = mask[..., None, :, :]
    if angle is not None:
        mask_rot = rotate(mask_rot, angle.detach(), mode="nearest")
    mask_rot = mask_rot[..., 0, :, :]

    # Replace image with adversarial image and mask with rotated mask
    return {
        "image": image_adv,
        "mask": mask_rot,
        "angle": angle,
        "hue": hue,
        "sat": sat,
        **kwargs,
    }


def compute_gain(anomaly_maps, mask, **kwargs):
    true_negatives = mask == 0
    negative_loss = torch.sum(anomaly_maps * true_negatives, dim=(1, 2)) / (
        torch.sum(true_negatives, dim=(1, 2)) + 1e-8
    )

    true_positives = mask == 1
    positive_loss = torch.sum(anomaly_maps * true_positives, dim=(1, 2)) / (
        torch.sum(true_positives, dim=(1, 2)) + 1e-8
    )

    # decrease negatives and increase positives
    return negative_loss + -positive_loss
