#
# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import os
import types
from copy import deepcopy

import torch
from lightning.pytorch.callbacks import Callback

from torch_rotation import rotate_three_pass
from kornia.color import rgb_to_hsv, hsv_to_rgb
from kornia.geometry.transform import rotate
from tqdm import trange
from anomalib.metrics import create_metric_collection
from collections import defaultdict


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
        seed: int = 2024,
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
            self.metrics = ["AUROC"]

        torch.manual_seed(seed)

    def on_test_epoch_start(self, trainer, pl_module):
        self.best = defaultdict(list)
        self.history = defaultdict(list)

    @torch.inference_mode(False)
    def on_test_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        device = pl_module.device

        # Create optimization variables for angle, hue, and saturation
        # FIXME: We should make this a proper nn.Module
        batch_size = batch["image"].shape[0]
        angle = torch.tensor(
            [self.angle_init] * batch_size,
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )
        hue = torch.full_like(angle, self.hue_init, requires_grad=True)
        sat = torch.full_like(angle, self.sat_init, requires_grad=True)

        # Metrics and parameters to save
        metrics = create_metric_collection(self.metrics, prefix="p")

        # FIXME: It would be nice to extract these from trainer.datamodule.test_data.transform
        image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device)
        image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)

        # FIXME: Ideally we would clone the trainer and run fit instead of creating our own optimization loop
        # Run optimization
        best_batch = None
        batch_history = defaultdict(list)

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
                "step": torch.full_like(angle, step),
            }

            # Perturb image
            adv_batch |= perturb_image(
                **adv_batch, image_mean=image_mean, image_std=image_std
            )

            # Run perturbed image through module. Note the module in-place
            # modifies the batch
            adv_batch |= pl_module.test_step(
                adv_batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx
            )

            # Compute adversarial loss from model outputs and perturbed mask
            adv_batch |= perturb_mask(**adv_batch)
            adv_batch |= compute_loss(**adv_batch)

            # Take optimization step
            optimizer.zero_grad()
            adv_batch["loss"].sum().backward()
            optimizer.step()

            # Compute per-example and batch pixel metrics
            adv_batch = adv_batch | compute_metrics(
                metrics,
                inputs=adv_batch["anomaly_maps"],
                targets=adv_batch["mask"].int(),
            )

            # Detach all tensors in batch
            adv_batch = {
                key: value.detach() if isinstance(value, torch.Tensor) else value
                for key, value in adv_batch.items()
            }

            # Save some items in batch to history
            for key, value in adv_batch.items():
                # Ignore 2D+ tensors (too much memory), python lists, and some specific keys
                if (
                    key in ["label", "step"]
                    or (isinstance(value, torch.Tensor) and value.ndim > 1)
                    or (isinstance(value, list))
                ):
                    continue

                batch_history[key].append(
                    value.to("cpu", non_blocking=True)
                    if isinstance(value, torch.Tensor)
                    else value
                )

            # Save initial batch since it is the best batch
            if best_batch is None:
                best_batch = deepcopy(adv_batch)

            # Find examples with lowest loss, if all negative mask, or lowest metric (pAUROC)
            is_negative = torch.sum(adv_batch["mask"], dim=(-2, -1)) == 0
            loss_is_lower = adv_batch["loss"] < best_batch["loss"]
            metric_is_lower = adv_batch["pAUROC"] < best_batch["pAUROC"]
            is_lower = torch.where(is_negative, loss_is_lower, metric_is_lower)
            lower = torch.where(is_lower)

            # Save best examples from batch
            for key in best_batch:
                value = adv_batch[key]
                if (
                    isinstance(value, torch.Tensor)
                    and value.ndim > 0
                    and value.shape[0] == len(is_lower)
                ):
                    best_batch[key][lower] = value[lower]
                elif isinstance(value, list):
                    # FIXME: Use numpy instead?
                    for i in lower[0]:
                        best_batch[key][i] = value[i]
                else:
                    best_batch[key] = value

            # Update metrics using best examples and save them to history
            best_metrics = compute_metrics(
                metrics,
                inputs=best_batch["anomaly_maps"],
                targets=best_batch["mask"].int(),
            )
            for key, value in best_metrics.items():
                batch_history[f"best_{key}"].append(
                    value.to("cpu", non_blocking=True)
                    if isinstance(value, torch.Tensor)
                    else value
                )
            best_batch = best_batch | best_metrics

            # Update progress bar with metrics
            pbar.set_postfix(
                {
                    "loss": f"{adv_batch['loss'].sum().item():.6g}",
                    "↓loss": f"{best_batch['loss'].sum().item():.6g}",
                    "pAUROC": f"{adv_batch['batch_pAUROC'].item():.4g}",
                    "↓pAUROC": f"{best_batch['batch_pAUROC'].item():.4g}",
                }
            )

        pbar.close()

        # Save best batch and history
        for key, value in best_batch.items():
            self.best[key].append(value)
        for key, value in batch_history.items():
            if isinstance(value[0], torch.Tensor):
                value = torch.stack(value, dim=-1)
            self.history[key].append(value)

        # Update batch with items from best batch
        for key in batch:
            value = best_batch[key]
            if isinstance(value, torch.Tensor):
                value = value.to(device)
            batch[key] = value

    def on_test_epoch_end(self, trainer, pl_module):
        results = {
            "best": flatten(self.best),
            "history": flatten(self.history),
        }

        torch.save(results, os.path.join(trainer.default_root_dir, "results.pt"))
        del self.best, self.history


def flatten(dict_of_lists):
    # Flatten list of tensors/lists into tensor/list
    flattened = {}

    for key, value in dict_of_lists.items():
        if isinstance(value[0], torch.Tensor) and value[0].ndim > 0:
            value = torch.concat(value)
        elif isinstance(value[0], torch.Tensor):
            value = torch.stack(value)
        elif isinstance(value[0], list):
            value = sum(value, [])
        flattened[key] = value

    return flattened


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

    # Additive hue perturbation with remainder
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
    mask_rot = mask.clone()
    mask_rot = mask_rot[..., None, :, :].float()

    if mode == "three_pass":
        theta = torch.deg2rad(angle)
        mask_rot = rotate_three_pass(mask_rot, theta, N=-1, padding_mode=padding_mode)
    else:
        mask_rot = rotate(mask_rot, angle, mode=mode, padding_mode=padding_mode)

    mask_rot = mask_rot[..., 0, :, :].int()

    return {
        "benign_mask": mask,
        "mask": mask_rot,
    }


# FIXME: Make this a @staticmethod
def compute_loss(anomaly_maps, mask, **kwargs):
    true_negatives = mask == 0
    negative_loss = -torch.sum(anomaly_maps * true_negatives, dim=(1, 2)) / (
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

    metric_collection.reset()
    for input, target in zip(inputs, targets):
        for name, value in metric_collection(input[None], target[None]).items():
            ret[name].append(value)

    for name in metric_collection.keys():
        ret[name] = torch.stack(ret[name])

    for name, value in metric_collection.compute().items():
        ret[f"batch_{name}"] = value[None]

    return ret


if __name__ == "__main__":
    image = torch.rand((16, 3, 240, 240))
    zeros = torch.zeros_like(image[:, 0, 0, 0])

    adv_image = perturb_image(image, angle=zeros, hue=zeros, sat=zeros)
    assert torch.allclose(adv_image["image"], adv_image["benign_image"], atol=0.1 / 255)

    image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
    image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])

    image = (image - image_mean[..., None, None]) / image_std[..., None, None]

    adv_image = perturb_image(
        image,
        angle=zeros,
        hue=zeros,
        sat=zeros,
        image_mean=image_mean,
        image_std=image_std,
    )
    assert torch.allclose(adv_image["image"], adv_image["benign_image"], atol=0.1 / 255)
