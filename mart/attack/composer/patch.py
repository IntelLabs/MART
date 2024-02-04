#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import torch
import torchvision.transforms.functional as F

__all__ = [
    "PertRectSize",
    "PertExtractRect",
    "PertRectPerspective",
]


class PertRectSize(torch.nn.Module):
    """Calculate the size of the smallest rectangle that can be transformed with the highest pixel
    fidelity."""

    @staticmethod
    def get_smallest_rect(coords):
        # Calculate the distance between two points.
        coords_shifted = torch.cat([coords[1:], coords[0:1]])
        w1, h2, w2, h1 = torch.sqrt(
            torch.sum(torch.pow(torch.subtract(coords, coords_shifted), 2), dim=1)
        )

        height = int(max(h1, h2).round())
        width = int(max(w1, w2).round())
        return height, width

    def forward(self, coords):
        height, width = self.get_smallest_rect(coords)
        return {"height": height, "width": width}


class PertExtractRect(torch.nn.Module):
    """Extract a small rectangular patch from the input size one."""

    def forward(self, perturbation, height, width):
        perturbation = perturbation[:, :height, :width]
        return perturbation


class PertRectPerspective(torch.nn.Module):
    """Pad perturbation to input size, then perspective transform the top-left rectangle."""

    def forward(self, perturbation, input, coords):
        # Pad to the input size.
        height_inp, width_inp = input.shape[-2:]
        height_pert, width_pert = perturbation.shape[-2:]
        height_pad = height_inp - height_pert
        width_pad = width_inp - width_pert
        perturbation = F.pad(perturbation, padding=[0, 0, width_pad, height_pad])

        # F.perspective() requires startpoints and endpoints in CPU.
        startpoints = torch.tensor(
            [[0, 0], [width_pert, 0], [width_pert, height_pert], [0, height_pert]]
        )
        endpoints = coords.cpu()

        perturbation = F.perspective(
            img=perturbation,
            startpoints=startpoints,
            endpoints=endpoints,
            interpolation=F.InterpolationMode.BILINEAR,
            fill=0,
        )

        return perturbation
