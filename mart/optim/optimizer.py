#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

import torch  # noqa: E402

__all__ = ["OptimizerFactory"]


class OptimizerFactory:
    """Create optimizers compatible with LightningModule.

    Also supports decay parameters for bias and norm modules independently.
    """

    def __init__(self, optimizer, **kwargs):
        weight_decay = kwargs.get("weight_decay", 0.0)

        self.bias_decay = kwargs.pop("bias_decay", weight_decay)
        self.norm_decay = kwargs.pop("norm_decay", weight_decay)
        self.optimizer = optimizer
        self.kwargs = kwargs

    def __call__(self, module):
        # Separate parameters into biases, norms, and weights
        bias_params = []
        norm_params = []
        weight_params = []
        modality_params = defaultdict(list)

        for param_name, param in module.named_parameters():
            if not param.requires_grad:
                continue

            # Find module by name
            module_name = ".".join(param_name.split(".")[:-1])
            _, param_module = next(filter(lambda nm: nm[0] == module_name, module.named_modules()))
            module_kind = param_module.__class__.__name__

            if module_kind == "ModalityParameterDict":
                # Identify modality-aware parameters for adversary.
                modality = param_name.split(".")[-1]
                modality_params[modality].append(param)
            elif "Norm" in module_kind:
                assert len(param.shape) == 1
                norm_params.append(param)
            elif isinstance(param, torch.nn.UninitializedParameter):
                # Assume lazy parameters are weights
                weight_params.append(param)
            elif len(param.shape) == 1:
                bias_params.append(param)
            else:  # Assume weights
                weight_params.append(param)

        params = []

        # Set modality-aware params.
        if len(modality_params) > 0:
            for modality, param in modality_params.items():
                # Take notes of modality for gradient modifier later.
                # Add modality-specific optim params.
                params.append(
                    {"params": param, "modality": modality} | self.kwargs.pop(modality, {})
                )

        # Set decay for bias and norm parameters
        if len(weight_params) > 0:
            params.append({"params": weight_params})  # use default weight decay
        if len(bias_params) > 0:
            params.append({"params": bias_params, "weight_decay": self.bias_decay})
        if len(norm_params) > 0:
            params.append({"params": norm_params, "weight_decay": self.norm_decay})

        return self.optimizer(params, **self.kwargs)
