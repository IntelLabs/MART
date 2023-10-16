#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#


def configure_optimizers(module, optimizer, lr_scheduler):
    config = {}
    config["optimizer"] = optimizer(module)

    if lr_scheduler is not None:
        # FIXME: I don't think this actually work correctly, but we don't have an example of an lr_scheduler that is not a DictConfig
        if "scheduler" in lr_scheduler:
            config["lr_scheduler"] = dict(lr_scheduler)
            config["lr_scheduler"]["scheduler"] = config["lr_scheduler"]["scheduler"](
                config["optimizer"]
            )
        else:
            config["lr_scheduler"] = lr_scheduler(config["optimizer"])

    return config
