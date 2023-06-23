#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import logging
from collections import OrderedDict
from contextlib import nullcontext
from typing import Callable, Iterable

import torch

__all__ = ["GroupNorm32", "SequentialDict", "ReturnKwargs", "CallWith", "Sum"]

logger = logging.getLogger(__name__)


class SequentialDict(torch.nn.ModuleDict):
    """A special Sequential container where we can rewire the input and output of each module in
    each sequence.

    First, we need to define all modules as name-object pairs in the dictionary `modules`.

    Second, we need to define a sequence of (some of) those modules in a list of dictionaries.

    Each module in the sequence can be defined by
    <module name>:
        _name_: <return key>
        _call_with_args_: <a list of *args>
        _return_as_dict_: <a list of keys to wrap the returned tuple as a dictionary>
        **kwargs

    All intermediate output from each module are stored in the dictionary `kwargs` in `forward()`
    as we execute a sequence, using the <return key> as the key.
    In this way, a module is able to consume results from other modules in terms of *args or **kwargs.
    We usually just use a module's name as the <return key>, in which case `_name_` can be omitted.
    We can also use a different name as <return key> by specifying `_name_` separately,
    so that we can re-run a module with a different set of input, and save the return value with
    <return key> in `kwargs` of `forward()`.

    If `_call_with_args_` is the only config of a module and the <return key> is the same as <module name>,
    we can simplify the configuration as
    <module name>:
        <a list of *args>

    In addition, note that we can always use *args as **kwargs in Python.

    Sequences should be represented as <step_key: sequence> in the sequences dictionary.
    """

    def __init__(self, modules, sequences=None):
        super().__init__(modules)

        self._sequences = {
            name: self.parse_sequence(sequence) for name, sequence in sequences.items()
        }
        self._sequences[None] = self

    def parse_sequence(self, sequence):
        if sequence is None:
            return self

        module_dict = OrderedDict()
        for module_info in sequence:
            # Treat strings as modules that don't require CallWith
            if isinstance(module_info, str):
                module_info = {module_info: {}}

            if not isinstance(module_info, dict) or len(module_info) != 1:
                raise ValueError(
                    f"Each module config in the sequence list should be a length-one dict: {module_info}"
                )

            module_name, module_cfg = list(module_info.items())[0]

            if not isinstance(module_cfg, dict):
                # We can omit the key of _call_with_args_ if it is the only config.
                module_cfg = {"_call_with_args_": module_cfg}

            # The return name could be different from module_name when a module is used more than once.
            return_name = module_cfg.pop("_name_", module_name)
            module = CallWith(self[module_name], **module_cfg)
            module_dict[return_name] = module
        return module_dict

    def forward(self, step=None, sequence=None, **kwargs):
        # Try to fetch the customized architectural graph.
        # Backward compatible. We may get rid of step in the future.
        if sequence is None:
            sequence = self._sequences[step]

        # Make a copy of sequence, because it will be destructed in the while loop.
        sequence = sequence.copy()

        while len(sequence) > 0:
            # Don't pop the first element yet, because it may be used to re-evaluate the model.
            key, module = next(iter(sequence.items()))

            # FIXME: Add better error message
            output = module(step=step, sequence=sequence, **kwargs)

            if key in kwargs:
                logger.warn(f"Module {module} replaces kwargs key {key}!")
            kwargs[key] = output

            # Pop the executed module to proceed with the sequence
            sequence.popitem(last=False)

        # return kwargs as DotDict
        return DotDict(kwargs)


class ReturnKwargs(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, **kwargs):
        return kwargs


class CallWith:
    def __init__(
        self,
        module: Callable,
        _call_with_args_: Iterable[str] | None = None,
        _return_as_dict_: Iterable[str] | None = None,
        _train_mode_: bool | None = None,
        _inference_mode_: bool | None = None,
        **kwarg_keys,
    ) -> None:
        super().__init__()

        self.module = module
        self.arg_keys = _call_with_args_
        self.kwarg_keys = kwarg_keys
        self.return_keys = _return_as_dict_
        self.train_mode = _train_mode_
        self.inference_mode = _inference_mode_

    def __call__(
        self,
        *args,
        _args_: Iterable[str] | None = None,
        _return_keys_: Iterable[str] | None = None,
        _train_mode_: bool | None = None,
        _inference_mode_: bool | None = None,
        **kwargs,
    ):
        module_name = self.module.__class__.__name__

        arg_keys = _args_ or self.arg_keys
        kwarg_keys = self.kwarg_keys
        _train_mode_ = _train_mode_ or self.train_mode
        _inference_mode_ = _inference_mode_ or self.inference_mode

        # Change and replace args and kwargs that we call module with
        if arg_keys is not None or len(kwarg_keys) > 0:
            arg_keys = arg_keys or []

            kwargs = DotDict(kwargs)  # we need to lookup values using dot strings
            args = list(args)  # tuple -> list

            # Sometimes we receive positional arguments because some modules use nn.Sequential
            # which has a __call__ function that passes positional args. So we pass along args
            # as it and assume these consume the first len(args) of arg_keys.
            arg_keys = arg_keys[len(args) :]

            # Extend args with selected kwargs using arg_keys
            try:
                args.extend(
                    [
                        kwargs[kwargs_key] if isinstance(kwargs_key, str) else kwargs_key
                        for kwargs_key in arg_keys
                    ]
                )
            except KeyError as ex:
                raise Exception(
                    f"{module_name} only received kwargs: {', '.join(kwargs.keys())}."
                ) from ex

            # Replace kwargs with selected kwargs using kwarg_keys
            try:
                kwargs = {
                    name: kwargs[kwargs_key] if isinstance(kwargs_key, str) else kwargs_key
                    for name, kwargs_key in kwarg_keys.items()
                }
            except KeyError as ex:
                raise Exception(
                    f"{module_name} only received kwargs: {', '.join(kwargs.keys())}."
                ) from ex

        # Apply train mode and inference mode, if necessary, and call module with args and kwargs
        context = nullcontext()
        if isinstance(self.module, torch.nn.Module):
            old_train_mode = self.module.training

            if _train_mode_ is not None:
                self.module.train(_train_mode_)

            if _inference_mode_ is not None:
                context = torch.inference_mode(mode=_inference_mode_)

        with context:
            # FIXME: Add better error message
            ret = self.module(*args, **kwargs)

        if isinstance(self.module, torch.nn.Module):
            if _train_mode_ is not None:
                self.module.train(old_train_mode)

        # Change returned values into dictionary, if necessary
        return_keys = _return_keys_ or self.return_keys
        if return_keys:
            if not isinstance(ret, tuple):
                raise Exception(
                    f"{module_name} does not return multiple unnamed variables, so we can not dictionarize the return."
                )
            if len(return_keys) != len(ret):
                raise Exception(
                    f"Module {module_name} returns {len(ret)} items, but {len(return_keys)} return_keys were specified."
                )
            ret = dict(zip(return_keys, ret))

        return ret


class DotDict(dict):
    def __init__(self, kwargs):
        super().__init__(kwargs)

    def __contains__(self, key):
        if super().__contains__(key):
            # Some keys may contain dot.
            return True

        key, *subkeys = key.split(".")

        if not super().__contains__(key):
            return False

        value = super().__getitem__(key)

        # Walk object hierarchy, preferring getattr over __getitem__
        for subkey in subkeys:
            if hasattr(value, subkey):
                value = getattr(value, subkey)
            elif isinstance(value, dict) and subkey in value:
                value = value[subkey]
            else:
                return False

        return True

    def __getitem__(self, key):
        if super().__contains__(key):
            # Some keys may contain dot.
            return super().__getitem__(key)

        key, *subkeys = key.split(".")
        value = super().__getitem__(key)

        # Walk object hierarchy, preferring getattr over __getitem__
        for subkey in subkeys:
            if hasattr(value, subkey):
                value = getattr(value, subkey)
            elif isinstance(value, dict) and subkey in value:
                value = value[subkey]
            else:
                raise KeyError(f"No {subkey} in " + ".".join([key, *subkeys]))

        return value


class GroupNorm32(torch.nn.GroupNorm):
    """GroupNorm with default num_groups=32; can be pass to ResNet's norm_layer.

    See: torch.nn.GroupNorm
    """

    def __init__(self, *args, **kwargs):
        super().__init__(32, *args, **kwargs)


# FIXME: This must exist already?!
class Sum(torch.nn.Module):
    def __init__(self, weights=None):
        super().__init__()

        self.weights = weights

    def forward(self, *values, weights=None):
        weights = weights or self.weights

        if weights is None:
            weights = [1 for _ in values]

        assert len(weights) == len(values)
        return sum(value * weight for value, weight in zip(values, weights))
