#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import logging
from typing import Callable, Iterable, OrderedDict

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
        _return_as_dict: <a list of keys to wrap the returned tuple as a dictionary>
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

        if "output" not in modules:
            raise ValueError("Modules must have an module named 'output'")

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

            output = module(step=step, sequence=sequence, **kwargs)

            if key in kwargs:
                logger.warn(f"Module {module} replaces kwargs key {key}!")
            kwargs[key] = output

            # Pop the executed module to proceed with the sequence
            sequence.popitem(last=False)

        return kwargs["output"]


class ReturnKwargs(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, **kwargs):
        return kwargs


class CallWith(torch.nn.Module):
    def __init__(
        self,
        module: Callable,
        _call_with_args_: Iterable[str] | None = None,
        _return_as_dict_: Iterable[str] | None = None,
        **kwarg_keys,
    ) -> None:
        super().__init__()

        self.module = module
        self.arg_keys = _call_with_args_ or []
        self.kwarg_keys = kwarg_keys or {}
        self.return_keys = _return_as_dict_

    def forward(self, *args, **kwargs):
        orig_class = self.module.__class__
        arg_keys = self.arg_keys
        kwarg_keys = self.kwarg_keys
        kwargs = DotDict(kwargs)

        # Sometimes we receive positional arguments because some modules use nn.Sequential
        # which has a __call__ function that passes positional args. So we pass along args
        # as it and assume these consume the first len(args) of arg_keys.
        remaining_arg_keys = arg_keys[len(args) :]

        for key in remaining_arg_keys + list(kwarg_keys.values()):
            if key not in kwargs:
                raise Exception(
                    f"Module {orig_class} wants arg named '{key}' but only received kwargs: {', '.join(kwargs.keys())}."
                )

        selected_args = [kwargs[key] for key in arg_keys[len(args) :]]
        selected_kwargs = {key: kwargs[val] for key, val in kwarg_keys.items()}

        # FIXME: Add better error message
        ret = self.module(*args, *selected_args, **selected_kwargs)

        if self.return_keys:
            if not isinstance(ret, tuple):
                raise Exception(
                    f"Module {orig_class} does not return multiple unnamed variables, so we can not dictionarize the return."
                )
            if len(self.return_keys) != len(ret):
                raise Exception(
                    f"Module {orig_class} returns {len(ret)} items, but {len(self.return_keys)} return_keys were specified."
                )
            ret = dict(zip(self.return_keys, ret))

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
                raise KeyError("No {subkey} in " + ".".join([key, *subkeys]))

        return value


class GroupNorm32(torch.nn.GroupNorm):
    """GroupNorm with default num_groups=32; can be pass to ResNet's norm_layer.

    See: torch.nn.GroupNorm
    """

    def __init__(self, *args, **kwargs):
        super().__init__(32, *args, **kwargs)


# FIXME: This must exist already?!
class Sum(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        return sum(args)
