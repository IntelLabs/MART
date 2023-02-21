#
# Copyright (C) 2022 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

from typing import Any, Callable

__all__ = ["CallableAdapter", "PartialAdapter"]


class CallableAdapter:
    """Adapter to make an object callable."""

    def __init__(self, redirecting_fn):
        """

        Args:
            redirecting_fn (str): name of the function that will be invoked in the `__call__` method.
        """
        assert redirecting_fn != ""

        self.redirecting_fn = redirecting_fn

    def __call__(self, instance):
        """

        Args:
            instance (object): instance to make callable.
        """

        function = getattr(instance, self.redirecting_fn)
        assert callable(function)
        return function


class PartialAdapter:
    """Make a partial class object callable."""

    def __init__(self, partial: Callable, adapter: Callable):
        """

        Args:
            partial (Callable): A partial of a class object.
            adapter (Callable): An adapter to make an object callable.
        """
        self.partial = partial
        self.adapter = adapter

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Turn a partial to a class object.
        instance = self.partial(*args, **kwargs)
        # Make the object callable.
        return self.adapter(instance)
