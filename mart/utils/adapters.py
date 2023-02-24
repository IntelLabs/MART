#
# Copyright (C) 2022 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

from typing import Any, Callable

__all__ = ["CallableAdapter", "PartialInstanceWrapper"]


class CallableAdapter:
    """Adapter to make an object callable."""

    def __init__(self, instance, redirecting_fn):
        """

        Args:
            instance (object): instance to make callable.
            redirecting_fn (str): name of the function that will be invoked in the `__call__` method.
        """
        assert instance is not None
        assert redirecting_fn != ""

        self.instance = instance
        self.redirecting_fn = redirecting_fn

    def __call__(self, *args, **kwargs):
        """

        Args:
            args (Any): values to use in the callable method.
            kwargs (Any): keyword values to use in the callable method.
        """
        function = getattr(self.instance, self.redirecting_fn)

        assert callable(function)

        return function(*args, **kwargs)


class PartialInstanceWrapper:
    """Make a partial class object callable."""

    def __init__(self, partial: Callable, wrapper: Callable):
        """

        Args:
            partial (Callable): A partial of a class object.
            adapter (Callable): An adapter that creates the `__call__` method.
        """
        self.partial = partial
        self.wrapper = wrapper

    def __call__(self, *args: Any, **kwargs: Any) -> Callable:
        # Turn a partial to a class object.
        instance = self.partial(*args, **kwargs)
        # Make the object callable.
        return self.wrapper(instance)
