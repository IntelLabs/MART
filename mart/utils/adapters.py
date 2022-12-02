#
# Copyright (C) 2022 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

__all__ = ["CallableAdapter"]


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
