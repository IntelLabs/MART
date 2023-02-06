from typing import List

import pytest

from tests.helpers.package_available import _SH_AVAILABLE

if _SH_AVAILABLE:
    import sh


def run_sh_command(command: List[str]):
    """Default method for executing shell commands with pytest and sh package."""
    msg = None
    try:
        # Return stdout to help debug failed tests.
        return sh.python(command)
    except sh.ErrorReturnCode as e:
        msg = e.stderr.decode()
    if msg:
        pytest.fail(msg=msg)
