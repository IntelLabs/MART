from typing import List

import pytest

from tests.helpers.package_available import _SH_AVAILABLE

if _SH_AVAILABLE:
    import sh


def run_sh_command(command: List[str]):
    """Default method for executing shell commands with pytest and sh package."""
    try:
        sh.python(command, _out=lambda log: print(log.strip()))
    except sh.ErrorReturnCode as e:
        msg = e.stderr.decode()
        # The error message could be empty.
        pytest.fail(msg=msg)
