from io import StringIO
from typing import List

import pytest

from tests.helpers.package_available import _SH_AVAILABLE

if _SH_AVAILABLE:
    import sh


def run_sh_command(command: List[str]):
    """Default method for executing shell commands with pytest and sh package."""
    msg = None
    try:
        with StringIO() as buf:
            # Turn off color by _tty_out=False
            sh.python(command, _out=buf, _tty_out=False)
            output = buf.getvalue()
        return output
    except sh.ErrorReturnCode as e:
        msg = e.stderr.decode()
    if msg:
        pytest.fail(msg=msg)
