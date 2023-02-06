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
            sh.python(command, _out=buf, _env={"NO_COLOR": "1"})
            output = buf.getvalue()
        return output
    except sh.ErrorReturnCode as e:
        msg = e.stderr.decode()
    # The error message could be empty.
    if msg is not None:
        pytest.fail(msg=msg)
