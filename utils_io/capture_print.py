import sys
import io
from contextlib import contextmanager


@contextmanager
def capture_print():
    """
    Context manager to capture the output of `print` statements.

    This function temporarily redirects `sys.stdout` and `sys.stderr` to
    `io.StringIO` objects, allowing you to capture printed output within
    a `with` block.

    Yields:
        tuple: A tuple containing the `StringIO` objects for `stdout` and `stderr`.
    """
    new_out, new_err = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        new_out.close()
        new_err.close()
