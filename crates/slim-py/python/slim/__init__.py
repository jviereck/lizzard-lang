"""slim: typed Python, checked and run by slim inside CPython.

Importing this package installs the import hook for slim modules (``.py``
files marked ``# slim: module``). ``python -m slim app.py`` runs a program.
"""

from . import _slim
from ._loader import CheckError, install

install()


def stats():
    """Counters of array element accesses (direct and generic)."""
    return _slim._stats()


__all__ = ["CheckError", "install", "stats"]
