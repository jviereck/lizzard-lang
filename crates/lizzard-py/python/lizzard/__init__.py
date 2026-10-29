"""lizzard: typed Python, checked and run by lizzard inside CPython.

Importing this package installs the import hook for lizzard modules (``.py``
files marked ``# lizzard: module``). ``python -m lizzard app.py`` runs a program.
"""

from . import _lizzard
from ._loader import CheckError, install

install()


def stats():
    """Counters of array element accesses (direct and generic)."""
    return _lizzard._stats()


__all__ = ["CheckError", "install", "stats"]
