# Plain Python helper for rebinding.py: changes globals and builtins.
import builtins
import sys

_saved = builtins.abs


def shadow_abs():
    sys.modules["__main__"].__dict__["abs"] = lambda x: -1


def unshadow_abs():
    del sys.modules["__main__"].__dict__["abs"]


def patch_builtin_abs():
    builtins.abs = lambda x: 99


def restore_builtin_abs():
    builtins.abs = _saved
