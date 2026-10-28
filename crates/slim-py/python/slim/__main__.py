"""``python -m slim app.py [args]``: run a program, checking it if it is a slim module."""

import os
import sys
import types

from ._loader import CheckError, install, is_slim_source, run_source


def main(argv):
    if not argv:
        print("usage: python -m slim app.py [args]", file=sys.stderr)
        return 2
    path = argv[0]
    with open(path, encoding="utf-8") as f:
        source = f.read()
    install()
    sys.argv = list(argv)
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    module = types.ModuleType("__main__")
    module.__file__ = path
    sys.modules["__main__"] = module
    try:
        if is_slim_source(path):
            run_source(source, path, module)
        else:
            exec(compile(source, path, "exec"), module.__dict__)
    except CheckError as e:
        for d in e.diagnostics:
            print(d, file=sys.stderr)
        return 1
    return 0


sys.exit(main(sys.argv[1:]))
