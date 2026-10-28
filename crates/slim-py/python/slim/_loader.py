"""Import hook for slim modules.

A ``.py`` file whose first lines contain ``# slim: module`` is checked by
slim before it runs: its functions are replaced by slim functions, and its
top level runs on CPython as usual.
"""

import ast
import importlib.abc
import importlib.machinery
import os
import sys
import warnings

from . import _slim

MARKER = "# slim: module"


class CheckError(Exception):
    """A slim module failed to type-check; ``diagnostics`` lists the errors."""

    def __init__(self, diagnostics):
        self.diagnostics = list(diagnostics)
        super().__init__("\n".join(self.diagnostics))


def is_slim_source(path):
    try:
        with open(path, encoding="utf-8") as f:
            for _ in range(5):
                line = f.readline()
                if not line:
                    break
                if line.strip() == MARKER:
                    return True
    except OSError:
        pass
    return False


def _pkg_version(name):
    from importlib import metadata

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _env():
    overlays = [os.path.join(os.path.dirname(__file__), "stubs")]
    extra = os.environ.get("SLIM_STUBS_PATH")
    if extra:
        overlays = [p for p in extra.split(os.pathsep) if p] + overlays
    roots = [os.path.abspath(p or os.getcwd()) for p in sys.path]
    return {
        "overlay_roots": overlays,
        "search_roots": [p for p in roots if os.path.isdir(p)],
        "version": tuple(sys.version_info[:2]),
        "platform": sys.platform,
        "stdlib": sorted(sys.stdlib_module_names),
        "pkg_version": _pkg_version,
    }


def run_source(source, path, module):
    """Check ``source`` and execute it as ``module``."""
    compiled, diagnostics, warns = _slim.compile_module(source, path, module.__name__, _env())
    for w in warns:
        warnings.warn(w, stacklevel=2)
    if diagnostics:
        raise CheckError(diagnostics)
    namespace = module.__dict__
    functions = {name: compiled.make_function(name, namespace) for name in compiled.function_names()}
    tree = ast.parse(source, path)
    for i, node in enumerate(tree.body):
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            load = ast.Subscript(ast.Name("__slim_functions__", ast.Load()), ast.Constant(node.name), ast.Load())
            tree.body[i] = ast.copy_location(ast.Assign([ast.Name(node.name, ast.Store())], load), node)
    ast.fix_missing_locations(tree)
    code = compile(tree, path, "exec")
    namespace["__slim_functions__"] = functions
    try:
        exec(code, namespace)
    finally:
        namespace.pop("__slim_functions__", None)


class SlimLoader(importlib.machinery.SourceFileLoader):
    def exec_module(self, module):
        path = self.get_filename(module.__name__)
        source = self.get_data(path).decode("utf-8")
        run_source(source, path, module)


class SlimFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not spec.origin or not spec.origin.endswith(".py"):
            return None
        if not is_slim_source(spec.origin):
            return None
        spec.loader = SlimLoader(fullname, spec.origin)
        return spec


def install():
    if not any(isinstance(f, SlimFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, SlimFinder())
