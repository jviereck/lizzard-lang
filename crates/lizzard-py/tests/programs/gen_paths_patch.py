# Plain Python helpers for gen_paths.py.
import sys


def handled_name():
    e = sys.exception()
    return type(e).__name__ if e is not None else "None"


def reenter(g):
    # A generator resuming itself: ValueError, as for Python generators.
    def inner():
        yield from g

    it = inner()
    try:
        next(it)
        return [next(g), next(it)]
    except ValueError as e:
        return str(e)


def send_first(g, v):
    # A just-started generator takes None only.
    try:
        g.send(v)
    except TypeError as e:
        return f"TypeError {e}"


def py_gen(n):
    # A Python generator under a Lizzard `yield from`.
    for i in range(n):
        yield i * 2
