# Plain Python helper for tail_calls.py: rebinds a module function.
import sys


def rebind():
    m = sys.modules["__main__"]
    m.countdown = m.other
