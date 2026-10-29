# Plain Python helper for slots.py: changes a class at runtime.
import sys


def make_x_constant():
    cls = sys.modules["__main__"].P
    cls.x = property(lambda self: 42.0, lambda self, v: None)
