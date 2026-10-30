# Plain Python helper for jit_paths.py: changes a class at runtime.
import sys


def patch():
    cls = sys.modules["__main__"].Body
    cls.step = lambda self, dt: setattr(self, "x", self.x - dt)
    cls.ratio = lambda self, d: self.x * d
