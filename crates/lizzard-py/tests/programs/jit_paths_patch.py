# Plain Python helper for jit_paths.py: changes a class at runtime.
import sys


def patch():
    cls = sys.modules["__main__"].Body
    cls.step = lambda self, dt: setattr(self, "x", self.x - dt)
    cls.ratio = lambda self, d: self.x * d


_held = []
_tagged = []


def hold(p):
    import weakref
    _held.append((weakref.ref(p), p.x))


def tag(p):
    p.extra = 1
    _tagged.append(p)


def report():
    # A weakly referenced instance keeps its values; tagged ones keep the tag.
    alive = [(r() is not None and r().x == x) for r, x in _held]
    return sum(alive), len(alive), all(hasattr(p, "extra") and p.extra == 1 for p in _tagged)


def names(p):
    return list(vars(p))
