# lizzard: module
# Constructing instances: plain classes take a fast path (allocate, call
# __init__) that must behave exactly like type.__call__ in every case it
# does not apply to or must notice. Its output must match plain CPython's.
from typing import Iterator

import constructors_patch as patch


class V:
    def __init__(self, x: float, y: float) -> None:
        self.x: float = x
        self.y: float = y

    def add(self, o: "V") -> "V":
        return V(self.x + o.x, self.y + o.y)


class Opts:
    def __init__(self, name: str, size: int = 3, *, flag: bool = False) -> None:
        self.name: str = name
        self.size: int = size
        self.flag: bool = flag


class Slotted:
    __slots__ = ("a",)

    def __init__(self, a: int) -> None:
        self.a: int = a


class Picky:
    def __init__(self, n: int) -> None:
        if n < 0:
            raise ValueError(f"negative: {n}")
        self.n: int = n


class Node:
    def __init__(self, depth: int) -> None:
        self.child: Node | None = Node(depth - 1) if depth > 0 else None


class Flip:
    def __init__(self, flip: bool) -> None:
        if flip:
            self.b: int = 2
            self.a: int = 1
        else:
            self.a = 1
            self.b = 2
        self.c: int = 3


def count(n: Node) -> int:
    c = n.child
    return 1 + (count(c) if c is not None else 0)


def make_points(k: int) -> Iterator[V]:
    for i in range(k):
        yield V(float(i), 1.0)


def build(cls: type[V], x: float) -> V:
    return cls(x, 0.0)


if __name__ == "__main__":
    a = V(0.0, 0.0)
    for _ in range(5):
        a = a.add(V(1.0, 0.5))
    print(a.x, a.y, type(a).__name__)
    o = Opts("x")
    p = Opts("y", 7, flag=True)
    print(o.name, o.size, o.flag, p.name, p.size, p.flag)
    print(Slotted(4).a)
    try:
        Picky(-1)
    except ValueError as e:
        print("ValueError", e)
    print(Picky(2).n, count(Node(30)))
    print([pt.x for pt in make_points(3)])
    # First assignments record the insertion order, as CPython does.
    print(patch.items(Flip(False)), patch.items(Flip(True)), patch.items(V(1.0, 2.0)))
    # Python subclasses, a custom __new__, a metaclass, an abstract class.
    for cls in [patch.py_subclass(V), patch.with_new(V), patch.with_meta(V)]:
        v = build(cls, 2.0)
        print(type(v).__name__, v.x, v.y, patch.extra(v))
    try:
        build(patch.returns_value(V), 1.0)
    except TypeError as e:
        print("TypeError", e)
    try:
        build(patch.abstract(V), 1.0)
    except TypeError as e:
        print("TypeError", e)
    # __init__ replaced on the class: the next construction uses the new one.
    print(build(V, 3.0).x)
    patch.replace_init(V)
    print(build(V, 3.0).x)
