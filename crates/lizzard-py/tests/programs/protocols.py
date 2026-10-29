# lizzard: module
# Protocols as parameter types: builtin values (int, float, str, list, ...)
# and classes match SupportsFloat, SupportsIndex, Sized, Hashable,
# Collection, Reversible and SupportsAbs by their methods; a function or
# lambda matches a callback protocol; a recursive protocol matches a tree.
# Its output must match plain CPython's.
from typing import Callable, Hashable, Protocol, Sized, SupportsAbs, SupportsFloat, SupportsIndex, SupportsInt
from collections.abc import Collection, Container, Reversible, Sequence


class Meters:
    def __init__(self, v: float) -> None:
        self.v: float = v

    def __float__(self) -> float:
        return self.v

    def __len__(self) -> int:
        return 3


class Point:
    def __init__(self, x: int) -> None:
        self.x: int = x

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Point) and other.x == self.x


class Handler(Protocol):
    def __call__(self, code: int, msg: str) -> bool: ...


class Node(Protocol):
    def children(self) -> Sequence["Node"]: ...


class Tree:
    def __init__(self) -> None:
        self.kids: list[Tree] = []

    def children(self) -> list["Tree"]:
        return self.kids


def as_float(x: SupportsFloat) -> float:
    return float(x)


def index(x: SupportsIndex) -> int:
    return [10, 20, 30][x]


def size(x: Sized) -> int:
    return len(x)


def key(x: Hashable) -> int:
    return hash(x) % 1


def first[T](xs: Collection[T]) -> T:
    for x in xs:
        return x
    raise ValueError("empty")


def backwards[T](xs: Reversible[T]) -> list[T]:
    return list(reversed(xs))


def dispatch(h: Handler) -> bool:
    return h(404, "missing")


def on_error(code: int, msg: str) -> bool:
    print("handler", code, msg)
    return code >= 400


def count(n: Node) -> int:
    return 1 + sum([count(c) for c in n.children()])


def magnitude[T](x: SupportsAbs[T]) -> T:
    return abs(x)


def main() -> None:
    print(as_float(3), as_float(2.5), as_float(True), as_float(Meters(1.5)))
    print(index(1), index(True))
    print(size("abc"), size([1, 2]), size({1: 2}), size(Meters(2.0)), size((1, 2, 3)))
    print(key(1), key("a"), key((1, 2)), key(2.5))
    print(first([4, 5]), first("xy"), first({7: "a"}))
    print(backwards([1, 2, 3]), backwards(range(3)))
    print(dispatch(on_error), dispatch(lambda c, m: c < 400))
    t = Tree()
    t.kids.append(Tree())
    t.kids[0].kids.append(Tree())
    print(count(t))
    m: int = magnitude(-3)
    f: float = magnitude(-2.5)
    print(m, f)
    c: Container[int] = [1, 2]
    print(3 in c)


if __name__ == "__main__":
    main()
