# lizzard: module
# Generators: yield, send, throw, close, return values, yield from (Lizzard and
# Python generators in both directions), finally on close, generator methods
# and sys.exception() across yields. Its output must match plain CPython's.
import sys
from typing import Generator, Iterator

import pygens


def count(n: int) -> Iterator[int]:
    i = 0
    while i < n:
        yield i
        i += 1


def fib() -> Iterator[int]:
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b


def take[T](it: Iterator[T], n: int) -> list[T]:
    out: list[T] = []
    for x in it:
        if len(out) >= n:
            break
        out.append(x)
    return out


def accumulate() -> Generator[int, int, str]:
    total = 0
    while True:
        x = yield total
        if x < 0:
            return f"done at {total}"
        total += x


def resilient() -> Generator[str, int, int]:
    handled = 0
    while True:
        try:
            v = yield f"ready {handled}"
        except ValueError as e:
            handled += 1
            print("resilient caught", e, "while handling:", sys.exception())
            continue
        if v == 99:
            return handled


def guarded(name: str) -> Generator[int, None, None]:
    try:
        yield 1
        yield 2
    finally:
        print("cleanup", name)


def chain(a: Iterator[int], b: Iterator[int]) -> Generator[int, None, str]:
    yield from a
    r = yield from pygens.countdown(2)
    yield from b
    return "chained after " + r


def outer() -> Generator[int, int, str]:
    r = yield from accumulate()
    yield -1
    return r


class Tree:
    def __init__(self, value: int, children: list[Tree]) -> None:
        self.value: int = value
        self.children: list[Tree] = children

    def __iter__(self) -> Iterator[int]:
        yield self.value
        for c in self.children:
            yield from c


def main() -> None:
    print(list(count(4)), take(fib(), 10), sum(count(5)), [x * x for x in count(4)])
    acc = accumulate()
    print(next(acc), acc.send(5), acc.send(7))
    try:
        acc.send(-1)
    except StopIteration as e:
        print("returned", e.value)
    r = resilient()
    print(next(r), r.throw(ValueError("one")), r.throw(ValueError("two")), r.send(1))
    try:
        r.send(99)
    except StopIteration as e:
        print("resilient returned", e.value)
    g = guarded("a")
    print(next(g))
    g.close()
    print(list(guarded("b")))
    g2 = guarded("c")
    next(g2)
    del g2
    print("chain:", list(chain(count(2), iter([7, 8]))))
    c = chain(count(1), count(1))
    out: list[int | str] = []
    while True:
        try:
            out.append(next(c))
        except StopIteration as e:
            out.append(e.value)
            break
    print(out)
    o = outer()
    print(next(o), o.send(3), o.send(-1), next(o, 0))
    print("python drives lizzard:", pygens.drive(resilient()))
    print("python drives python:", pygens.drive(pygens.echo()))
    d = pygens.delegate(accumulate())
    print(next(d), d.send(4))
    try:
        d.send(-5)
    except StopIteration as e:
        print(e.value)
    print(list(Tree(1, [Tree(2, [Tree(4, [])]), Tree(3, [])])))
    e = pygens.echo()
    w = chain(e, count(0))
    print(next(w))
    w.close()


if __name__ == "__main__":
    main()
