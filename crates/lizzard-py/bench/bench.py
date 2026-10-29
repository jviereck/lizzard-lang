# lizzard: module
# Benchmarks: `python bench.py` (CPython), `python -m lizzard bench.py` (compiled),
# `LIZZARD_JIT=0 python -m lizzard bench.py` (interpreted). Build the extension
# with `maturin develop --release` first.
import time
from typing import Callable


def loop(n: int) -> int:
    t = 0
    for i in range(n):
        t += i * i
    return t


def fib(n: int) -> int:
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)


class V:
    def __init__(self, x: float, y: float) -> None:
        self.x: float = x
        self.y: float = y

    def add(self, o: "V") -> "V":
        return V(self.x + o.x, self.y + o.y)


def vecs(n: int) -> float:
    a = V(0.0, 0.0)
    d = V(1.0, 0.5)
    for _ in range(n):
        a = a.add(d)
    return a.x + a.y


class Body:
    def __init__(self, x: float, v: float) -> None:
        self.x: float = x
        self.v: float = v

    def step(self, dt: float) -> None:
        self.x += self.v * dt


def bodies(n: int) -> float:
    bs = [Body(float(i), 1.0) for i in range(100)]
    for _ in range(n // 100):
        for b in bs:
            b.step(0.01)
    return bs[0].x


class VS:
    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float) -> None:
        self.x: float = x
        self.y: float = y

    def add(self, o: "VS") -> "VS":
        return VS(self.x + o.x, self.y + o.y)


def vecs_slots(n: int) -> float:
    a = VS(0.0, 0.0)
    d = VS(1.0, 0.5)
    for _ in range(n):
        a = a.add(d)
    return a.x + a.y


class BodyS:
    __slots__ = ("x", "v")

    def __init__(self, x: float, v: float) -> None:
        self.x: float = x
        self.v: float = v

    def step(self, dt: float) -> None:
        self.x += self.v * dt


def bodies_slots(n: int) -> float:
    bs = [BodyS(float(i), 1.0) for i in range(100)]
    for _ in range(n // 100):
        for b in bs:
            b.step(0.01)
    return bs[0].x


def lists(n: int) -> int:
    xs: list[int] = []
    for i in range(n):
        xs.append(i)
    s = 0
    for x in xs:
        if x % 3 == 0:
            s += x
    return s


def timeit(name: str, fn: Callable[[int], object], arg: int) -> None:
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        fn(arg)
        best = min(best, time.perf_counter() - t0)
    print(f"{name:9} {best:.3f}s")


if __name__ == "__main__":
    timeit("loop", loop, 2_000_000)
    timeit("fib", fib, 25)
    timeit("vecs", vecs, 500_000)
    timeit("vecs/s", vecs_slots, 500_000)
    timeit("bodies", bodies, 500_000)
    timeit("bodies/s", bodies_slots, 500_000)
    timeit("lists", lists, 1_000_000)
