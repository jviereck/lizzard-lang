# lizzard: module
# Paths of compiled code: specialized, guarded, inlined code meeting values
# its caches did not see. Its output must match CPython's, compiled
# (eagerly too) and interpreted.
from collections.abc import Iterable

import jit_paths_patch


class Body:
    def __init__(self, x: float, v: float) -> None:
        self.x: float = x
        self.v: float = v

    def step(self, dt: float) -> None:
        self.x += self.v * dt

    def scaled(self, k: float) -> float:
        return self.x * k

    def ratio(self, d: float) -> float:
        return self.x / d


def run(bs: list[Body], n: int) -> float:
    for _ in range(n):
        for b in bs:
            b.step(0.5)
    t = 0.0
    for b in bs:
        t += b.scaled(2.0)
    return t


def ints(n: int) -> int:
    t = 0
    for i in range(n):
        t += i * i * i * 1000003
    return t


def mixed(a: float, b: float) -> float:
    return a * b + a - b


def divide(bs: list[Body], d: float) -> float:
    t = 0.0
    for b in bs:
        t += b.ratio(d)
    return t


def keep(b: Body) -> float:
    # A float read, then the attribute overwritten in place: the read
    # value must not change.
    old = b.x
    b.x = b.x + 1.0
    return old


def mods(n: int) -> int:
    s = 0
    for i in range(-n, n):
        s += i % 7 + i // 3 + (i & 5) - (i | 2) + (i ^ 3)
    return s


def boom(n: int) -> int:
    if n == 0:
        raise ValueError("boom")
    return boom(n - 1) + 1


def catch(n: int) -> int:
    try:
        return boom(n)
    except ValueError as e:
        return -len(str(e))


def sq(n: int) -> int:
    if n < 0:
        return -n * n
    return n * n


def sq1(k: int) -> int:
    return sq(k) + 1


def sqs(n: int, k: int) -> int:
    s = 0
    for i in range(n):
        s += sq1(i - n // 2) % 1000 + sq1(k) % 7
    return s


def total(xs: list[int]) -> int:
    t = 0
    for x in xs:
        t += x
    return t


def total_of(xs: Iterable[int]) -> int:
    t = 0
    for x in xs:
        t += x
    return t


def halves(xs: list[int]) -> list[int]:
    out: list[int] = []
    it = iter(xs)
    for x in it:
        if x == 3:
            break
        out.append(x)
    for x in it:
        out.append(-x)
    # Exhausted: stays so, even as the list grows.
    xs.append(99)
    for x in it:
        out.append(x)
    return out


def grow(xs: list[int]) -> int:
    n = 0
    for x in xs:
        if x < 5:
            xs.append(x + 10)
        n += 1
    return n


def checked(n: int) -> int:
    if n == -77:
        raise ValueError("seventy-seven")
    return n * n


def boxes(n: int) -> list[int]:
    return [n, n + 1]


def half(x: float) -> float:
    return x / 2.0


def lazies(m: int, k: int) -> int:
    s = 0
    for i in range(m):
        # Arguments unboxed here, passed on unboxed.
        s += checked(i * k) % 1000 + boxes(i + k)[1] + int(half(i * 0.5))
    return s


def long_loop(n: int) -> str:
    t = 0
    f = 0.0
    k = 0
    for i in range(n):
        t += i * 3
        f += i * 0.25
        k = k + 1
    return f"{t} {f} {k}"


def deep(n: int) -> int:
    return deep(n + 1) + 1


class P:
    def __init__(self, x: float, y: float) -> None:
        self.x: float = x
        self.y: float = y

    def moved(self, d: float) -> "P":
        return P(self.x + d, self.y)


class PS:
    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float) -> None:
        self.x: float = x
        self.y: float = y

    def moved(self, d: float) -> "PS":
        return PS(self.x + d, self.y)


def walk(n: int) -> P:
    p = P(0.0, 1.0)
    for _ in range(n):
        p = p.moved(1.0)
    return p


def walk_slots(n: int) -> PS:
    p = PS(0.0, 1.0)
    for _ in range(n):
        p = p.moved(1.0)
    return p


def trail(n: int) -> list[P]:
    # Every step kept: none may be constructed in.
    ps: list[P] = []
    p = P(0.0, 0.0)
    for _ in range(n):
        ps.append(p)
        p = p.moved(0.5)
    return ps


def watched(n: int) -> float:
    # Weak references and extra attributes from Python: not reused.
    p = P(0.0, 0.0)
    for i in range(n):
        if i % 50 == 0:
            jit_paths_patch.hold(p)
        if i % 70 == 0:
            jit_paths_patch.tag(p)
        p = p.moved(1.0)
    return p.x


if __name__ == "__main__":
    bs = [Body(float(i), 1.0) for i in range(5)]
    for _ in range(3):
        print(run(bs, 50))
    # Ints become floats (the float paths see ints) and back.
    for x in range(300):
        bs[x % 5].x = x
    print(run(bs, 10))
    print([type(b.x).__name__ for b in bs])
    # Arithmetic overflowing an i64.
    print(ints(500), ints(5000))
    # A float hint meeting two ints: the result stays an int.
    print(mixed(2.0, 3.0), mixed(2, 3), mixed(2, 3.5))
    # Division by zero inside an inlined method.
    print(divide(bs, 2.0))
    try:
        divide(bs, 0.0)
    except ZeroDivisionError as e:
        print("ZeroDivisionError", e)
    b = Body(1.5, 0.0)
    for _ in range(300):
        keep(b)
    print(keep(b), b.x)
    print(mods(300))
    # Instances about to be released, constructed in.
    w = walk(1000)
    print(w.x, w.y, jit_paths_patch.names(w), walk_slots(1000).x)
    t = trail(400)
    print(len({id(p) for p in t}), t[0].x, t[-1].x)
    print(watched(500))
    print(jit_paths_patch.report())
    # Ints returned unboxed from inlined calls, then too big for that.
    print(sqs(300, 3), sqs(10, 2**31), sqs(10, 2**40), sqs(10, -(2**40)), sqs(10, 2**70))
    print(sq1(2**31), sq1(2**40), sq1(-(2**40)), sq1(2**70), sq1(3))
    # Inlined calls whose arguments stay unboxed, then need their objects.
    print(lazies(300, 1), lazies(10, 2**31), lazies(5, 2**40), lazies(3, -(2**62)))
    try:
        lazies(100, -7)
    except ValueError as e:
        print("ValueError", e)
    # Values kept unboxed across loop safepoints.
    print(long_loop(5000), long_loop(100000))
    # Iterating lists inline; then other iterables.
    print(sum([total([i, 2, 3]) for i in range(300)]), total(list(range(10))))
    print(halves([1, 2, 3, 4, 5]), [len(halves([1, 2, 3, i])) for i in range(300)][-1])
    print([grow([1, 7, 3]) for _ in range(300)][-1])
    print(sum([total_of([i, 2]) for i in range(300)]), total_of(range(5)), total_of({1: 0, 2: 0}), total_of((4, 5)))
    # Exceptions leaving directly called compiled code.
    print([catch(i % 7) for i in range(300)][-5:])
    try:
        deep(0)
    except RecursionError:
        print("RecursionError")
    # The class changes: guards miss, inlined code hands over.
    jit_paths_patch.patch()
    print(run(bs, 3))
    print(divide(bs, 4.0))
