# lizzard: module
# Paths of compiled code: specialized, guarded, inlined code meeting values
# its caches did not see. Its output must match CPython's, compiled
# (eagerly too) and interpreted.
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
