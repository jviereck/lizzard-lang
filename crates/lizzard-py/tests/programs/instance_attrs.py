# lizzard: module
# Instance attributes read and written directly in CPython's inline values,
# and every case where that must not happen or must notice a change: unset
# and deleted attributes, a materialized or replaced __dict__, a class
# attribute or property added later, __getattr__, subclasses at the same
# site, more attributes than the shared keys hold, values set from Python
# code, and threads writing while others read. Output must match CPython.
import threading

import instance_attrs_patch as patch


class P:
    def __init__(self, x: float, y: float) -> None:
        self.x: float = x
        self.y: float = y

    def bump(self) -> None:
        self.x = self.x + 1.0
        self.y = self.y * 2.0


class Q(P):
    def __init__(self, x: float, y: float) -> None:
        super().__init__(x, y)
        self.z: float = 0.0


def total(ps: list[P]) -> float:
    t = 0.0
    for p in ps:
        p.bump()
        t += p.x + p.y
    return t


def read_x(p: P) -> float:
    return p.x


def read_safely(p: P) -> str:
    try:
        return str(p.x)
    except AttributeError as e:
        return f"AttributeError: {e}"


class Big:
    def __init__(self) -> None:
        self.a0: int = 0
        self.a1: int = 1
        self.a2: int = 2
        self.a3: int = 3
        self.a4: int = 4
        self.a5: int = 5
        self.a6: int = 6
        self.a7: int = 7
        self.a8: int = 8
        self.a9: int = 9


def big_sum(b: Big) -> int:
    return b.a0 + b.a3 + b.a9


class Greeter:
    def __init__(self, name: str) -> None:
        self.name: str = name

    def greet(self, punct: str) -> str:
        return "hi " + self.name + punct


class Loud(Greeter):
    def greet(self, punct: str) -> str:
        return "HI " + self.name.upper() + punct


def greet_all(gs: list[Greeter]) -> list[str]:
    out: list[str] = []
    for g in gs:
        out.append(g.greet("!"))
    return out


def hammer(p: P, n: int) -> None:
    for _ in range(n):
        p.x = p.x + 1.0


if __name__ == "__main__":
    ps: list[P] = [P(1.0, 2.0), Q(3.0, 4.0), P(0.5, 0.25)]
    for _ in range(3):
        print(total(ps))
    q = ps[1]
    # Unset and deleted attributes: AttributeError from the generic path.
    print(read_safely(patch.blank(P)))
    patch.delete(q, "x")
    print(read_safely(q))
    q.x = 9.0
    print(read_x(q))
    # A materialized __dict__ shares the values; a replaced one does not.
    print(patch.materialize(ps[0]), read_x(ps[0]))
    ps[0].x = 11.0
    print(read_x(ps[0]), patch.dict_x(ps[0]))
    patch.replace_dict(ps[2])
    print(read_x(ps[2]))
    ps[2].x = 13.0
    print(read_x(ps[2]), patch.dict_x(ps[2]))
    # Values set from Python code are seen.
    patch.set_x(ps[0], 21.0)
    print(read_x(ps[0]))
    # Many attributes, and a class with __getattr__ at the same sites.
    print(big_sum(Big()), big_sum(Big()))
    print(read_x(patch.with_getattr(P)))
    # A property added to the class later shadows the instance value.
    patch.add_property(P)
    print(read_x(P(5.0, 6.0)), read_x(ps[0]))
    patch.remove_property(P)
    print(read_x(ps[0]))
    # Methods: cached per call site, but an instance attribute that shadows
    # the method, a method replaced on the class, and subclasses still win.
    gs: list[Greeter] = [Greeter("a"), Loud("b"), Greeter("c")]
    print(greet_all(gs), greet_all(gs))
    patch.shadow_method(gs[2], "greet")
    print(greet_all(gs))
    patch.replace_method(Greeter, "greet")
    print(greet_all(gs))
    # Threads incrementing their own objects while others read them.
    objs = [P(0.0, 0.0) for _ in range(4)]
    ts = [threading.Thread(target=hammer, args=(o, 5000)) for o in objs]
    for t in ts:
        t.start()
    seen = [read_x(o) for o in objs for _ in range(100)]
    for t in ts:
        t.join()
    print([o.x for o in objs], all([0.0 <= v <= 5000.0 for v in seen]))
