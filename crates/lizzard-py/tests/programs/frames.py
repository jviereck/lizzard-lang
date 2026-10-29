# lizzard: module
# Calls between Lizzard functions: recursion, exceptions crossing several
# frames (caught in callers, finally blocks on the way out), methods,
# generators calling functions and functions driving generators, closures,
# keyword and default arguments, deep recursion and RecursionError.
# Its output must match plain CPython's.
from typing import Iterator

log: list[str] = []


def fib(n: int) -> int:
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)


def is_even(n: int) -> bool:
    if n == 0:
        return True
    return is_odd(n - 1)


def is_odd(n: int) -> bool:
    if n == 0:
        return False
    return is_even(n - 1)


def fail(depth: int) -> int:
    try:
        if depth == 0:
            raise ValueError("bottom")
        return fail(depth - 1) + 1
    finally:
        log.append(f"finally {depth}")


def catch_it() -> str:
    try:
        fail(3)
        return "no"
    except ValueError as e:
        return f"caught {e}"


def reraise(n: int) -> int:
    try:
        return fail(n)
    except ValueError:
        log.append("reraise")
        raise


class Acc:
    def __init__(self, start: int) -> None:
        self.total: int = start

    def add(self, x: int) -> int:
        self.total = self.total + x
        return self.total

    def add_twice(self, x: int) -> int:
        self.add(x)
        return self.add(x)


def squares(n: int) -> Iterator[int]:
    for i in range(n):
        yield square(i)


def square(x: int) -> int:
    return x * x


def total(n: int) -> int:
    t = 0
    for v in squares(n):
        t = t + v
    return t


def scaled(x: int, factor: int = 3) -> int:
    return x * factor


def make_adder(k: int) -> int:
    def add(x: int) -> int:
        return x + k

    return add(1) + add(2)


def depth(n: int) -> int:
    if n == 0:
        return 0
    return 1 + depth(n - 1)


def runaway(n: int) -> int:
    return runaway(n + 1)


if __name__ == "__main__":
    print(fib(15), is_even(100), is_odd(7))
    print(catch_it())
    try:
        reraise(2)
    except ValueError as e:
        print("outer", e)
    print(log)
    a = Acc(1)
    print(a.add_twice(5), a.total)
    print(total(6), scaled(4), scaled(4, factor=5), make_adder(10))
    print(depth(500))
    try:
        runaway(0)
    except RecursionError:
        print("RecursionError")
    print(fib(10))
