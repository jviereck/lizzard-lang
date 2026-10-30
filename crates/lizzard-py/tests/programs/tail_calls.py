# lizzard: module
# Self tail calls run as loops; their output must match CPython's,
# RecursionError at the recursion limit included.
from typing import Any

import tail_calls_patch


def fact(n: int, acc: int) -> int:
    if n <= 1:
        return acc
    return fact(n - 1, acc * n)


def gcd(a: int, b: int) -> int:
    if b == 0:
        return a
    return gcd(b, a % b)


def twice(x: int, y: int, k: int) -> int:
    if k == 0:
        return x + y
    return twice(y, y, k - 1)


def search(xs: list[int], i: int, want: int) -> int:
    for x in xs[i:]:
        if x == want:
            return i
        return search(xs, i + 1, want)
    return -1


def unbound(n: int) -> int:
    if n > 2:
        y = n
    if n == 0:
        return 0
    try:
        print("y", y)
    except UnboundLocalError:
        print("unbound")
    return unbound(n - 1)


def guarded(n: int) -> int:
    if n == 0:
        return 0
    try:
        return guarded(n - 1)
    finally:
        pass


def loose(n: Any, acc: int) -> int:
    if n == 0:
        return acc
    return loose(n - 1, acc + 1)


def countdown(n: int) -> int:
    if n == 0:
        return 0
    if n == 3:
        tail_calls_patch.rebind()
    return countdown(n - 1)


def other(n: int) -> int:
    print("other", n)
    return -n


def down(n: int) -> int:
    return down(n + 1)


def total(n: int, acc: float) -> float:
    if n == 0:
        return acc
    return total(n - 1, acc + n * 0.5)


if __name__ == "__main__":
    print(fact(5, 1), fact(30, 1))
    print([gcd(i * 7, 91) for i in range(1, 300)][-6:])
    print([twice(i, i + 1, 3) for i in range(300)][-3:])
    print(sum([search([1, 2, 3, 4, 5], 0, i) for i in range(300)]))
    print(unbound(5))
    print(guarded(50), loose(50, 0))
    print(sum([total(100, 0.0) for _ in range(300)]))
    print(fact(800, 1) % 1000003, gcd(10**30 + 7, 10**20 + 3))
    try:
        fact(5000, 1)
    except RecursionError:
        print("RecursionError")
    try:
        down(0)
    except RecursionError:
        print("RecursionError")
    print(countdown(6))
