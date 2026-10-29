# lizzard: module
# Nested functions and closures (nonlocal, recursion, several levels), lambdas
# typed from Callable (key= functions, map, filter, defaults) and generic
# higher-order functions. Its output must match plain CPython's.
from typing import Callable


class Person:
    def __init__(self, name: str, age: int) -> None:
        self.name: str = name
        self.age: int = age

    def __repr__(self) -> str:
        return f"Person({self.name}, {self.age})"


def make_counter(start: int) -> Callable[[], int]:
    count = start

    def inc() -> int:
        nonlocal count
        count += 1
        return count

    return inc


def make_adders() -> list[Callable[[int], int]]:
    out: list[Callable[[int], int]] = []
    for n in range(3):
        out.append(lambda x, n=n: x + n)
    return out


def compose(f: Callable[[int], int], g: Callable[[int], int]) -> Callable[[int], int]:
    return lambda x: f(g(x))


def apply_twice[T](f: Callable[[T], T], x: T) -> T:
    return f(f(x))


def fib_memo(n: int) -> int:
    memo: dict[int, int] = {}

    def fib(k: int) -> int:
        if k < 2:
            return k
        if k in memo:
            return memo[k]
        v = fib(k - 1) + fib(k - 2)
        memo[k] = v
        return v

    return fib(n)


def outer_inner() -> str:
    tag = "x"

    def middle() -> str:
        def inner() -> str:
            return tag * 2

        return inner() + "!"

    tag = "y"
    return middle()


def double(x: int) -> int:
    return x * 2


def main() -> None:
    c = make_counter(10)
    print(c(), c(), c())
    print([f(10) for f in make_adders()])
    inc_then_double = compose(double, lambda v: v + 1)
    print(inc_then_double(4), apply_twice(double, 3), apply_twice(lambda s: s + "!", "hi"))
    people = [Person("ann", 31), Person("bob", 25), Person("cy", 40)]
    print(sorted(people, key=lambda p: p.age))
    print(max(people, key=lambda p: p.age).name, min(people, key=lambda p: len(p.name)).name)
    people.sort(key=lambda p: p.name, reverse=True)
    print([p.name for p in people])
    print(list(map(lambda p: p.age * 2, people)), list(filter(lambda p: p.age > 30, people)))
    print(fib_memo(30), outer_inner())
    print(list(map(double, [1, 2])))


if __name__ == "__main__":
    main()
