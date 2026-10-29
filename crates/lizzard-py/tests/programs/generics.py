# lizzard: module
# Property setters, and generic functions (PEP 695 and TypeVar), with bounds.
# Its output must match plain CPython's.
from typing import TypeVar

T = TypeVar("T")


class Temperature:
    def __init__(self, celsius: float) -> None:
        self._c: float = celsius

    @property
    def celsius(self) -> float:
        return self._c

    @celsius.setter
    def celsius(self, value: float) -> None:
        if value < -273.15:
            raise ValueError("below absolute zero")
        self._c = value

    @property
    def fahrenheit(self) -> float:
        return self._c * 9 / 5 + 32


class Shape:
    def area(self) -> float:
        return 0.0


class Sq(Shape):
    def __init__(self, s: float) -> None:
        self.s: float = s

    def area(self) -> float:
        return self.s * self.s


def first[U](xs: list[U]) -> U:
    return xs[0]


def pair[A, B](a: A, b: B) -> tuple[B, A]:
    return (b, a)


def largest[S: Shape](shapes: list[S]) -> S:
    best = shapes[0]
    for s in shapes:
        if s.area() > best.area():
            best = s
    return best


def identity(x: T) -> T:
    return x


def repeat[V](x: V, n: int) -> list[V]:
    return [x for _ in range(n)]


def main() -> None:
    t = Temperature(20.0)
    t.celsius = 100.0
    print(t.celsius, t.fahrenheit)
    try:
        t.celsius = -300.0
    except ValueError as e:
        print("rejected:", e)
    n: int = first([3, 4])
    s: str = first(["a"])
    b, a = pair(1, "x")
    print(n, s, b.upper(), a + 1)
    big = largest([Sq(1.0), Sq(3.0), Sq(2.0)])
    print(big.s, identity(5) + 1, identity("q").upper(), repeat("ab", 3))


if __name__ == "__main__":
    main()
