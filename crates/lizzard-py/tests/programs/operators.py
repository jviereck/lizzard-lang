# lizzard: module
# Operator overloading: arithmetic and reflected operators, in-place operators,
# @, unary operators, rich and chained comparisons, truthiness, len, indexing,
# membership, iteration, __call__, and builtins dispatching to dunders (abs,
# float, hash, min, max, sorted). Its output must match plain CPython's.
from typing import Iterator


class V:
    def __init__(self, x: float, y: float) -> None:
        self.x: float = x
        self.y: float = y

    def __add__(self, o: V) -> V:
        return V(self.x + o.x, self.y + o.y)

    def __iadd__(self, o: V) -> V:
        self.x += o.x
        self.y += o.y
        return self

    def __mul__(self, k: float) -> V:
        return V(self.x * k, self.y * k)

    def __rmul__(self, k: float) -> V:
        return V(self.x * k, self.y * k)

    def __matmul__(self, o: V) -> float:
        return self.x * o.x + self.y * o.y

    def __neg__(self) -> V:
        return V(-self.x, -self.y)

    def __abs__(self) -> float:
        return (self.x * self.x + self.y * self.y) ** 0.5

    def __eq__(self, o: object) -> bool:
        return isinstance(o, V) and self.x == o.x and self.y == o.y

    def __lt__(self, o: V) -> bool:
        return abs(self) < abs(o)

    def __le__(self, o: V) -> bool:
        return abs(self) <= abs(o)

    def __bool__(self) -> bool:
        return self.x != 0 or self.y != 0

    def __len__(self) -> int:
        return 2

    def __getitem__(self, i: int) -> float:
        return self.x if i == 0 else self.y

    def __contains__(self, v: float) -> bool:
        return v == self.x or v == self.y

    def __iter__(self) -> Iterator[float]:
        yield self.x
        yield self.y

    def __call__(self, k: float) -> V:
        return self * k

    def __float__(self) -> float:
        return abs(self)

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __repr__(self) -> str:
        return f"V({self.x}, {self.y})"


def main() -> None:
    a = V(3.0, 4.0)
    b = V(1.0, 2.0)
    c = a + b
    c += b
    print(c, a * 2.0, 2.0 * a, a @ b, -a, abs(a), a == V(3.0, 4.0), a != b)
    print(b < a, a > b, b <= a, a >= b, 1.0 < 2.0 < 3.0, b < a <= a)
    print(bool(V(0.0, 0.0)), len(a), a[1], 4.0 in a, list(a), a(0.5), float(a), hash(a) == hash(V(3.0, 4.0)))
    print(sorted([a, b, c]), max(a, b), min([a, b, c]))


if __name__ == "__main__":
    main()
