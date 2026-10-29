# lizzard: module
# Dataclasses: fields with defaults and default_factory, inheritance, frozen,
# kw_only and generic dataclasses. Its output must match plain CPython's.
from dataclasses import dataclass, field


@dataclass
class Point:
    x: float
    y: float = 0.0

    def norm2(self) -> float:
        return self.x * self.x + self.y * self.y

    def moved(self, dx: float) -> Point:
        return Point(self.x + dx, self.y)


@dataclass
class Point3(Point):
    z: float = 0.0


@dataclass(frozen=True, order=True)
class Version:
    major: int
    minor: int

    def bump(self) -> Version:
        return Version(self.major, self.minor + 1)


@dataclass
class Inventory:
    owner: str
    items: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, item: str) -> None:
        self.items.append(item)
        self.counts[item] = self.counts.get(item, 0) + 1


@dataclass(kw_only=True)
class Options:
    verbose: bool = False
    level: int


@dataclass
class Pair[T]:
    first: T
    second: T

    def swap(self) -> Pair[T]:
        return Pair(self.second, self.first)


def main() -> None:
    p = Point(3.0, 4.0)
    print(p, p.norm2(), p.moved(1.0), Point(1.0), p == Point(3.0, 4.0))
    q = Point3(1.0, 2.0, 3.0)
    print(q, q.z, Point3(1.0, z=5.0))
    v = Version(1, 2)
    print(v, v.bump(), v < v.bump(), v >= Version(2, 0), v == Version(1, 2), max(v, v.bump()))
    inv = Inventory("ann")
    inv.add("apple")
    inv.add("apple")
    inv.add("pear")
    print(inv, Inventory("bob").items)
    print(Options(level=3), Options(level=1, verbose=True).verbose)
    pr = Pair(1, 2).swap()
    n: int = pr.first
    print(pr, n + 1)


if __name__ == "__main__":
    main()
