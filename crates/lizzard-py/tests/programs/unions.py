# lizzard: module
# Optional values and unions: narrowing by `is None`, `isinstance`, truthiness,
# `and`/`or`, early returns, loops and `assert`. Its output must match plain
# CPython's.
from typing import Optional


class Node:
    def __init__(self, value: int, next: Node | None = None) -> None:
        self.value: int = value
        self.next: Node | None = next


class LinkedList:
    def __init__(self) -> None:
        self.head: Node | None = None
        self.size: int = 0

    def push(self, v: int) -> None:
        self.head = Node(v, self.head)
        self.size += 1

    def find(self, v: int) -> Node | None:
        cur = self.head
        while cur is not None:
            if cur.value == v:
                return cur
            cur = cur.next
        return None

    def last(self) -> Optional[int]:
        if self.head is None:
            return None
        cur = self.head
        while cur.next is not None:
            cur = cur.next
        return cur.value

    def values(self) -> list[int]:
        out: list[int] = []
        cur = self.head
        while cur:
            out.append(cur.value)
            cur = cur.next
        return out


class Circle:
    def __init__(self, r: float) -> None:
        self.r: float = r

    def area(self) -> float:
        return 3.0 * self.r * self.r


class Square:
    def __init__(self, s: float) -> None:
        self.s: float = s

    def area(self) -> float:
        return self.s * self.s


def describe(x: int | str | None) -> str:
    if x is None:
        return "nothing"
    if isinstance(x, int):
        return f"int {x + 1}"
    return "str " + x.upper()


def total(shapes: list[Circle | Square]) -> float:
    t = 0.0
    for s in shapes:
        t += s.area()
        if isinstance(s, Square):
            t += s.s
    return t


def first_even(xs: list[int]) -> int | None:
    for x in xs:
        if x % 2 == 0:
            return x
    return None


def main() -> None:
    ll = LinkedList()
    for v in [3, 1, 4, 1, 5]:
        ll.push(v)
    print(ll.values(), ll.size, ll.last())
    n = ll.find(4)
    print(n is not None and n.value == 4, ll.find(9))
    if n is not None and n.next is not None:
        print("after 4:", n.next.value)
    print(describe(None), describe(41), describe("hi"))
    print(total([Circle(1.0), Square(2.0)]))
    e = first_even([1, 3, 6, 8])
    print(e if e is not None else -1, (first_even([1]) or 0) + 1)
    val: int | None = None
    for i in range(3):
        if val is None:
            val = i
        else:
            val += i
    assert val is not None
    print(val * 2)
    empty = LinkedList()
    print(empty.last(), empty.values())


if __name__ == "__main__":
    main()
