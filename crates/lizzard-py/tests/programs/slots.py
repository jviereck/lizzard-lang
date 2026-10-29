# lizzard: module
# __slots__ classes: compiled attribute loads and stores use an inline cache
# keyed on the type's version tag; replacing an attribute on the class (a
# new version) must take effect at once. Its output must match CPython's.
class P:
    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float) -> None:
        self.x: float = x
        self.y: float = y

    def norm2(self) -> float:
        return self.x * self.x + self.y * self.y


class Q(P):
    __slots__ = ("z",)

    def __init__(self, x: float, y: float, z: float) -> None:
        super().__init__(x, y)
        self.z: float = z


def total(ps: list[P]) -> float:
    t = 0.0
    for p in ps:
        t += p.x + p.norm2()
        p.y = p.y + 1.0
    return t


def unset() -> None:
    q = Q.__new__(Q)
    try:
        print(q.x)
    except AttributeError as e:
        print("AttributeError", e)


if __name__ == "__main__":
    ps: list[P] = [P(1.0, 2.0), Q(3.0, 4.0, 5.0), P(0.5, 0.5)]
    for _ in range(3):
        print(total(ps))
    unset()
    import slots_patch
    slots_patch.make_x_constant()  # the class changes: the caches must miss
    print(total(ps))
