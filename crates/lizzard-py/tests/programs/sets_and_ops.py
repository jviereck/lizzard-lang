# lizzard: module
# Sets (displays, comprehensions, methods, operators), bitwise and shift
# operators, powers and more builtins. Its output must match plain CPython's.


def sets() -> None:
    a = {1, 2, 3}
    b: set[int] = {3, 4}
    a.add(10)
    a.discard(2)
    print(sorted(a), sorted(a | b), sorted(a & b), sorted(a - b), sorted(a ^ b), 3 in a, 2 in a)
    print(len(a), {1, 2} <= {1, 2, 3}, a.issubset([1, 3, 10, 11]), sorted(a.union([7])))
    evens = {x for x in range(10) if x % 2 == 0}
    seen: set[str] = set()
    for w in "a b a c b".split():
        if w not in seen:
            seen.add(w)
    print(sorted(evens), sorted(seen), set() == set(), sorted(set([3, 1, 3])))
    c = set(b)
    c |= {9}
    b.update({0})
    print(sorted(c), sorted(b))


def bits() -> None:
    x = 0b1011
    print(x & 6, x | 4, x ^ 1, x << 2, x >> 1, ~x, -x, +x, 2 ** 10, 2.0 ** 3, True & False, True | False)
    flags = 0
    flags |= 1 << 3
    flags ^= 1
    flags <<= 1
    print(flags, bin(flags), hex(255), oct(8), divmod(17, 5), round(2.567, 2), round(2.5))
    n = 7
    p = n ** 2
    print(p + 1, any([0, 0, 1]), all([]), ord("a"), chr(98), next(iter([5, 6])))


if __name__ == "__main__":
    sets()
    bits()
