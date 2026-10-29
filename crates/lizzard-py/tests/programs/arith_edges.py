# lizzard: module
# Arithmetic and comparisons at the edges of the int and float fast paths:
# compact and big ints, negatives, bools, mixed types, division by zero,
# NaN, -0.0 and infinities. Its output must match plain CPython's.


def ints(a: int, b: int) -> list[object]:
    return [a + b, a - b, a * b, a < b, a <= b, a == b, a != b, a > b, a >= b]


def floats(a: float, b: float) -> list[object]:
    out: list[object] = [a + b, a - b, a * b, a < b, a == b, a != b, a >= b]
    try:
        out.append(a / b)
    except ZeroDivisionError as e:
        out.append(f"ZeroDivisionError: {e}")
    return out


def mixed(a: int, b: float) -> list[object]:
    return [a + b, a * b, b - a, a < b, a == b]


def flags(a: bool, b: bool) -> list[object]:
    return [a + b, a - b, a * b, a < b, a == b]


def count(n: int) -> int:
    t = 0
    i = 0
    while i < n:
        if i % 3 == 0:
            t += i * i
        i += 1
    return t


if __name__ == "__main__":
    edge = (1 << 30) - 1
    for a, b in [(0, 0), (3, 4), (-5, 7), (edge, edge), (-edge, edge), (edge, 2), (1 << 40, 3), (-(1 << 62), -(1 << 62)), (10**30, -1)]:
        print(a, b, ints(a, b))
    nan = float("nan")
    inf = float("inf")
    for x, y in [(1.5, 2.0), (-0.0, 0.0), (1.0, 0.0), (nan, nan), (inf, -inf), (1e308, 10.0), (0.1, 0.2)]:
        print(x, y, floats(x, y))
    print(mixed(3, 0.5), mixed(-2, -2.0))
    print(flags(True, True), flags(False, True))
    print(count(10000))
