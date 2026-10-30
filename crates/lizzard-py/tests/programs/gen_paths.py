# lizzard: module
# Generators resumed by compiled code and by the interpreter: values,
# ends, exceptions and the handled exception, throw/close, re-entry.
from collections.abc import Generator, Iterable, Iterator

import gen_paths_patch


def count(n: int) -> Iterator[int]:
    i = 0
    while i < n:
        yield i
        i += 1


def squares(n: int) -> int:
    t = 0
    for x in count(n):
        t += x * x
    return t


def big(n: int) -> int:
    # The sum leaves the machine ints.
    t = 2**62
    for x in count(n):
        t += x * 2**40
    return t


def failing(n: int) -> Iterator[int]:
    for i in range(n):
        if i == 7:
            raise ValueError(f"at {i}")
        yield i


def caught(n: int) -> int:
    t = 0
    try:
        for x in failing(n):
            t += x
    except ValueError as e:
        t += 1000 + len(str(e))
    return t


def uncaught(n: int) -> int:
    t = 0
    for x in failing(n):
        t += x
    return t


def handled() -> Iterator[str]:
    try:
        raise KeyError("inner")
    except KeyError:
        yield "in handler"
        yield "still"
    yield "after"


def handled_seen() -> list[str]:
    out: list[str] = []
    try:
        raise ValueError("outer")
    except ValueError:
        for s in handled():
            out.append(s + ":" + gen_paths_patch.handled_name())
    out.append("done:" + gen_paths_patch.handled_name())
    return out


def guarded(log: list[str]) -> Iterator[int]:
    try:
        yield 1
        yield 2
        yield 3
    finally:
        log.append("finally")


def early() -> list[str]:
    log: list[str] = []
    for x in guarded(log):
        if x == 2:
            break
    log.append("after loop")
    return log


def delegating(n: int) -> Generator[int, None, int]:
    r = yield from returning(n)
    yield r
    return r + 1


def returning(n: int) -> Generator[int, None, int]:
    for i in range(n):
        yield i
    return 100


def total_of(xs: Iterable[int]) -> int:
    t = 0
    for x in xs:
        t += x
    return t


def mixed(k: int) -> int:
    t = 0
    for i in range(k):
        t += total_of([1, 2, 3]) + total_of(count(4)) + total_of((5, 6))
    return t


def pair() -> Generator[int, None, None]:
    yield 0
    yield 1


def squares_of(n: int, k: int) -> Iterator[int]:
    # Computed ints: handed to a loop that takes ints as they are.
    for i in range(n):
        yield i * k


def sum_squares(n: int, k: int) -> int:
    t = 0
    for x in squares_of(n, k):
        t += x
    return t


def kept(n: int) -> list[int]:
    # The ints are needed as objects too.
    out: list[int] = []
    for x in squares_of(n, 3):
        out.append(x)
    return out


def flags(n: int) -> Iterator[int]:
    for i in range(n):
        yield i % 2 == 0


def count_flags(n: int) -> int:
    t = 0
    for x in flags(n):
        t += x
    return t


def guarded_ints(n: int) -> Iterator[int]:
    for i in range(n):
        try:
            if i == 3:
                raise KeyError(i)
            yield i
        except KeyError:
            yield -1


def sum_guarded(n: int) -> int:
    t = 0
    for x in guarded_ints(n):
        t += x
    return t


def chained(n: int) -> Iterator[int]:
    yield from squares_of(n, 2)
    yield 7


def sum_chained(n: int) -> int:
    t = 0
    for x in chained(n):
        t += x
    return t


def rest_of(g: Iterator[int]) -> int:
    t = 0
    for x in g:
        t += x
    return t


def thrown() -> Generator[int, None, None]:
    try:
        yield 1
    except KeyError as e:
        yield 100 + len(str(e))
    yield 3


if __name__ == "__main__":
    print([squares(n) for n in (0, 1, 300)][-1], squares(3000))
    print(big(5), big(2000))
    print(caught(5), caught(20), [caught(20) for _ in range(300)][-1])
    try:
        uncaught(20)
    except ValueError as e:
        print("ValueError", e)
    print(handled_seen())
    print([early() for _ in range(300)][-1])
    print(list(delegating(3)), sum([total_of(delegating(50)) for _ in range(300)]))
    print(mixed(300))
    print([sum_squares(n, 3) for n in range(300)][-1], sum_squares(1000, 2**40), sum_squares(10, 2**62))
    print(kept(5), [len(kept(50)) for _ in range(300)][-1])
    print([count_flags(n) for n in range(300)][-1], count_flags(5))
    print([sum_guarded(10) for _ in range(300)][-1], [sum_chained(20) for _ in range(300)][-1])
    part = squares_of(10, 1)
    print(next(part), next(part), rest_of(part), [rest_of(squares_of(n, 1)) for n in range(300)][-1])
    g = thrown()
    print(next(g), g.throw(KeyError("k")), next(g), list(g))
    print(gen_paths_patch.reenter(count(3)))
    h = pair()
    print(gen_paths_patch.send_first(h, 5), list(h))
