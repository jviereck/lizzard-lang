# lizzard: module
# Lists, dicts and tuples: displays, indexing and slicing, methods,
# iteration, unpacking, comprehensions, membership and augmented
# assignment. Its output must match plain CPython's.


def lists() -> None:
    xs = [5, 3, 8, 1]
    xs.sort()
    print(xs, xs[1:3], xs[::-1], xs[-1], len(xs))
    print(sorted(xs, reverse=True), sum(xs), min(xs), max(xs), max(3, 7))
    xs[0] += 100
    del xs[1]
    xs.insert(0, 7)
    xs.extend([9, 9])
    print(xs, 8 in xs, 2 not in xs, xs.index(8), xs.count(9), xs.pop(), xs.pop(0))
    ys = xs + [0] * 2
    zs = ys
    zs += [42]
    print(ys, zs is ys, ys == [101, 5, 8, 9, 0, 0, 42])
    grid = [[r * 3 + c for c in range(3)] for r in range(2)]
    grid[1][0] = -1
    print(grid, grid[1][2], [row[0] for row in grid])
    floats: list[float] = [1, 2.5]
    floats.append(3)
    print(floats, sum(floats))
    empty: list[str] = []
    print(empty, not empty, list(range(2, 5)), list(reversed(xs)))


def dicts() -> None:
    d: dict[str, list[int]] = {"a": [1], "b": []}
    d["b"].append(2)
    d["c"] = [3, 4]
    d.setdefault("e", []).append(5)
    for k in sorted(d):
        print(k, d[k], d.get(k, [])[0:1])
    for i, (k, v) in enumerate(d.items()):
        print(i, k, len(v))
    print(list(d.keys()), list(d.values()), "a" in d, "z" in d, d.get("z", [0]))
    del d["a"]
    print(d.pop("e"), len(d), d)
    counts: dict[str, int] = {}
    for w in "the cat and the hat and the bat".split():
        counts[w] = counts.get(w, 0) + 1
    print(counts, max(counts.values()))
    squares = {n: n * n for n in range(6) if n % 2 == 1}
    inverse = {v: k for k, v in squares.items()}
    print(squares, inverse, dict(squares) == squares)


def tuples() -> None:
    t = (1, 2.5, "x")
    n, f, s = t
    print(t + (True,), t[-1], t[0:2], n + f, s * 3, len(t), 2.5 in t)
    pairs = list(zip([1, 2, 3], ["a", "b", "c"]))
    print(pairs, pairs[2][1])
    for i, (num, ch) in enumerate(pairs):
        print(i, num, ch)
    a, b = 1, 2
    a, b = b, a
    print(a, b, divmod_pair(17, 5))
    q, r = divmod_pair(9, 4)
    print(q, r)


def divmod_pair(a: int, b: int) -> tuple[int, int]:
    return (a // b, a % b)


def strings() -> None:
    words = "the quick brown fox".split()
    print(", ".join([w.upper() for w in words]), "-".join(words), words[1][::-1])
    total = 0
    for w in words:
        if w.startswith("q"):
            break
        total += len(w)
    print(total, f"{3.14159:.2f}|{42:>5}|{'s'!r}|{words[0]:^7}|")
    chars = [c for c in "hello" if c != "l"]
    print(chars, "ell" in "hello", "hello"[1:4], "a,b,,c".split(","))


if __name__ == "__main__":
    lists()
    dicts()
    tuples()
    strings()
