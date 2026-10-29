# lizzard: module
# Literal types: aliases of literal unions, literal arguments and defaults,
# narrowing by ==, and literals widening to their base type in arithmetic
# and in unannotated variables. Its output must match plain CPython's.
from typing import Literal

type Mode = Literal["r", "w", "a"]
Level = Literal[0, 1, 2]


def open_mode(m: Mode) -> str:
    if m == "r":
        return "reading"
    if m != "w":
        return "appending"
    return "writing " + m.upper()


def level_name(n: Level, loud: Literal[True, False] = False) -> str:
    names = ["low", "mid", "high"]
    s = names[n] + str(n + 1)
    return s.upper() if loud else s


def pick(flag: bool) -> Mode:
    m: Mode = "w" if flag else "a"
    return m


def main() -> None:
    print(open_mode("r"), open_mode("w"), open_mode("a"))
    print(level_name(0), level_name(2, True))
    x: Literal["r", "w"] = "r"
    print(open_mode(x), pick(True), pick(False))
    y = "r"
    modes: list[Mode] = ["r", "w"]
    print(modes, [open_mode(k) for k in modes])


if __name__ == "__main__":
    main()
