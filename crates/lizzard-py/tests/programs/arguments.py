# lizzard: module
# *args, **kwargs, keyword-only and positional-only parameters, defaults and
# calls with * and **. Its output must match plain CPython's.


def total(*nums: int) -> int:
    t = 0
    for n in nums:
        t += n
    return t


def describe(name: str, *tags: str, sep: str = ", ", **extra: int) -> str:
    parts = [name, sep.join(tags)]
    for k in sorted(extra):
        parts.append(f"{k}={extra[k]}")
    return " | ".join(parts)


def kw_only(a: int, *, scale: float = 2.0, offset: float) -> float:
    return a * scale + offset


def pos_only(a: int, b: int = 10, /) -> int:
    return a - b


class Logger:
    def __init__(self, prefix: str, *, upper: bool = False) -> None:
        self.prefix: str = prefix
        self.upper: bool = upper

    def log(self, *msgs: str, **fields: str) -> str:
        line = self.prefix + " ".join(msgs)
        for k, v in fields.items():
            line += f" {k}={v}"
        return line.upper() if self.upper else line


def main() -> None:
    print(total(), total(1, 2, 3), describe("x"), describe("x", "a", "b", sep="/", n=1, m=2))
    print(kw_only(3, offset=0.5), kw_only(3, scale=1.0, offset=1.0), pos_only(15), pos_only(15, 5))
    nums = [4, 5, 6]
    t = (1, "two")
    opts: dict[str, int] = {"n": 3}
    print(total(*nums), total(0, *nums, *[7]), describe("y", *["p", "q"], **opts), describe(*("z",), sep="-"))
    lg = Logger("> ", upper=True)
    print(lg.log("hello", "world", who="me"), Logger("# ").log(*["a", "b"], **{"k": "v"}))
    pair_first(*t)


def pair_first(a: int, b: str) -> None:
    print("pair", a, b)


if __name__ == "__main__":
    main()
