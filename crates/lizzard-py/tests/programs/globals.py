# lizzard: module
# Module globals read by functions, and written after `global`. Its output must
# match plain CPython's.
SCALE: float = 2.5
NAMES = ["a", "b"]
counter = 0
registry: dict[str, int] = {}
limit: int


def scaled(x: float) -> float:
    return x * SCALE


def bump() -> int:
    global counter
    counter += 1
    return counter


def register(name: str) -> None:
    registry[name] = len(registry)
    NAMES.append(name)


def first() -> str:
    return NAMES[0]


if __name__ == "__main__":
    limit = 3
    print(scaled(2.0))
    for _ in range(limit):
        bump()
    print(counter, bump())
    register("c")
    print(registry, NAMES, first())
    SCALE = 1.0
    print(scaled(2.0))
