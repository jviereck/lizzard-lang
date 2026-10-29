# The `+local` modifier of earlier versions is gone.
struct Counter:
    n: int


def main() -> ():
    c: Counter<+local> = Counter { n: 0 }
