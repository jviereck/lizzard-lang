# lizzard: module
# Small workloads for count.sh: one loop iteration, one method call, one
# fib(10) per unit.
def loop(n: int) -> int:
    t = 0
    for i in range(n):
        t += i * i
    return t


def fib(n: int) -> int:
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)


class Body:
    def __init__(self, x: float, v: float) -> None:
        self.x: float = x
        self.v: float = v

    def step(self, dt: float) -> None:
        self.x += self.v * dt


def bodies(n: int) -> None:
    b = Body(0.0, 1.0)
    for _ in range(n):
        b.step(0.01)


def fibs(n: int) -> None:
    for _ in range(n):
        fib(10)
