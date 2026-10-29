def fib(n: int) -> int:
    if n < 2:
        return n
    fib(n - 1) + fib(n - 2)

def classify(n: int) -> str:
    if n > 100:
        return "huge"
    size = if n > 10:
        "big"
    else:
        "small"
    size


def main() -> ():
    print(fib(20))
    print(classify(5), classify(50), classify(500))
    total = 0
    for i in range(10):
        total += i
    print("total", total)
    s = "hello"
    t = s++
    print(s, t)
    u = s
    print(u + " world")
    x = 3.5 * 2
    print(x, 7 / 2, 7 // 2, -7 % 3, 2 ** 10)
