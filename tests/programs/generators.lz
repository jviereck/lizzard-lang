def count_up(n: int):
    i = 0
    while i < n:
        yield i
        i += 1
    return "done"

def echo():
    total = 0
    while True:
        v = yield total
        if v is None:
            break
        total += v

def chain():
    yield from count_up(2)
    yield from ["a", "b"]

def guarded():
    try:
        yield 1
        yield 2
    finally:
        print("generator cleanup")

def catcher():
    while True:
        try:
            yield "waiting"
        except ValueError as e:
            print("generator got", repr(e))

def squares(n: int):
    for i in range(n):
        yield i * i


def main() -> ():
    for x in count_up(3):
        print("for", x)

    e = echo()
    print(next(e))
    print(e.send(5))
    print(e.send(10))

    print(list(chain()))

    gg = guarded()
    print(next(gg))
    gg.close()

    c = catcher()
    print(next(c))
    print(c.throw(ValueError("boom")))

    print(sum(squares(5)))
