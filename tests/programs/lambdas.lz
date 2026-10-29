def apply(f: function, x: int) -> int:
    f(x)

def make_counter(start: int) -> function + Send:
    prefix = "count"
    |step: int| {
        print(prefix, start + step)
        start + step
    }

# captured lists move into the closure
def summer() -> function:
    data = [1, 2, 3]
    || sum(data)


def main() -> ():
    double = |x: int| x * 2
    print(apply(double, 21))

    c = make_counter(10)
    c(1)
    c(5)

    names = ["ann", "bob"]
    greet = |n: str| "hi " + n
    for n in names:
        print(greet(n))

    print(summer()())
