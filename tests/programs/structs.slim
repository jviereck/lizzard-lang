# Structs: literals (with field shorthand), associated functions (generic
# ones too), `Self`, methods across several `impl` blocks, `Drop`, and
# exceptions as structs.

struct Point:
    x: float
    y: float

impl Point:
    def new(x: float, y: float) -> Self:
        Point { x, y }
    def origin() -> Self:
        Self::new(0.0, 0.0)
    def shifted(self, dx: float) -> Self:
        Self { x: self.x + dx, y: self.y }

impl Point:                          # a second inherent impl
    def norm2(self) -> float:
        self.x * self.x + self.y * self.y

struct Pair<A, B>:
    first: A
    second: B

impl<A, B> Pair<A, B>:
    def of(a: A + keep, b: B + keep) -> Self:
        Pair { first: a, second: b }
    def flip(self) -> Pair<B, A>:
        Pair { first: self.second++, second: self.first++ }
    def with_first<C>(self, c: C + keep) -> Pair<C, B>:    # own type parameter
        Pair { first: c, second: self.second++ }

struct Noisy:
    name: str

impl Drop for Noisy:
    def __drop__(self) -> ():
        print("dropping", self.name)

struct Done

impl Exception for Done

struct AppError:
    msg: str
    code: int

impl Exception for AppError

def scope() -> ():
    n = Noisy { name: "a" }
    print("in scope", n.name)


def main() -> ():
    p = Point::new(3.0, 4.0)
    print(p.norm2(), p.shifted(1.0).x, Point::origin().norm2())
    q = Pair::of(1, "one")
    r = q.flip()
    print(r.first, r.second)
    s = Pair::<int, str>::of(2, "two").with_first([1.5])
    print(s.first, s.second)
    t: Pair<str, bool> = Pair::of("t", True)
    print(t.first, t.second)
    make = Point::new                    # an associated function as a value
    print(make(1.0, 1.0).norm2())
    scope()

    try:
        raise Done
    except Done:
        print("caught Done")
    try:
        raise AppError { msg: "bad thing", code: 7 }
    except AppError as e:
        print("caught", e.msg, e.code)
