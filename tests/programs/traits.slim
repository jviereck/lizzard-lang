# Traits: definitions with required and default methods, `Self`, generic
# traits, implementations in `impl Trait for Type:` blocks, and their use in
# annotations, lists and bounds. Items are order-independent: `Square`
# implements `Shape` before the trait is defined below.

struct Square:
    side: float

impl Square:
    def new(side: float) -> Self:
        Square { side }

impl Shape for Square:                  # `describe` is inherited
    def area(self) -> float:
        self.side * self.side
    def scale(self, k: float) -> Self:
        Square { side: self.side * k }

impl Ord for Square:
    def __lt__(self, other: Square) -> bool:
        self.side < other.side

struct Rect:
    w: float
    h: float

impl Shape for Rect:
    def area(self) -> float:
        self.w * self.h
    def scale(self, k: float) -> Self:
        Rect { w: self.w * k, h: self.h * k }
    def describe(self) -> str:          # overrides the default
        "rect " + str(self.w) + "x" + str(self.h)

trait Shape:
    def area(self) -> float: ...                  # required
    def scale(self, k: float) -> Self: ...        # `Self`: the implementing type
    def describe(self) -> str:                    # default
        "shape of area " + str(self.area())
    def twice(self) -> Self:                      # a default returning `Self`
        self.scale(2.0)

trait Source<T>:
    def next_item(self) -> Option<T>: ...

struct Repeat<T>:
    value: T

impl<T> Source<T> for Repeat<T>:
    def next_item(self) -> Option<T>:
        Some(self.value++)

struct Countdown:
    n: int

impl Source<int> for Countdown:
    def next_item(self) -> Option<int>:
        if self.n == 0:
            return None
        self.n -= 1
        Some(self.n + 1)

def total(xs: list<Shape>) -> float:               # any `Shape`
    t = 0.0
    for x in xs:
        t = t + x.area()
    t

def biggest<T: Shape + Ord>(xs: list<T>) -> float:  # bounds
    sorted(xs)[len(xs) - 1].area()

def pull<T>(s: Source<T>) -> Option<T>:            # a generic trait type
    s.next_item()

def drain(s: Source<int>) -> list<int>:
    out: list<int> = []
    while True:
        x = s.next_item()
        if x is None:
            break
        out.append(x)
    out


def main() -> ():
    shapes: list<Shape + Send> = [Square::new(2.0), Rect { w: 1.0, h: 3.0 }]
    print(total([Square::new(1.0), Rect { w: 2.0, h: 2.0 }]))
    for s in shapes:
        print(s.describe(), s.twice().area())
    print(biggest([Square::new(3.0), Square::new(1.0), Square::new(2.0)]))
    print(pull(Repeat { value: "hi" }).some_or("none"))
    print(drain(Countdown { n: 3 }))
    print(isinstance(shapes[0], Shape), isinstance(shapes[0], Source))
