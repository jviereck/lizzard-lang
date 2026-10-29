# Trait-qualified methods: two traits may define the same method name for
# one type. Inherent methods win, `Trait::m(x)` picks a trait's method, a
# trait-typed or bounded receiver uses its trait's, and an ambiguous call on
# an untyped value raises at runtime.

trait Shape:
    def area(self) -> float: ...
    def summary(self) -> str:                 # `self.area()` is Shape's here
        "shape area " + str(self.area())

trait Land:
    def area(self) -> float: ...              # same name, other meaning (acres)
    def summary(self) -> str:
        "land area " + str(self.area())

trait Named:
    def name(self) -> str:
        "trait name"

struct Plot:
    side: float

impl Plot:
    def name(self) -> str:                    # inherent: wins over Named's
        "inherent name"
    def double(self) -> float:
        self.side * 2.0

impl Shape for Plot:
    def area(self) -> float:
        self.side * self.side

impl Land for Plot:
    def area(self) -> float:
        self.side * self.side / 4046.0

impl Named for Plot

def shape_area(s: Shape) -> float:            # a trait-typed receiver
    s.area()

def land_area<T: Land>(x: T) -> float:        # a bounded receiver
    x.area()

def untyped(x):
    x.area()


def main() -> ():
    p = Plot { side: 4046.0 }
    print(Shape::area(p), Land::area(p))
    print(Shape::summary(p))
    print(Land::summary(p))
    print(shape_area(p), land_area(p))
    print(p.name(), Named::name(p))
    print(Plot::double(p), Plot::name(p))        # `Type::m(x)`: a method with x as self
    print(p.double())
    try:
        untyped(p)
    except TypeError as e:
        print("TypeError:", e)
