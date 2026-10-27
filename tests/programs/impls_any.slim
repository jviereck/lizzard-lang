# Impls for any type: builtin types, specialized and conditional impls of
# structs, and conditional inherent impls. Such an impl applies to the types
# its target matches whose type arguments satisfy its bounds; that is
# checked statically when the types are known, and at runtime otherwise.

trait Describe:
    def describe(self) -> str: ...
    def shout(self) -> str:                      # a default method
        self.describe().upper() + "!"

impl Describe for int:
    def describe(self) -> str:
        "the int " + str(self)

impl<T> Describe for list<T>:
    def describe(self) -> str:
        "a list of " + str(len(self))

struct Box<T>:
    value: T

impl Describe for Box<int>:                      # only `Box<int>`
    def describe(self) -> str:
        "a box of " + str(self.value)

impl<T: Ord> Ord for Box<T>:                     # conditional: `T: Ord`
    def __lt__(self, other: Self) -> bool:
        self.value < other.value

impl<T: Num> Box<T>:                             # conditional inherent impl
    def doubled(self) -> T:
        self.value + self.value

struct Plain:
    n: int

def show<T: Describe>(x: T) -> str:
    x.describe()

def untyped(x):
    x++

def values(bs: list<Box<int>>) -> list<int>:
    out: list<int> = []
    for b in bs:
        out.append(b.value)
    out


def main() -> ():
    print((5).describe(), [1, 2, 3].describe(), (7).shout())
    print(Describe::describe(8), show(9), show(["a"]))
    b = Box { value: 4 }
    f = Box { value: 1.5 }
    print(b.describe(), b.shout(), b.doubled(), f.doubled())
    print(values(sorted([Box { value: 3 }, Box { value: 1 }, Box { value: 2 }])))
    print(Box { value: 1 } < Box { value: 2 }, Box { value: "b" } < Box { value: "a" })
    p = Box { value: Plain { n: 1 } }
    print(isinstance(b, Ord), isinstance(p, Ord), isinstance(b, Describe), isinstance(f, Describe))
    print(isinstance(5, Describe), isinstance("five", Describe), isinstance([1.5], Describe))
    try:
        untyped(p) < untyped(p)
    except TypeError as e:
        print("TypeError:", e)
    try:
        show(untyped(f))
    except TypeError as e:
        print("TypeError:", e)
