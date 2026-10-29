struct Box<T>:
    value: T

impl<T: Ord> Ord for Box<T>:
    def __lt__(self, other: Self) -> bool:
        self.value < other.value

struct Plain:
    n: int

def smallest<T: Ord>(xs: list<T>) -> T:
    sorted(xs)[0]


def main() -> ():
    smallest([Box { value: 2 }, Box { value: 1 }])
    smallest([Box { value: Plain { n: 1 } }])
