trait Shape:
    def area(self) -> float: ...

trait Land:
    def acres(self) -> float: ...

struct Plot:
    side: float

impl Shape for Plot:
    def area(self) -> float:
        self.side


def main() -> ():
    p = Plot { side: 1.0 }
    print(Land::acres(p))
