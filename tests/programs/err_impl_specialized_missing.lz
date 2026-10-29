trait D:
    def d(self) -> str: ...

struct Box<T>:
    value: T

impl D for Box<int>:
    def d(self) -> str:
        "int box"


def main() -> ():
    Box { value: "s" }.d()
