# Bounds are checked at runtime when type arguments are inferred from `any`.
def smallest<T: Ord>(xs: list<T>) -> Option<T>:
    sorted(xs).first()

def untyped(x: any) -> any:
    x++


def main() -> ():
    print(smallest(untyped([3, 1, 2])))
    print(smallest(untyped([{"a": 1}])))
