# Storing a statically wrong type into a typed list is a compile error.
def f(l: list<str>) -> ():
    l.append(3)


def main() -> ():
    pass
