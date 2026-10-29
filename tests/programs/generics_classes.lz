# Generic structs: type arguments are part of the instance's runtime type.

struct Stack<T>:
    items: list<T>

impl<T> Stack<T>:
    def new() -> Self:
        Stack { items: [] }
    def push(self, x: T + keep) -> ():
        self.items.append(x)
    def pop(self) -> Option<T>:
        self.items.pop()

impl<T> Len for Stack<T>:
    def __len__(self) -> int:
        len(self.items)

impl<T> Iterable<T> for Stack<T>:
    def __iter__(self) -> Iterator<T>:
        for x in self.items:
            yield x++

struct Box<T>:
    value: T

impl<T> Box<T>:
    def new(value: T + keep) -> Self:
        Box { value }
    def get(self) -> T:
        self.value++
    def swap(self, v: T + keep) -> T:
        old <- self.value <- v
        old

struct Pair<A, B>:
    first: A
    second: B

impl<A, B> Pair<A, B>:
    def new(a: A + keep, b: B + keep) -> Self:
        Pair { first: a, second: b }
    def flip(self) -> Pair<B, A>:
        Pair { first: self.second++, second: self.first++ }

def untyped(x: any) -> any:
    x++

def total(s: Stack<int>) -> int:
    n = 0
    for x in s:
        n += x
    n

def main() -> ():
    # Explicit type arguments.
    s = Stack::<int>::new()
    s.push(1)
    s.push(2)
    s.push(3)
    print(s, len(s), total(s))
    top = s.pop()
    if top is not None:
        print("popped", top + 10)

    # From the context.
    names: Stack<str> = Stack::new()
    names.push("ada")
    names.push("grace")
    for n in names:
        print(n.upper())

    # Inferred from the constructor arguments.
    b = Box::new(5)
    print(b, b.get() * 2)
    old = b.swap(7)
    print(old, b.value)
    p = Pair::new("x", 1.5)
    q = p.flip()
    print(p, q, q.first + 1.0, q.second.upper())

    # Nested generic types.
    nested = Stack::<list<int>>::new()
    nested.push([1, 2])
    nested.push([3])
    for xs in nested:
        print(xs, len(xs))
    boxes: list<Box<str>> = [Box::new("a"), Box { value: "b" }]
    print(boxes)

    # The runtime type is checked through untyped paths.
    try:
        untyped(s).push("four")
    except TypeError as e:
        print("error:", e)
    try:
        total(untyped(names))
    except TypeError as e:
        print("error:", e)
    try:
        bad: list<Box<int>> = untyped(boxes)
    except TypeError as e:
        print("error:", e)
    print(isinstance(b, Box), s)
