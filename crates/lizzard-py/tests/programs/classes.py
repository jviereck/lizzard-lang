# lizzard: module
# Classes and subclasses: attributes, methods, properties, class and static
# methods, dunders, super(), multiple inheritance, generic classes and
# exceptions. Its output must match plain CPython's.


class Shape:
    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    def area(self) -> float:
        return 0.0

    def describe(self) -> str:
        return f"{self.name} with area {self.area():.2f}"


class Circle(Shape):
    def __init__(self, r: float) -> None:
        super().__init__("circle")
        self.r: float = r

    def area(self) -> float:
        return 3.14159 * self.r * self.r


class Rect(Shape):
    def __init__(self, w: float, h: float) -> None:
        super().__init__("rect")
        self.w: float = w
        self.h: float = h

    def area(self) -> float:
        return self.w * self.h

    @staticmethod
    def square(side: float) -> Rect:
        return Rect(side, side)


class Vec:
    x: float
    y: float

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def __add__(self, other: Vec) -> Vec:
        return Vec(self.x + other.x, self.y + other.y)

    def __mul__(self, k: float) -> Vec:
        return Vec(self.x * k, self.y * k)

    def __neg__(self) -> Vec:
        return Vec(-self.x, -self.y)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Vec):
            return self.x == other.x and self.y == other.y
        return False

    def __repr__(self) -> str:
        return f"Vec({self.x}, {self.y})"

    @property
    def norm2(self) -> float:
        return self.x * self.x + self.y * self.y

    @classmethod
    def zero(cls) -> Vec:
        return cls(0.0, 0.0)


class Counter:
    instances: int = 0

    def __init__(self, start: int = 0, step: int = 1) -> None:
        self.value: int = start
        self.step: int = step
        Counter.instances += 1

    def tick(self) -> int:
        self.value += self.step
        return self.value


class Stack[T]:
    items: list[T]

    def __init__(self) -> None:
        self.items = []

    def push(self, x: T) -> None:
        self.items.append(x)

    def pop(self) -> T:
        return self.items.pop()

    def peek(self) -> T:
        return self.items[-1]

    def __len__(self) -> int:
        return len(self.items)

    def __bool__(self) -> bool:
        return len(self.items) > 0


class Pair[K, V]:
    def __init__(self, k: K, v: V) -> None:
        self.k: K = k
        self.v: V = v

    def flip(self) -> Pair[V, K]:
        return Pair(self.v, self.k)

    def as_tuple(self) -> tuple[K, V]:
        return (self.k, self.v)


class Node:
    def __init__(self, value: int) -> None:
        self.value: int = value
        self.children: list[Node] = []

    def add(self, child: Node) -> Node:
        self.children.append(child)
        return child

    def total(self) -> int:
        t = self.value
        for c in self.children:
            t += c.total()
        return t

    def depth(self) -> int:
        best = 0
        for c in self.children:
            best = max(best, c.depth())
        return best + 1

    def __str__(self) -> str:
        if not self.children:
            return str(self.value)
        inner = " ".join([str(c) for c in self.children])
        return f"({self.value} {inner})"


class Animal:
    def __init__(self, name: str) -> None:
        self.name: str = name

    def sound(self) -> str:
        raise NotImplementedError

    def speak(self) -> str:
        return f"{self.name} says {self.sound()}"


class Dog(Animal):
    def sound(self) -> str:
        return "woof"


class Puppy(Dog):
    def sound(self) -> str:
        return super().sound() + "!"

    def speak(self) -> str:
        return "small " + super().speak()


class Swimmer:
    def move(self) -> str:
        return "swim"

    def dive(self) -> str:
        return "dive"


class Walker:
    def move(self) -> str:
        return "walk"


class Duck(Walker, Swimmer):
    pass


class InsufficientFunds(Exception):
    def __init__(self, needed: float) -> None:
        super().__init__(f"need {needed:.2f} more")
        self.needed: float = needed


class Account:
    def __init__(self, owner: str, balance: float = 0.0) -> None:
        self.owner: str = owner
        self.balance: float = balance
        self.history: list[tuple[str, float]] = []

    def deposit(self, amount: float) -> None:
        self.balance += amount
        self.history.append(("deposit", amount))

    def withdraw(self, amount: float) -> None:
        if amount > self.balance:
            raise InsufficientFunds(amount - self.balance)
        self.balance -= amount
        self.history.append(("withdraw", amount))


class Savings(Account):
    def __init__(self, owner: str, rate: float) -> None:
        super().__init__(owner, 100.0)
        self.rate: float = rate

    def add_interest(self) -> float:
        interest = self.balance * self.rate
        self.deposit(interest)
        return interest


def total_area(shapes: list[Shape]) -> float:
    t = 0.0
    for s in shapes:
        t += s.area()
    return t


def shapes_demo() -> None:
    shapes: list[Shape] = [Circle(1.0), Rect(2.0, 3.0), Rect.square(2.0)]
    for s in shapes:
        print(s.describe())
        if isinstance(s, Rect):
            print("  sides", s.w, s.h)
    print(f"total {total_area(shapes):.3f}")


def vec_demo() -> None:
    a = Vec(1.0, 2.0)
    b = Vec(3.0, 4.0)
    c = a + b * 2.0
    print(c, -c, c.norm2, c == Vec(7.0, 10.0), c == a, Vec.zero())


def counter_demo() -> None:
    c = Counter()
    d = Counter(10, step=5)
    for _ in range(3):
        c.tick()
        d.tick()
    print(c.value, d.value, Counter.instances)


def stack_demo() -> None:
    s: Stack[int] = Stack()
    for i in range(5):
        s.push(i * i)
    print(len(s), s.peek(), bool(s))
    out: list[int] = []
    while s:
        x = s.pop()
        if x == 4:
            continue
        out.append(x)
    print(out, bool(s))
    words = Stack[str]()
    words.push("hi")
    print(words.pop().upper())
    p = Pair("a", 1)
    q = p.flip()
    n: int = q.k
    print(q.as_tuple(), n + 1)
    k, v = q.as_tuple()
    print(k * 2, v.upper())


def tree_demo() -> None:
    root = Node(1)
    a = root.add(Node(2))
    a.add(Node(4))
    root.add(Node(3)).add(Node(5)).add(Node(6))
    print(root, root.total(), root.depth())


def animals_demo() -> None:
    pets: list[Animal] = [Dog("rex"), Puppy("bit")]
    for p in pets:
        print(p.speak())
    duck = Duck()
    print(duck.move(), duck.dive(), [c.__name__ for c in Duck.__mro__])


def bank_demo() -> None:
    acct = Savings("ann", 0.5)
    acct.withdraw(30.0)
    print(acct.add_interest(), acct.balance)
    print([kind for kind, amount in acct.history if amount > 10.0])
    e = InsufficientFunds(2.5)
    print(e, e.needed, isinstance(e, Exception))


if __name__ == "__main__":
    shapes_demo()
    vec_demo()
    counter_demo()
    stack_demo()
    tree_demo()
    animals_demo()
    bank_demo()
