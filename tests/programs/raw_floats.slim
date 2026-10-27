# Float arithmetic compiles to raw f64 temporaries; float locals reuse their
# box when no one else holds it; float fields are read and built unboxed.
struct P:
    x: float
    y: float

struct Box<T>:
    v: T

impl P:
    def norm2(self) -> float:
        self.x * self.x + self.y * self.y
    def scaled(self, k: float) -> P:
        P { x: self.x * k, y: self.y * k }

def f(c: bool, a: float, b: float) -> float:
    x = 1.0
    x = a if c else b * 2.0
    x

def main() -> ():
    x = 1.5
    keep = x++
    x = x * 2.0 + 1.0
    print(x, keep)
    p = P { x: 1.0, y: 2.0 }
    q = P { x: 3.0, y: 4.0 }
    print(p.x * q.x + p.y * q.y, p.x / q.y, 2 * p.y + 1, p.x - 3)
    print(f(True, 1.0, 5.0), f(False, 1.0, 5.0))
    xs: list<float> = [x]
    x = x + 0.5
    print(xs, x)
    try:
        y = p.x / (q.x - 3.0)
        print(y)
    except ZeroDivisionError as e:
        print("zde", e)
    z = 0.0
    for i in range(5):
        z = z + i * 0.5
    print(z)
    z += 1.0
    z -= 0.25 * 2
    print(z)
    b = Box { v: 1.5 }
    b2 = Box { v: b.v * 2.0 }
    print(b2.v + 1.0)
    s = p.scaled(3.0)
    print(s.x, s.y, s.norm2())
    acc = 0.0
    alias = acc++
    for i in range(3):
        acc = acc + s.norm2()
    print(acc, alias)
    n = 3
    print(n / 2 + 0.5, 7 * 0.5, n * 1.0 - 1)
