# Inline small strings (up to 30 bytes), dict hashing and probing, builtin
# method caches and fused loop instructions.

def main() -> ():
    a = "x" * 29
    b = a + "y"
    c = b + "z"
    print(len(a), len(b), len(c), b == a + "y", c > b, c[30])
    u = "é" * 15
    print(len(u), u + "!", (u + "é")[15])
    print(str(9223372036854775807), str(-9223372036854775807 - 1), str(0), str(-42))
    k: dict<str, int> = {}
    for i in range(100):
        k["key" + str(i)] = i
    print(k["key7"], k["key99"], len(k), "key100" in k)
    d: dict<int, int> = {}
    for i in range(1000):
        d[i * 1048576] = i
        d[-i - 1] = -i
    t = 0
    for i in range(1000):
        t += d[i * 1048576] + d[-i - 1]
    print(t, len(d))
    f: dict<float, str> = {1.0: "one", 2.5: "two and a half"}
    print(f[1.0], f[2.5])
    print(d.pop(0), d.get(0), len(d))
    d[0] = 5
    print(d[0], len(d))
    xs: list<int> = []
    for i in range(10):
        xs.append(i * i)
    s = 0
    for x in xs:
        s += x
        if x > 40:
            break
    print(s, x, xs)
    ys = ["a", "b"]
    for y in ys:
        pass
    print(y)
