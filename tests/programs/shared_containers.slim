# Eight threads hammer one list and one dict: pushes, pops, element stores,
# borrowed reads, inserts and deletes, with rc_run on every thread. Races may
# reorder or lose nothing here (every write is serialized), and no read may
# see freed memory.
def worker(l: list<str> + keep, d: dict<str, str> + keep, id: int, n: int) -> int:
    popped = 0
    for i in range(n):
        l.append("w" + str(id) + "-" + str(i))
        if len(l) > 0:
            first = l.first()
            if first is not None:
                assert len(first) > 0
        x = l.pop()
        if x is not None:
            popped += 1
        if len(l) > 2:
            l[1] <- "s" + str(i)
        key = "k" + str(i % 16)
        d[key] <- "v" + str(id)
        v = d.get(key)
        if v is not None:
            assert len(v) > 0
        if i % 3 == 0:
            d.pop(key, None)
        if i % 50 == 0:
            rc_run()
    popped

def main() -> ():
    l = ["seed" + "0", "seed" + "1", "seed" + "2"]
    d: dict<str, str> = {}
    ts: list<Thread> = []
    for id in range(8):
        ts.append(spawn(worker, l++, d++, id, 2000))
    total = 0
    for t in ts:
        total += t.join()
    print("pushed", 8 * 2000, "popped", total, "left", len(l))
    print("consistent:", 3 + 8 * 2000 - total == len(l))
    print("dict keys <= 16:", len(d) <= 16)
