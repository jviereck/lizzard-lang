# lizzard: module
# Threads: a Lizzard class deriving from threading.Thread, locks, events,
# conditions, semaphores and barriers, all checked against Lizzard's
# definitions of the threading module. Its output must match plain CPython's.
import threading


class Worker(threading.Thread):
    def __init__(self, n: int, lock: threading.Lock, results: list[int]) -> None:
        super().__init__(name=f"worker-{n}")
        self.n: int = n
        self.lock: threading.Lock = lock
        self.results: list[int] = results
        self.total: int = 0

    def run(self) -> None:
        t = 0
        for i in range(self.n):
            t += i * i
        self.total = t
        with self.lock:
            self.results.append(t)


def squares(n: int) -> int:
    return sum([i * i for i in range(n)])


def workers() -> None:
    lock = threading.Lock()
    results: list[int] = []
    ws = [Worker(1000 * (k + 1), lock, results) for k in range(4)]
    for w in ws:
        w.daemon = True
        w.start()
    for w in ws:
        w.join()
    for w in ws:
        print(w.name, w.daemon, w.is_alive(), w.total == squares(w.n), w.ident is not None)
    print(sorted(results) == [w.total for w in ws])
    print(lock.locked(), lock.acquire(), lock.locked())
    lock.release()


def counter(lock: threading.Lock, box: list[int], n: int) -> None:
    for _ in range(n):
        with lock:
            box[0] += 1


def targets() -> None:
    lock = threading.Lock()
    box = [0]
    ts = [threading.Thread(target=counter, args=(lock, box, 1000)) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=10.0)
    print("count", box[0])


def handshake() -> None:
    ready = threading.Event()
    cond = threading.Condition()
    items: list[str] = []

    def producer() -> None:
        ready.wait()
        with cond:
            items.append("item")
            cond.notify_all()

    t = threading.Thread(target=producer)
    t.start()
    print("set before", ready.is_set())
    ready.set()
    with cond:
        print("waited", cond.wait_for(lambda: len(items) > 0, timeout=10.0))
    t.join()
    print(items)


def gates() -> None:
    sem = threading.BoundedSemaphore(2)
    print(sem.acquire(), sem.acquire(), sem.acquire(blocking=False))
    sem.release()
    try:
        sem.release(2)
    except ValueError as e:
        print("ValueError", e)
    barrier = threading.Barrier(3)
    order: list[int] = []
    rl = threading.RLock()

    def arrive(i: int) -> None:
        barrier.wait(timeout=10.0)
        with rl:
            with rl:
                order.append(i)

    ts = [threading.Thread(target=arrive, args=(i,)) for i in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print(barrier.parties, barrier.n_waiting, barrier.broken, sorted(order))
    barrier.abort()
    try:
        barrier.wait()
    except threading.BrokenBarrierError:
        print("broken", barrier.broken)


if __name__ == "__main__":
    workers()
    targets()
    handshake()
    gates()
    print(threading.current_thread() is threading.main_thread(), threading.active_count())
