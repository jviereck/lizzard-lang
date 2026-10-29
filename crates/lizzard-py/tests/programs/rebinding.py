# lizzard: module
# Globals and builtins read by compiled code are cached per site; any change
# to the module's globals or the builtins (rebinding, shadowing a builtin,
# deleting the shadow, threads rebinding concurrently) must be seen at once.
# Its output must match plain CPython's.
import threading

import rebinding_patch

SCALE: int = 2


def scaled(x: int) -> int:
    return x * SCALE


def magnitude(x: int) -> int:
    return abs(x)


def rebind(v: int) -> None:
    global SCALE
    SCALE = v


def worker(out: list[int], n: int) -> None:
    for _ in range(n):
        out.append(scaled(1))


if __name__ == "__main__":
    print(scaled(5), magnitude(-3))
    rebind(10)
    print(scaled(5))
    SCALE = 3
    print(scaled(5))
    # A global named like a builtin shadows it; deleting it unshadows.
    rebinding_patch.shadow_abs()
    print(magnitude(-3))
    rebinding_patch.unshadow_abs()
    print(magnitude(-3))
    rebinding_patch.patch_builtin_abs()
    print(magnitude(-3))
    rebinding_patch.restore_builtin_abs()
    print(magnitude(-3))
    # Threads reading a global while it changes see old or new values only.
    out: list[int] = []
    ts = [threading.Thread(target=worker, args=(out, 2000)) for _ in range(4)]
    for t in ts:
        t.start()
    for v in range(4, 50):
        rebind(v)
    for t in ts:
        t.join()
    print(len(out), all([2 <= x <= 49 for x in out]))
    rebind(7)
    print(scaled(1))
