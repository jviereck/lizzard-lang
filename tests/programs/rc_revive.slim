# A borrow of a list element outlives the list's ownership of it: the callee
# pops the element and drops it (count 0, queued), then revives the borrow
# with `++`. rc_run must not free the revived object.
struct Noisy:
    name: str

impl Noisy:
    def new(name: str + keep) -> Self:
        Noisy { name: name }

impl Drop for Noisy:
    def __drop__(self) -> ():
        print("drop", self.name)

def take(l: list, x: Noisy) -> Noisy:
    old = l.pop()
    del old
    print("freed while pinned:", rc_run())
    y = x++
    print("freed after revive:", rc_run())
    y

def main() -> ():
    rc_auto(False)
    l = [Noisy::new("a" + "b")]
    r = take(l, l[0])
    print("freed after call:", rc_run())
    print("still alive:", r.name)
    del r
    print("freed after del:", rc_run())
