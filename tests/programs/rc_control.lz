struct Res:
    name: str

impl Res:
    def new(n: str + keep) -> Self:
        Res { name: n }

impl Drop for Res:
    def __drop__(self) -> ():
        print("drop", self.name)

def scope() -> ():
    a = Res::new("a")
    b = Res::new("b")
    c = Res::new("c")


def main() -> ():
    rc_auto(False)

    scope()
    print("after scope, pending =", rc_pending())
    print("freed (limit 1):", rc_run(1))
    print("freed rest:", rc_run())

    holder = [Res::new("in-list")]
    holder.pop()
    print("popped, nothing freed yet")
    rc_run()

    r = Res::new("shared")
    r2 = r++
    r = Res::new("replacement")
    rc_run()
    print("shared still alive via r2:", r2.name)
