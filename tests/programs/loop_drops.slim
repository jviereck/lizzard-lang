# A local assigned in a loop body drops the previous iteration's value.
struct Noisy:
    name: str

impl Noisy:
    def new(name: str + keep) -> Self:
        Noisy { name: name }

impl Drop for Noisy:
    def __drop__(self) -> ():
        print("drop", self.name)

def run() -> ():
    rc_auto(False)
    i = 0
    while i < 3:
        x = Noisy::new("w" + str(i))
        i += 1
    print("after while, freed", rc_run())
    for s in ["a" + "1", "b" + "2", "c" + "3"]:
        n = Noisy::new(s)
    print("after for, freed", rc_run())
    y = Noisy::new("moved")
    for k in range(2):
        keep_it = y
        y = Noisy::new("again " + str(k))
    print("after range, freed", rc_run())


def main() -> ():
    run()
    print("end, freed", rc_run())
