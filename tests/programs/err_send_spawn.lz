# A closure capturing a value that is not `Send` is a `function`, not
# a `function + Send`, so it cannot run on another thread.
struct Bag:
    item: any

impl Bag:
    def new(item: any + keep) -> Self:
        Bag { item }

def main() -> ():
    b = Bag::new(1)
    work = || str(b.item)
    print(work())
    spawn(work).join()
