# A value that is not `Send` cannot be stored as `any + Send`.
struct Bag:
    item: any

impl Bag:
    def new(item: any + keep) -> Self:
        Bag { item }

def main() -> ():
    mine = Bag::new(1)
    shared: dict<str, any + Send> = {}
    shared["b"] <- mine
