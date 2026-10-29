# Channels connect threads, so their items must be `Send`.
struct Bag:
    item: any

impl Bag:
    def new(item: any + keep) -> Self:
        Bag { item }

def main() -> ():
    ch: Channel<Bag> = Channel()
