# Static element types: list<T>, dict<K, V> and the items of for loops.
struct Item:
    name: str

impl Item:
    def new(name: str + keep) -> Self:
        Item { name }
    def shout(self) -> str:
        self.name.upper()

def names(items: list<Item>) -> list<str>:
    out: list<str> = []
    for it in items:
        out.append(it.shout())
    out

def main() -> ():
    items: list<Item> = [Item::new("a" + ""), Item { name: "b" + "" }]
    print(names(items))
    first = items.first()
    if first is not None:
        print(first.name)
    counts: dict<str, int> = {}
    counts["x"] = 1
    for k in counts:
        print(k, counts[k] + 1)
