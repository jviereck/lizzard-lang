trait Log:
    def log(self, msg: str) -> (): ...
    def hello(self) -> ():
        self.log("hello from default")

struct Console:
    lines: int

impl Console:
    def new() -> Self:
        Console { lines: 0 }

impl Log for Console:
    def log(self, msg: str) -> ():
        self.lines += 1
        print("console:", msg)

struct Service:
    out: Console
    last: Option<str>
    count: int

impl Service:
    def new(out: Console + keep) -> Self:
        Service { out, last: None, count: 0 }

impl Log for Service:
    def log(self, msg: str) -> ():
        self.out.log(msg)
        old <- self.last <- Some(msg++)
        if old is not None:
            print("previous:", old)
        self.count += 1

def use_log(l: Log) -> ():
    l.log("via trait")


def main() -> ():
    s = Service::new(Console::new())
    s.log("one")
    s.log("two")
    s.hello()
    use_log(s)
    print(s.count, s.out.lines)
    label = s.last++.some_or("none")
    print(label)
    print(s)
