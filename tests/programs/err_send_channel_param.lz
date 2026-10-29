# `Channel<T>` needs `T: Send`.
struct Pool<T>:
    jobs: Channel<T>

impl<T> Pool<T>:
    def new() -> Self:
        Pool { jobs: Channel() }


def main() -> ():
    p = Pool::<int>::new()
