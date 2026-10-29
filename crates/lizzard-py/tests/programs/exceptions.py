# lizzard: module
# Exceptions: try/except/else/finally with return, break and continue, raise
# from, bare raise, __context__ chaining, sys.exc_info(), exception classes,
# and with. Its output must match plain CPython's.
import sys


class AppError(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"code {code}")
        self.code: int = code


class NotFound(AppError):
    pass


def parse(s: str) -> int:
    try:
        return int(s)
    except ValueError:
        return -1


def lookup(d: dict[str, int], k: str) -> int:
    try:
        v = d[k]
    except KeyError as e:
        print("missing", e, type(e).__name__)
        raise NotFound(404) from e
    else:
        print("found", k)
        return v
    finally:
        print("lookup done", k)


def classify(x: int) -> str:
    try:
        if x < 0:
            raise ValueError("negative")
        if x == 0:
            raise ZeroDivisionError("zero")
        if x > 100:
            raise AppError(x)
        return "ok"
    except (ValueError, ZeroDivisionError) as e:
        return "bad: " + str(e)
    except AppError as e:
        return f"app {e.code}"


def loop_with_finally() -> list[str]:
    log: list[str] = []
    for i in range(5):
        try:
            if i == 1:
                continue
            if i == 3:
                break
            log.append(f"body {i}")
        finally:
            log.append(f"finally {i}")
    return log


def return_in_finally_order() -> int:
    n = 0
    try:
        n = 1
        return n
    finally:
        print("finally sees", n)


def context_chain() -> None:
    try:
        try:
            d: dict[str, int] = {}
            d["x"]
        except KeyError:
            raise RuntimeError("while handling")
    except RuntimeError as e:
        print(type(e.__context__).__name__, e.__cause__, e.__suppress_context__)
    try:
        try:
            raise ValueError("inner")
        except ValueError as e:
            raise AppError(1) from None
    except AppError as e:
        print(e.__cause__, e.__suppress_context__, type(e.__context__).__name__)


def reraise() -> None:
    try:
        try:
            raise NotFound(7)
        except AppError:
            print("handled, exc_info:", type(sys.exc_info()[1]).__name__)
            raise
    except NotFound as e:
        print("reraised", e.code)
    print("after:", sys.exc_info()[1])


class Resource:
    def __init__(self, name: str, swallow: bool) -> None:
        self.name: str = name
        self.swallow: bool = swallow

    def __enter__(self) -> Resource:
        print("enter", self.name)
        return self

    def __exit__(self, t: object, e: object, tb: object) -> bool:
        print("exit", self.name, t is not None)
        return self.swallow


def use_resources() -> int:
    with Resource("a", False) as a, Resource("b", True) as b:
        print("using", a.name, b.name)
        raise ValueError("swallowed by b")
    with Resource("c", False):
        print("in c")
    try:
        with Resource("d", False):
            raise KeyError("d")
    except KeyError:
        print("propagated from d")
    for i in range(3):
        with Resource(f"loop{i}", False):
            if i == 1:
                break
    with Resource("ret", False):
        return 42


def main() -> None:
    print(parse("12"), parse("x"))
    print(lookup({"a": 1}, "a"))
    try:
        lookup({}, "b")
    except NotFound as e:
        print("caught", e, e.code, type(e.__cause__).__name__)
    print(classify(-1), classify(0), classify(500), classify(5))
    print(loop_with_finally())
    print(return_in_finally_order())
    context_chain()
    reraise()
    print(use_resources())


def ret_in_except() -> str:
    for i in range(3):
        try:
            if i < 2:
                raise ValueError(str(i))
            return "no error"
        except ValueError as e:
            if i == 0:
                continue
            return "returned from except " + str(e)
        finally:
            print("finally", i, sys.exc_info()[0])
    return "end"


def error_in_finally() -> None:
    try:
        try:
            raise KeyError("first")
        finally:
            raise RuntimeError("from finally")
    except RuntimeError as e:
        print("got", e, "context:", repr(e.__context__))


def error_in_handler() -> None:
    try:
        try:
            raise ValueError("a")
        except ValueError:
            x: list[int] = []
            print(x[3])
    except IndexError as e:
        print("handler error", e, "context:", repr(e.__context__))
    print("exc_info clear:", sys.exc_info() == (None, None, None))


def nested_finally_return() -> int:
    try:
        try:
            return 1
        finally:
            print("inner finally")
    finally:
        print("outer finally")


def bad(n: int) -> int:
    if n > 2:
        raise ValueError(f"too big: {n}")
    return n


def main2() -> None:
    print(ret_in_except())
    error_in_finally()
    error_in_handler()
    print(nested_finally_return())
    total = 0
    for n in range(5):
        try:
            total += bad(n)
        except ValueError as e:
            print("skip", e)
    print(total)
    e = "still a str"
    print(e)


if __name__ == "__main__":
    main()
    main2()
