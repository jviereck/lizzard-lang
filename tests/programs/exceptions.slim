struct AppError:
    msg: str
    code: int

impl Exception for AppError

def fail(code: int) -> int:
    raise AppError { msg: "app failed", code }

# finally with return
def f() -> str:
    try:
        return "from try"
    finally:
        print("cleanup")

# nested handlers and re-raise
def g() -> ():
    try:
        try:
            x = 1 // 0
        except ZeroDivisionError:
            print("inner caught, re-raising")
            raise
    except ArithmeticError as e:
        print("outer caught:", repr(e))


def main() -> ():
    try:
        fail(7)
    except AppError as e:
        print("AppError", e.code, e.msg)

    # exception groups are traits
    for k in ["a", "z"]:
        try:
            d = {"a": 1}
            print(d[k])
        except LookupError as e:
            print("lookup failed:", e)

    print(f())

    g()

    # try as an expression, else clause
    v = try:
        int("42")
    except ValueError:
        0
    else:
        print("no error")
        100
    print(v)

    # raise from
    try:
        try:
            int("x")
        except ValueError as e:
            raise RuntimeError("bad config") from e
    except RuntimeError as e:
        print("RuntimeError:", e)

    # except tuple and bare except
    for i in range(3):
        try:
            if i == 0:
                raise KeyError("k")
            elif i == 1:
                raise TypeError("t")
            else:
                raise IndexError("i")
        except (KeyError, TypeError) as e:
            print("tuple handler:", repr(e))
        except:
            print("bare handler")
