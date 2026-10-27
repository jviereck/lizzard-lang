# Out-of-bounds, negative and ill-typed array accesses raise; nothing is
# written. The values are untyped so the checks happen at runtime.
def attempt(f: function, a: array) -> ():
    try:
        f(a)
    except IndexError as e:
        print("IndexError:", e.msg)
    except OverflowError as e:
        print("OverflowError:", e.msg)
    except TypeError as e:
        print("TypeError:", e.msg)
    except ValueError as e:
        print("ValueError:", e.msg)

def store(a: array, i: int, v: any) -> ():
    a[i] = v


def main() -> ():
    a = array(u8, 3)
    attempt(|a| a[3], a)
    attempt(|a| a[-1], a)
    attempt(|a| store(a, 0, 256), a)
    attempt(|a| store(a, 0, -1), a)
    attempt(|a| store(a, 0, 1.5), a)
    attempt(|a| store(a, 5, 1), a)
    attempt(|a| a.view(2, 2), a)
    attempt(|a| array(f64, 2).copy_from(array(f32, 2)), a)
    attempt(|a| array(str, 2), a)
    print(a)
