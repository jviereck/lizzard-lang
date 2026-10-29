# A plain Python module (not Lizzard): generators and drivers for generators.py.
def countdown(n):
    while n > 0:
        yield n
        n -= 1
    return "liftoff"


def echo():
    received = []
    try:
        while True:
            x = yield len(received)
            received.append(x)
    except GeneratorExit:
        print("python echo closed with", received)
        raise


def drive(gen):
    out = [next(gen)]
    out.append(gen.send(10))
    out.append(gen.send(20))
    try:
        gen.throw(ValueError("stop it"))
    except StopIteration as e:
        out.append(("stopped", e.value))
    except ValueError as e:
        out.append(("raised", str(e)))
    return out


def delegate(gen):
    result = yield from gen
    return ("python delegate got", result)
