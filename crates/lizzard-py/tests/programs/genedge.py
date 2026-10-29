# lizzard: module
# Generator edge cases, driven from plain Python by test_lizzard.Generators.
import sys
from typing import Generator, Iterator


def simple() -> Generator[int, int, str]:
    x = yield 1
    y = yield x + 1
    return f"r{y}"


def leaks_stop() -> Iterator[int]:
    yield 1
    raise StopIteration("oops")


def stubborn() -> Iterator[int]:
    while True:
        try:
            yield 1
        except GeneratorExit:
            print("  ignoring GeneratorExit")


def closer() -> Generator[int, None, str]:
    try:
        yield 1
    except GeneratorExit:
        return "closed value"
    return "never"


def in_handler() -> Iterator[str]:
    try:
        raise KeyError("inner")
    except KeyError:
        yield f"handling {sys.exception()!r}"
        yield f"still {sys.exception()!r}"
    yield f"after {sys.exception()!r}"


def thrower() -> Iterator[int]:
    try:
        yield 1
    except ValueError as e:
        yield 100 + len(e.args)
    yield 2


def bad_finally() -> Iterator[int]:
    try:
        yield 1
    finally:
        raise RuntimeError("finally failed")


def deep() -> Iterator[int]:
    yield from simple_iter()


def simple_iter() -> Iterator[int]:
    yield 1
    raise ValueError("deep failure")


def self_ref() -> Iterator[int]:
    yield next(G)


G = self_ref()
