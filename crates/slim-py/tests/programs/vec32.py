# slim: module
import numpy as np

type Vec32 = np.ndarray[tuple[int], np.dtype[np.float32]]


def add_loop(a: Vec32, b: Vec32) -> Vec32:
    if a.shape != b.shape:
        raise ValueError("expected two arrays of the same length")
    out = np.empty_like(a)
    for i in range(len(a)):
        out[i] = a[i] + b[i]
    return out


def scale(a: Vec32, k: float) -> Vec32:
    out = np.empty_like(a)
    for i in range(len(a)):
        out[i] = a[i] * k
    return out


def get(a: Vec32, i: int) -> np.float32:
    return a[i]


if __name__ == "__main__":
    x = np.arange(5, dtype=np.float32)
    y = np.full(5, 0.5, dtype=np.float32)
    print(add_loop(x, y))
    print(get(x, 3), get(x, -1))
