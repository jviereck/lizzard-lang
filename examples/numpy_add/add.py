# lizzard: module
import numpy as np
import numpy.typing as npt


def add(a: npt.NDArray[np.float32], b: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    # A normal call into NumPy: the overlay resolves float32 + float32 to NDArray[float32],
    # and lizzard checks the result's type and dtype when the call returns.
    return a + b


def add_loop(a: npt.NDArray[np.float32], b: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    # The same thing as a compiled loop, for 1-D arrays.
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("expected two 1-D arrays of the same length")
    out = np.empty_like(a)                 # NDArray[float32]
    for i in range(len(a)):
        out[i] = a[i] + b[i]               # unknown ndim: a[i] is Any, so this goes through NumPy
    return out


if __name__ == "__main__":
    x = np.arange(5, dtype=np.float32)
    y = np.full(5, 0.5, dtype=np.float32)
    print(add(x, y))                       # [0.5 1.5 2.5 3.5 4.5]
    print(add_loop(x, y))                  # [0.5 1.5 2.5 3.5 4.5]
    add(x, x.astype(np.float64))           # lizzard: static error, b must be float32
