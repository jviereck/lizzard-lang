# A small package's own stubs (the base layer).
import sys
from typing import Any, Generic, TypeVar, TypeAlias

from ._bits import B32, B64

_T_co = TypeVar("_T_co", covariant=True)
_T = TypeVar("_T")
_N = TypeVar("_N")

class scalar(Generic[_N]):
    def value(self) -> float: ...

f32: TypeAlias = scalar[B32]
class f64(scalar[B64], float): ...

class Box(Generic[_T_co]):
    def get(self) -> _T_co: ...
    def label(self) -> str: ...

class Cell(Generic[_T]):
    def get(self) -> _T: ...

def make_box() -> Box[f64]: ...
def make_cell() -> Cell[f64]: ...

if sys.version_info >= (3, 99):
    def future() -> int: ...
else:
    def future() -> str: ...
