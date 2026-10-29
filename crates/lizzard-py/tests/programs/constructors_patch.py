# Plain Python helper for constructors.py: classes Lizzard code may not define.
import abc


def py_subclass(base):
    class Sub(base):
        def __init__(self, x, y):
            super().__init__(x * 10, y)
            self.extra = "sub"
    return Sub


def with_new(base):
    class WithNew(base):
        def __new__(cls, x, y):
            obj = super().__new__(cls)
            obj.extra = "new"
            return obj
    return WithNew


def with_meta(base):
    class Meta(type):
        def __call__(cls, *args):
            obj = super().__call__(*args)
            obj.extra = "meta"
            return obj
    return Meta("WithMeta", (base,), {})


def returns_value(base):
    class Bad(base):
        def __init__(self, x, y):
            return 42
    return Bad


def abstract(base):
    class Abstract(base, abc.ABC):
        @abc.abstractmethod
        def must(self):
            ...
    return Abstract


def replace_init(cls):
    def __init__(self, x, y):
        self.x = -x
        self.y = y
    cls.__init__ = __init__


def extra(obj):
    return getattr(obj, "extra", None)


def items(obj):
    return list(vars(obj).items())
