# Plain Python helper for instance_attrs.py: what Lizzard code may not do.


def blank(cls):
    return cls.__new__(cls)


def delete(obj, name):
    delattr(obj, name)


def materialize(obj):
    return sorted(vars(obj))


def dict_x(obj):
    return obj.__dict__["x"]


def replace_dict(obj):
    obj.__dict__ = {"x": 12.0, "y": 1.0}


def set_x(obj, v):
    obj.x = v


def with_getattr(base):
    class G(base):
        def __getattr__(self, name):
            return 99.0

    g = G.__new__(G)
    return g


def add_property(cls):
    cls.x = property(lambda self: -1.0, lambda self, v: None)


def remove_property(cls):
    del cls.x


def shadow_method(obj, name):
    setattr(obj, name, lambda punct: "shadowed" + punct)


def replace_method(cls, name):
    setattr(cls, name, lambda self, punct: "replaced " + self.name + punct)
