import torch
from collections.abc import Iterable


def get_context(funcname):
    return Context(funcname)


def _prepare(ctx, args, kwargs):
    # print("_prepare({})".format(ctx.name))
    return ctx.prepare(*args, **kwargs)


def _wrap(ctx, arg):
    # print("_wrap({})".format(ctx.name))
    return ctx.wrap(arg)


def default_names(tensor):
    return (None,) * tensor.dim()


class BaseTensor:
    def __init__(self, tensor, names=None):
        self.tensor = tensor
        if names is None:
            names = default_names(tensor)
        self.names = names

    def __repr__(self):
        return self.tensor.__repr__()

    def is_callable(self, attr):
        fn = getattr(self.tensor, attr)
        return hasattr(fn, '__call__'), fn

    def __getitem__(self, *args, **kwargs):
        return lift(self.tensor.__getitem__(*args, **kwargs))

    def __format__(self, spec):
        return self.tensor.__format__(spec)

    def __getattr__(self, attr):
        funcname = attr
        can_call, attr = self.is_callable(attr)
        if not can_call:
            return attr

        def fn(*args, **kwargs):
            ctx = get_context(funcname)
            args, kwargs = _prepare(ctx, args, kwargs)
            out = attr(*args, **kwargs)
            out = _wrap(ctx, out)
            return out

        return fn

    def __add__(self, other):
        return torch.add(self, other)

    def __sub__(self, other):
        return torch.sub(self, other)

    def __mul__(self, other):
        return torch.mul(self, other)

    def __truediv__(self, other):
        return torch.div(self, other)

    def __rsub__(self, other):
        return torch.rsub(self, other)

    def __rdiv__(self, other):
        return torch.rdiv(self, other)

    def __ge__(self, other):
        return torch.ge(self, other)

    def __gt__(self, other):
        return torch.gt(self, other)


def lift(thing):
    if isinstance(thing, tuple):
        return tuple(lift(list(thing)))
    if isinstance(thing, list):
        return [lift(foo) for foo in thing]
    if isinstance(thing, torch.Tensor):
        return BaseTensor(thing)
    return thing


def lower(thing):
    if isinstance(thing, tuple):
        return tuple(lower(list(thing)))
    if isinstance(thing, list):
        return [lower(foo) for foo in thing]
    if isinstance(thing, BaseTensor):
        return thing.tensor
    return thing


def lower_all(*args, **kwargs):
    args = [lower(arg) for arg in args]
    kwargs = {k: lower(v) for k, v in kwargs.items()}
    return args, kwargs


class Context:
    def __init__(self, name):
        self.name = name

    def prepare(self, *args, **kwargs):
        return lower_all(*args, **kwargs)

    def wrap(self, outputs):
        return lift(outputs)


class CtorContext:
    def prepare(self, *args, names=None, **kwargs):
        self.names = names
        return lower_all(*args, **kwargs)

    def wrap(self, out):
        if self.names is None:
            self.names = default_names(out)
        return BaseTensor(out, self.names)
