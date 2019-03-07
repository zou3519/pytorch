import torch
import itertools
from typing import TypeVar
from collections.abc import Iterable
import warnings

factories = {'randn', 'rand', 'zeros', 'ones', 'tensor'}
pw_unary = {'relu'}
pw_binary = {'add', 'sub', 'div', 'truediv', 'mul', 'rdiv', 'rsub', 'rtruediv'}


def get_context(funcname):
    if funcname in factories:
        return FactoryContext()
    if funcname in pw_unary:
        return PointwiseUnaryContext()
    if funcname in pw_binary:
        return PointwiseBinaryContext()
    if funcname == 'mm':
        return MMContext()
    return Context(funcname)


def _prepare(ctx, *args, **kwargs):
    # print("_prepare({})".format(ctx.name))
    return ctx.prepare(*args, **kwargs)


def _wrap(ctx, arg):
    # print("_wrap({})".format(ctx.name))
    return ctx.wrap(arg)


def default_names(tensor):
    return (None,) * tensor.dim()


class NameCheck:
    def __init__(self):
        self.typevars = {}

    def match(self, tensor, *names):
        for name, annotated in zip(tensor.names, names):
            if name is None or annotated is None:
                continue
            if isinstance(annotated, TypeVar):
                typ = str(annotated)[1:]
                if typ in self.typevars.keys():
                    if name != self.typevars[typ]:
                        raise RuntimeError(
                            ('Name mismatch: {} was previously matched with \'{}\' ' +
                             'but is now also matched with \'{}\'').format(
                                 typ, self.typevars[typ], name))
                else:
                    self.typevars[typ] = name
            elif name != annotated:
                raise RuntimeError('Name mismatch: {} and {}'.format(name, annotated))
        return self

    def lookup(self, *names):
        result = []
        for name in names:
            if isinstance(name, TypeVar):
                typ = str(name)[1:]
                result.append(self.typevars[typ])
            else:
                result.append(name)
        return tuple(result)


class BaseTensor:
    def __init__(self, tensor, names=None):
        self.tensor = tensor
        if names is None:
            names = default_names(tensor)
        self.names = names

    def __repr__(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = self.tensor.__repr__()
            return '{},\n       names={}'.format(out, self.names)

    def is_callable(self, attr):
        fn = getattr(self.tensor, attr)
        return hasattr(fn, '__call__'), fn

    def __getattr__(self, attr):
        funcname = attr
        can_call, attr = self.is_callable(attr)
        if not can_call:
            return attr

        def fn(*args, **kwargs):
            ctx = get_context(funcname)
            args, kwargs = _prepare(ctx, self, *args, **kwargs)
            out = attr(*args[1:], **kwargs)
            out = _wrap(ctx, out)
            return out

        return fn

    # XXX: should really be attribute-only
    def namedshape(self):
        return tuple(zip(tensor.size(), self.names))

    def set_names(self, *names, **kwargs):
        if names is None:
            self.names = default_names(self.tensor)

        if kwargs is None:
            assert len(names) == len(self.names)
            self.names = names
        else:
            assert len(names) == 0
            for k, v in kwargs.items():
                assert k in self.names
                self.names[self.names.index(k)] = v

    def __getitem__(self, *args, **kwargs):
        return lift(self.tensor.__getitem__(*args, **kwargs))

    def __format__(self, spec):
        return self.tensor.__format__(spec)

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
        warnings.warn('{}: name inference NYI'.format(name))

    def prepare(self, *args, **kwargs):
        return lower_all(*args, **kwargs)

    def wrap(self, outputs):
        # raise RuntimeError('NYI: {}'.format(self.name))
        return lift(outputs)


class FactoryContext:
    def prepare(self, *args, names=None, **kwargs):
        self.names = names
        return lower_all(*args, **kwargs)

    def wrap(self, out):
        if self.names is None:
            self.names = default_names(out)
        return BaseTensor(out, self.names)


class PointwiseUnaryContext:
    def prepare(self, tensor):
        self.names = tensor.names
        return [lower(tensor)], {}

    def wrap(self, out):
        return BaseTensor(out, self.names)


class PointwiseBinaryContext:
    def prepare(self, tensor, other):
        outnames = []
        for n1, n2 in itertools.zip_longest(reversed(tensor.names), reversed(other.names)):
            if n1 is None:
                outnames.append(n2)
                continue
            if n2 is None:
                outnames.append(n1)
                continue
            if n1 != n2:
                raise RuntimeError(
                    'Names {}, {} do not match when adding Tensor{}, Tensor{}'.format(
                        n1, n2, tensor.names, other.names))
            outnames.append(n1)
        self.names = tuple(reversed(outnames))
        if len(other.names) > len(tensor.names):
            self.names = other.names
        return [lower(tensor), lower(other)], {}

    def wrap(self, out):
        return BaseTensor(out, self.names)


A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')


class MMContext:
    def prepare(self, tensor, other):
        nc = NameCheck()
        nc.match(tensor, A, B).match(other, B, C)
        self.outnames = nc.lookup(A, C)
        return [lower(tensor), lower(other)], {}

    def wrap(self, out):
        return BaseTensor(out, self.outnames)
