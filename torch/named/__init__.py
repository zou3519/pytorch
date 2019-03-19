import torch
import itertools
from typing import TypeVar
from collections.abc import Iterable
import warnings

factories = {'randn', 'rand', 'zeros', 'ones', 'tensor'}
pw_unary = {'abs', 'acos', 'asin', 'atan', 'atan2', 'ceil', 'clamp', 'cos',
            'cosh', 'digamma', 'erf', 'erfc', 'erfinv', 'exp', 'expm1',
            'floor', 'log10', 'log2', 'mvlgamma', 'neg', 'reciprocal', 'round', 'relu'}
pw_binary = {'add', 'sub', 'div', 'truediv', 'mul',
             'rdiv', 'rsub', 'rtruediv', 'ne', 'eq', 'ge', 'le', 'gt', 'lt'}


def get_context(funcname):
    if funcname in factories:
        return FactoryContext(funcname)
    if funcname in pw_unary:
        return PointwiseUnaryContext(funcname)
    if funcname in pw_binary:
        return PointwiseBinaryContext(funcname)
    if funcname == 'mm':
        return MMContext(funcname)
    return Context(funcname)


def _prepare(ctx, *args, **kwargs):
    return ctx.prepare(*args, **kwargs)


def _wrap(ctx, arg):
    return ctx.wrap(arg)


def default_names(tensor):
    return (None,) * tensor.dim()


def typevar_name(typevar):
    return str(typevar)[1:]


class NameCheck:
    def __init__(self, weak=False):
        self.typevars = {}
        self.weak = weak

    def match(self, tensor, *names):
        for name, annotated in zip(tensor.names, names):
            if name is None or annotated is None:
                continue
            if isinstance(annotated, str):
                if not self.weak and name != annotated:
                    raise RuntimeError('Name mismatch: {} and {}'.format(name, annotated))
                continue

            assert isinstance(annotated, TypeVar)
            typ = typevar_name(annotated)
            if typ not in self.typevars.keys():
                self.typevars[typ] = name
                continue

            if name == self.typevars[typ]:
                continue

            if weak:
                self.typevars[typ] = float('NaN')
                continue

            raise RuntimeError(
                ('Name mismatch: {} was previously matched with \'{}\' ' +
                 'but is now also matched with \'{}\'').format(
                     typ, self.typevars[typ], name))

        return self

    def lookup(self, *names):
        result = []
        for name in names:
            if isinstance(name, TypeVar):
                typ = typevar_name(name)
                result.append(self.typevars[typ])
            else:
                result.append(name)
        return tuple(result)


class BaseTensor:
    def __init__(self, tensor, names=None, weaknames=False):
        self.tensor = tensor
        if names is None:
            names = default_names(tensor)
        self.names = names
        self.weaknames = weaknames

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

    def __rtruediv__(self, other):
        return torch.mul(self.reciprocal(), other)

    def __div__(self, other):
        return torch.div(self, other)

    def __rdiv__(self, other):
        return torch.mul(self.reciprocal(), other)

    def __rsub__(self, other):
        return torch.rsub(self, other)

    def __ne__(self, other):
        return torch.ne(self, other)

    def __eq__(self, other):
        return torch.ne(self, other)

    def __ge__(self, other):
        return torch.ge(self, other)

    def __le__(self, other):
        return torch.ge(self, other)

    def __lt__(self, other):
        return torch.lt(self, other)

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
        return lift(outputs)


class FactoryContext:
    def __init__(self, name):
        self.name = name

    def prepare(self, *args, names=None, **kwargs):
        self.names = names
        return lower_all(*args, **kwargs)

    def wrap(self, out):
        if self.names is None:
            self.names = default_names(out)
        return BaseTensor(out, self.names)


class PointwiseUnaryContext:
    def __init__(self, name):
        self.name = name

    def prepare(self, tensor):
        self.names = tensor.names
        return [lower(tensor)], {}

    def wrap(self, out):
        return BaseTensor(out, self.names)


class PointwiseBinaryContext:
    def __init__(self, name):
        self.name = name

    def prepare(self, tensor, other):
        if not isinstance(other, torch.Tensor):
            self.names = tensor.names
            return [lower(tensor), lower(other)], {}

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
    def __init__(self, name):
        self.name = name

    def prepare(self, tensor, other):
        nc = NameCheck().match(tensor, A, B).match(other, B, C)
        self.outnames = nc.lookup(A, C)
        return [lower(tensor), lower(other)], {}

    def wrap(self, out):
        return BaseTensor(out, self.outnames)
