import torch
from .checker import NameChecker
from .registry import Registry


def tensor_ctor(op):
    def fn(*args, names=None, **kwargs):
        tensor = op(*args, **kwargs)
        if names is None:
            names = (None,) * tensor.dim()

        if len(names) != tensor.dim():
            raise RuntimeError('Got {} names but tensor has {} dim',
                               len(names), tensor.dim())
        not_none_names = [n for n in names if n is not None]
        if len(set(not_none_names)) != len(not_none_names):
            raise RuntimeError('Cannot create tensor with duplicate names')

        tensor.names = tuple(names)
        return tensor
    fn.__name__ = op.__name__
    return fn


def pointwise_unary_op(op):
    def fn(tensor):
        output = op(tensor)
        output.names = tensor.names
        return output
    fn.__name__ = op.__name__
    return fn


torch_registry = Registry(torch)

torch_registry.register(tensor_ctor(torch.randn))
torch_registry.register(tensor_ctor(torch.tensor))
torch_registry.register(tensor_ctor(torch.rand))

torch_registry.register(pointwise_unary_op(torch.neg))
