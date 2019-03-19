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


def safe_get_names(tensor):
    if not hasattr(tensor, 'names'):
        tensor.names = (None,) * tensor.dim()
    return tensor.names


def pointwise_unary_op(op):
    def fn(tensor):
        output = op(tensor)
        output.names = safe_get_names(tensor)
        return output
    fn.__name__ = op.__name__
    return fn


TORCH = torch
torch_registry = Registry(TORCH)

torch_registry.register(TORCH.get_default_dtype)
torch_registry.register(TORCH.no_grad)
torch_registry.register(TORCH.is_grad_enabled)
torch_registry.register(TORCH.set_grad_enabled)

torch_registry.register(tensor_ctor(TORCH.randn))
torch_registry.register(tensor_ctor(TORCH.tensor))
torch_registry.register(tensor_ctor(TORCH.rand))

torch_registry.register(pointwise_unary_op(TORCH.abs))
torch_registry.register(pointwise_unary_op(TORCH.ceil))
torch_registry.register(pointwise_unary_op(TORCH.isfinite))
torch_registry.register(pointwise_unary_op(TORCH.neg))

# TODO: needs names
torch_registry.register(TORCH.masked_select)
