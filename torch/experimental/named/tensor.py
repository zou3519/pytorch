import torch
from .torch import *
from .registry import Registry
from .checker import NameCheck

TENSOR = torch.Tensor  # old torch.Tensor


def append_names(string_fn):
    def fn(self, *args, **kwargs):
        string = string_fn(self, *args, **kwargs)
        return '{}\n       names={}'.format(string, self.names)
    fn.__name__ == string_fn.__name__
    return fn


def set_names(tensor, names):
    return set_names_(tensor[:], names)


def rename_(tensor, **kwargs):
    mutable_names = list(tensor.names)
    for old_name, new_name in kwargs.items():
        mutable_names[mutable_names.index(old_name)] = new_name
    tensor.names = tuple(mutable_names)
    return tensor


def rename(tensor, **kwargs):
    return rename_(set_names_(tensor[:], tensor.names), **kwargs)


old_size = TENSOR.size


def size(tensor, dim):
    if isinstance(dim, str):
        newdim = lookup_dim(tensor, dim)
    else:
        newdim = dim
    return old_size(tensor, newdim)


tensor_registry = Registry(TENSOR)
tensor_registry.register(TENSOR.dim)
tensor_registry.register(TENSOR.__str__)
tensor_registry.register(append_names(TENSOR.__repr__), '__repr__')
tensor_registry.register(TENSOR.__bool__)
tensor_registry.register(TENSOR.__format__)
tensor_registry.register(TENSOR._nnz)
tensor_registry.register(TENSOR.numel)
tensor_registry.register(TENSOR.is_floating_point)
tensor_registry.register(size)
tensor_registry.register(TENSOR.item)
tensor_registry.register(TENSOR.tolist)
tensor_registry.register(TENSOR.t)
tensor_registry.register(TENSOR.__iter__, '__iter__')

# Tensor methods
tensor_registry.register(set_names)
tensor_registry.register(set_names_)
tensor_registry.register(rename_)
tensor_registry.register(rename)

# Ignores names
tensor_registry.register(TENSOR.reshape)
tensor_registry.register(TENSOR.view)

tensor_registry.register(pointwise_unary_op(TENSOR.double))
tensor_registry.register(pointwise_unary_op(TENSOR.float))

tensor_registry.register(pointwise_unary_op(TENSOR.abs))
tensor_registry.register(pointwise_unary_op(TENSOR.ceil))
tensor_registry.register(softmax)

tensor_registry.register(pointwise_binary_op(TENSOR.__and__), '__and__')
tensor_registry.register(pointwise_binary_op(TENSOR.__iand__), '__iand__')
tensor_registry.register(pointwise_binary_op(TENSOR.__or__), '__or__')
tensor_registry.register(pointwise_binary_op(TENSOR.__ior__), '__ior__')
tensor_registry.register(pointwise_binary_op(TENSOR.__xor__), '__xor__')
tensor_registry.register(pointwise_binary_op(TENSOR.__ixor__), '__ixor__')
tensor_registry.register(pointwise_binary_op(TENSOR.__lshift__), '__lshift__')
tensor_registry.register(pointwise_binary_op(TENSOR.__ilshift__), '__ilshift__')
tensor_registry.register(pointwise_binary_op(TENSOR.__rshift__), '__rshift__')
tensor_registry.register(pointwise_binary_op(TENSOR.__irshift__), '__irshift__')
tensor_registry.register(pointwise_binary_op(TENSOR.__eq__), '__eq__')
tensor_registry.register(pointwise_binary_op(TENSOR.__eq__), 'equal')
tensor_registry.register(pointwise_binary_op(TENSOR.__eq__), 'eq')
tensor_registry.register(pointwise_binary_op(TENSOR.__ne__), '__ne__')
tensor_registry.register(pointwise_binary_op(TENSOR.__ne__), 'ne')
tensor_registry.register(pointwise_binary_op(TENSOR.__gt__), '__gt__')
tensor_registry.register(pointwise_binary_op(TENSOR.__gt__), 'gt')
tensor_registry.register(pointwise_binary_op(TENSOR.__ge__), '__ge__')
tensor_registry.register(pointwise_binary_op(TENSOR.__ge__), 'ge')
tensor_registry.register(pointwise_binary_op(TENSOR.__lt__), '__lt__')
tensor_registry.register(pointwise_binary_op(TENSOR.__lt__), 'lt')
tensor_registry.register(pointwise_binary_op(TENSOR.__le__), '__le__')
tensor_registry.register(pointwise_binary_op(TENSOR.__le__), 'le')
tensor_registry.register(pointwise_binary_op(TENSOR.__rtruediv__), '__rtruediv__')
tensor_registry.register(pointwise_binary_op(TENSOR.__truediv__), '__truediv__')
tensor_registry.register(pointwise_binary_op(TENSOR.__rdiv__), '__rdiv__')
tensor_registry.register(pointwise_binary_op(TENSOR.__div__), '__div__')
tensor_registry.register(pointwise_binary_op(TENSOR.__add__), '__add__')
tensor_registry.register(pointwise_binary_op(TENSOR.__mul__), '__mul__')
tensor_registry.register(pointwise_binary_op(TENSOR.__sub__), '__sub__')
tensor_registry.register(pointwise_binary_op(TENSOR.__rsub__), '__rsub__')

tensor_registry.register(min_or_max_op(TENSOR.min))
tensor_registry.register(min_or_max_op(TENSOR.max))

tensor_registry.register(reduction_op(TENSOR.sum))
tensor_registry.register(reduction_op(TENSOR.mean))
tensor_registry.register(reduction_op(TENSOR.std))
tensor_registry.register(reduction_op(TENSOR.var))

tensor_registry.register(fixme_unsafe_op(TENSOR.__getitem__))

tensor_registry.register(transpose)

# FIXME: the following probably don't work
tensor_registry.register(bmm)
tensor_registry.register(mm)
tensor_registry.register(matmul)
