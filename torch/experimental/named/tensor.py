import torch
from .torch import pointwise_unary_op
from .registry import Registry

TENSOR = torch.Tensor  # old torch.Tensor


def append_names(string_fn):
    def fn(self, *args, **kwargs):
        string = string_fn(self, *args, **kwargs)
        return '{}\n       names={}'.format(string, self.names)
    fn.__name__ == string_fn.__name__
    return fn


tensor_registry = Registry(TENSOR)
tensor_registry.register(TENSOR.dim)
tensor_registry.register(TENSOR.__str__)
tensor_registry.register(append_names(TENSOR.__repr__), '__repr__')
tensor_registry.register(TENSOR.__bool__)
tensor_registry.register(TENSOR.__format__)
tensor_registry.register(TENSOR._nnz)
tensor_registry.register(TENSOR.numel)
tensor_registry.register(TENSOR.is_floating_point)
tensor_registry.register(TENSOR.size)
tensor_registry.register(TENSOR.item)
tensor_registry.register(TENSOR.tolist)

# Ignores names
tensor_registry.register(TENSOR.reshape)
tensor_registry.register(TENSOR.view)

tensor_registry.register(pointwise_unary_op(TENSOR.double))
tensor_registry.register(pointwise_unary_op(TENSOR.float))

tensor_registry.register(pointwise_unary_op(TENSOR.abs))
tensor_registry.register(pointwise_unary_op(TENSOR.ceil))

# TODO: needs names
tensor_registry.register(TENSOR.__eq__, '__eq__')
tensor_registry.register(TENSOR.__ne__, '__ne__')
tensor_registry.register(TENSOR.__gt__, '__gt__')
tensor_registry.register(TENSOR.__lt__, '__lt__')
tensor_registry.register(TENSOR.__ne__, 'ne')
tensor_registry.register(TENSOR.__and__, '__and__')
tensor_registry.register(TENSOR.__iter__, '__iter__')
tensor_registry.register(TENSOR.__truediv__, '__truediv__')
tensor_registry.register(TENSOR.min)
tensor_registry.register(TENSOR.max)
tensor_registry.register(TENSOR.__getitem__)
