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


def split_dim(tensor, dim, new_dim_ordered_shape):
    idim = lookup_dim(tensor, dim)

    new_names = list(new_dim_ordered_shape.keys())
    new_names = tuple(list(tensor.names[:idim]) + new_names + list(tensor.names[idim + 1:]))

    new_dim_shape = tuple(v for v in new_dim_ordered_shape.values())
    new_shape = tensor.shape[:idim] + new_dim_shape + tensor.shape[idim + 1:]
    return set_names_(tensor.view(new_shape), new_names)


def join_dims(tensor, dims, new_dim):
    idims = [lookup_dim(tensor, dim) for dim in dims]
    for i, v in enumerate(idims):
        if i == len(idims) - 1:
            continue
        if v + 1 != idims[i + 1]:
            raise RuntimeError('join_dims: dims must be consecutive in memory')

    new_dim_size = 1
    for idim in idims:
        new_dim_size *= tensor.shape[idim]

    outnames = tensor.names[:idims[0]] + (new_dim,) + tensor.names[idims[-1] + 1:]
    outshape = tensor.shape[:idims[0]] + (new_dim_size,) + tensor.shape[idims[-1] + 1:]

    return set_names_(tensor.view(outshape), outnames)


def join_dims_exper(tensor, dims, new_dim, where=None):
    if where is None:
        where = dims[-1]

    idims = [lookup_dim(tensor, dim) for dim in dims]
    for i, v in enumerate(idims):
        if i == len(idims) - 1:
            continue
        if v + 1 != idims[i + 1]:
            raise RuntimeError('join_dims: dims must be consecutive in memory')
    wheredim = lookup_dim(tensor, wheredim)

    new_dim_size = 1
    for idim in idims:
        new_dim_size *= tensor.shape[idim]

    outnames = tensor.names[:idims[0]] + (new_dim,) + tensor.names[idims[-1] + 1:]
    outshape = tensor.shape[:idims[0]] + (new_dim_size,) + tensor.shape[idims[-1] + 1:]

    return set_names_(tensor.view(outshape), outnames)


def align_as(tensor, other):
    # XXX: This needs to handle the None case
    outnames = other.names
    ptr1 = len(tensor.names) - 1
    ptr2 = len(other.names) - 1

    shape = []
    while ptr1 >= 0 and ptr2 >= 0:
        if tensor.names[ptr1] == other.names[ptr2]:
            shape.append(other.shape[ptr2])
            ptr1 -= 1
            ptr2 -= 1
        else:
            ptr2 -= 1
            shape.append(1)
    shape.reverse()
    return set_names_(tensor.view(shape), outnames)


def check_is_permutation(names, other_names):
    error_msg = ('Names {} is not a permutation of {}'
                 .format(names, other_names))
    name_counts = {name: 1 for name in names}
    for name in other_names:
        if name not in name_counts:
            raise RuntimeError(error_msg)
        name_counts[name] -= 1
        if name_counts[name] is 0:
            del name_counts[name]
    if len(name_counts.keys()) is not 0:
        raise RuntimeError(error_msg)

old_permute = torch.Tensor.permute


def shift(tensor, old_dims, new_dims):
    safe_get_names(tensor)
    check_is_permutation(old_dims, new_dims)
    verify_exists_in_order(old_dims, tensor.names)

    permutation = [x for x in range(tensor.dim())]

    old_dims_idx = [lookup_dim(tensor, n) for n in old_dims]
    new_dims_idx = [lookup_dim(tensor, n) for n in new_dims]
    for old_idx, new_idx in zip(old_dims_idx, new_dims_idx):
        permutation[old_idx] = new_idx

    outnames = tuple([tensor.names[i] for i in permutation])

    return set_names_(old_permute(tensor, permutation), outnames)


tensor_registry = Registry(TENSOR)
tensor_registry.register(TENSOR.dim)
tensor_registry.register(align_as)
tensor_registry.register(TENSOR.__len__)
tensor_registry.register(TENSOR.ndimension)
tensor_registry.register(TENSOR.__str__)
tensor_registry.register(append_names(TENSOR.__repr__), '__repr__')
tensor_registry.register(TENSOR.__bool__)
tensor_registry.register(TENSOR.__format__)
tensor_registry.register(TENSOR._nnz)
tensor_registry.register(TENSOR.numel)
tensor_registry.register(TENSOR.is_floating_point)
tensor_registry.register(size)
tensor_registry.register(shift)
tensor_registry.register(split_dim)
tensor_registry.register(join_dims)
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
tensor_registry.register(pointwise_unary_op(TENSOR.contiguous))
tensor_registry.register(pointwise_unary_op(TENSOR.clone))
tensor_registry.register(pointwise_unary_op(TENSOR.uniform_))
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
tensor_registry.register(pointwise_binary_op(TENSOR.__iadd__), '__iadd__')
tensor_registry.register(pointwise_binary_op(TENSOR.__mul__), '__mul__')
tensor_registry.register(pointwise_binary_op(TENSOR.__imul__), '__imul__')
tensor_registry.register(pointwise_binary_op(TENSOR.__sub__), '__sub__')
tensor_registry.register(pointwise_binary_op(TENSOR.__isub__), '__isub__')
tensor_registry.register(pointwise_binary_op(TENSOR.__rsub__), '__rsub__')

tensor_registry.register(min_or_max_op(TENSOR.min))
tensor_registry.register(min_or_max_op(TENSOR.max))

tensor_registry.register(reduction_op(TENSOR.sum))
tensor_registry.register(reduction_op(TENSOR.mean))
tensor_registry.register(reduction_op(TENSOR.std))
tensor_registry.register(reduction_op(TENSOR.var))

tensor_registry.register(fixme_unsafe_op(TENSOR.__getitem__))

tensor_registry.register(transpose)

tensor_registry.register(bmm)
tensor_registry.register(masked_fill)
tensor_registry.register(masked_fill_)
tensor_registry.register(mm)
tensor_registry.register(matmul)
