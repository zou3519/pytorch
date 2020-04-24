import torch
import functools
from torch import Tensor
from typing import Optional
from collections import OrderedDict

VMAP_LEVEL = 0

# @torch.jit.script
# def broadcast_to(tensor: Tensor, ndim: int):
#     old_sizes = tensor.sizes()
#     if old_sizes == ndim:
#         return tensor
#     assert len(old_sizes) <= ndim
#     diff = ndim - len(old_sizes)
#     for i in range(diff):
#         tensor.unsqueeze(0)
#     return tensor
# 
# @torch.jit.script
# def move_batch_dim_to_front(tensor: Tensor,
#                             batch_dim: Optional[int],
#                             result_dim: int):
#     if batch_dim is None:
#         return broadcast_to(tensor, result_dim)
#     extra_dims = result_dim - tensor.dim()
#     result = broadcast_to(tensor, result_dim)
#     return result.transpose(0, batch_dim + extra_dims)
# 
# @torch.jit.script
# def min_result_dim(tensor: Tensor, batch_dim: Optional[int]) -> int:
#     result = tensor.dim()
#     if batch_dim is None:
#         result += 1
#     return result
# 
# @torch.jit.script
# def mul_batching_rule(self: Tensor, self_bdim: Optional[int],
#                       other: Tensor, other_bdim: Optional[int]):
#     self_dim = min_result_dim(self, self_bdim)
#     other_dim = min_result_dim(other, other_bdim)
#     result_dim = max(self_dim, other_dim)
# 
#     self = move_batch_dim_to_front(self, self_bdim, result_dim)
#     other = move_batch_dim_to_front(other, other_bdim, result_dim)
#     return self * other, 0

EXPECTED_BDIM_MSG = (
    'vmap: Expected None or a dim index to map over for arg {idx} but got {dim}'
)

NESTED_COLLECTION_MSG = (
    'vmap: Got in_dim={dim} for arg {idx}, but arg {idx} is not a Tensor (got '
    '{arg_type}) so it cannot be vmap\'ed over. If you were trying to vmap over a '
    'Tensor inside a collection, we do not yet support that; otherwise, use None '
    'as the respective in_dim.'
)

FLAT_TUPLE_MSG = (
    'vmap: Expected `in_dims` to be a flat tuple of None or int but got in_dim={dim} '
    'for arg {idx}. If you were trying to vmap over a Tensor inside a collection, we '
    ' do not yet support that.'
)

NOT_TENSORS_MSG = (
    'vmap: in_dims is {in_dims} so we attempted to vmap over dim {in_dims} for '
    'all inputs. However, we can\'t do this for inputs that are not Tensors '
    '(inputs at indices {indices} are not tensors). Perhaps you meant to use '
    '`vmap(func, in_dims={suggested_in_dims}, ...)`.'
)




def _collection_has_tensors(collection):
    if isinstance(collection, dict):
        return _collection_has_tensors(collection.values())
    if isinstance(collection, OrderedDict):
        return _collection_has_tensors(collection.values())
    if isinstance(collection, tuple):
        pass
    if isinstance(collection, list):
        pass


def _validate_in_dims(dims, args):
    if isinstance(dims, int):
        for idx, arg in enumerate(args):
            if idx is None or isinstance(arg, Tensor):
                continue
            suggested_in_dims = tuple(dims if isinstance(arg, Tensor) else None for arg in args)
            incorrect_idxs = [idx for idx, in_dim in zip(range(len(args)), suggested_in_dims)
                              if in_dim is None]
            raise ValueError(NOT_TENSORS_MSG.format(in_dims=dims,
                                                    indices=incorrect_idxs,
                                                    suggested_in_dims=suggested_in_dims))
        return

    if not isinstance(dims, tuple):
        raise ValueError('vmap: Expected `in_dims` to be int or tuple, got: ' + str(type(dims)))

    for idx, (dim, arg) in enumerate(zip(dims, args)):
        if isinstance(arg, Tensor):
            if dim is not None and not isinstance(dim, int):
                raise ValueError(EXPECTED_BDIM_MSG.format(idx=idx, dim=dim))
            continue
        if dim is None:
            continue
        if isinstance(dim, int):
            raise ValueError(NESTED_COLLECTION_MSG.format(dim=dim, idx=idx,
                                                          arg_type=type(arg)))
        raise ValueError(FLAT_TUPLE_MSG.format(dim=dim, idx=idx))


def _make_batched(args, dims, level):
    _validate_in_dims(dims, args)
    if isinstance(dims, int):
        dims = tuple(dims for _ in range(len(args)))

    batch_size = None
    batch_sizes = [arg.size(dim)
                   for arg, dim in zip(args, dims)
                   if isinstance(arg, Tensor) and dim is not None]
    if batch_sizes:
        batch_size = batch_sizes[0]
        assert all([size == batch_size for size in batch_sizes])
    return [torch._make_batched(arg, dim, level)
            if isinstance(arg, Tensor) and dim is not None else arg
            for arg, dim in zip(args, dims)], batch_size


def _unwrap_batched_single(output, batch_size):
    if batch_size is None:
        return output
    if isinstance(output, torch.Tensor):
        if torch._is_batched(output):
            return torch._unwrap_batched(output, 0)
        output = output.expand(batch_size, *output.shape)
        return output
    else:
        assert False  # NYI


def _unwrapped_batched(batched_outputs, batch_size):
    return [_unwrap_batched_single(out, batch_size)
            for out in batched_outputs]


def vmap(fn, in_dims=0):
    @functools.wraps(fn)
    def wrapped(*args):
        global VMAP_LEVEL
        VMAP_LEVEL += 1
        try:
            batched_inputs, batch_size = _make_batched(args, in_dims, VMAP_LEVEL)
            batched_outputs = fn(*batched_inputs)
            # TODO: we assume only one output for now
            return _unwrap_batched_single(batched_outputs, batch_size)
        finally:
            VMAP_LEVEL -= 1
    return wrapped
