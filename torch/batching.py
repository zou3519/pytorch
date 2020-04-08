import torch
import functools
from torch import Tensor
from typing import Optional

VMAP_LEVEL = 0

@torch.jit.script
def broadcast_to(tensor: Tensor, ndim: int):
    old_sizes = tensor.sizes()
    if old_sizes == ndim:
        return tensor
    assert len(old_sizes) <= ndim
    diff = ndim - len(old_sizes)
    for i in range(diff):
        tensor.unsqueeze(0)
    return tensor

@torch.jit.script
def move_batch_dim_to_front(tensor: Tensor,
                            batch_dim: Optional[int],
                            result_dim: int):
    if batch_dim is None:
        return broadcast_to(tensor, result_dim)
    extra_dims = result_dim - tensor.dim()
    result = broadcast_to(tensor, result_dim)
    return result.transpose(0, batch_dim + extra_dims)

@torch.jit.script
def min_result_dim(tensor: Tensor, batch_dim: Optional[int]) -> int:
    result = tensor.dim()
    if batch_dim is None:
        result += 1
    return result

@torch.jit.script
def mul_batching_rule(self: Tensor, self_bdim: Optional[int],
                      other: Tensor, other_bdim: Optional[int]):
    self_dim = min_result_dim(self, self_bdim)
    other_dim = min_result_dim(other, other_bdim)
    result_dim = max(self_dim, other_dim)

    self = move_batch_dim_to_front(self, self_bdim, result_dim)
    other = move_batch_dim_to_front(other, other_bdim, result_dim)
    return self * other, 0


def _make_batched(args, dims, level):
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


def vmap(fn, in_axes):
    @functools.wraps(fn)
    def wrapped(*args):
        global VMAP_LEVEL
        VMAP_LEVEL += 1
        try:
            batched_inputs, batch_size = _make_batched(args, in_axes, VMAP_LEVEL)
            batched_outputs = fn(*batched_inputs)
            # TODO: we assume only one output for now
            return _unwrap_batched_single(batched_outputs, batch_size)
        finally:
            VMAP_LEVEL -= 1
    return wrapped
