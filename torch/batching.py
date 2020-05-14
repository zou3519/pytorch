import torch
import functools
from torch import Tensor

INVALID_BDIM_MSG = (
    'vmap: Expected None or a dim index to map over for arg {idx} but got {dim}'
)

VMAP_ONLY_TENSORS_MSG = (
    'vmap: Got in_dim={dim} for arg {idx}, but arg {idx} is not a Tensor (got '
    '{arg_type}) so it cannot be vmap\'ed over. If you were trying to vmap over a '
    'Tensor inside a collection, we do not yet support that; otherwise, use None '
    'as the respective in_dim.'
)

REQUIRES_FLAT_TUPLE_MSG = (
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

def _validate_in_dims(dims, args):
    # If dims is an int, it is used as the dim to vmap over for *all* inputs.
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

    # If dims is a tuple, then dims[i] is the dim to vmap over for arg[i].
    for idx, (dim, arg) in enumerate(zip(dims, args)):
        if isinstance(arg, Tensor):
            if dim is not None and not isinstance(dim, int):
                raise ValueError(INVALID_BDIM_MSG.format(idx=idx, dim=dim))
            continue
        if dim is None:
            continue
        if isinstance(dim, int):
            raise ValueError(VMAP_ONLY_TENSORS_MSG.format(dim=dim, idx=idx,
                                                          arg_type=type(arg)))
        raise ValueError(REQUIRES_FLAT_TUPLE_MSG.format(dim=dim, idx=idx))


def _maybe_broadcast_in_dims(dims, args):
    if isinstance(dims, int):
        return tuple(dims for _ in range(len(args)))
    return dims


SAME_VMAP_DIM_MSG = (
    'vmap: Some of the specified dimensions to vmap over have different sizes but '
    'but we expected them to all be the same size. Got sizes {sizes}.'
)


def _get_batch_size(args, dims):
    batch_size = None
    batch_sizes = [arg.size(dim)
                   for arg, dim in zip(args, dims)
                   if isinstance(arg, Tensor) and dim is not None]
    if batch_sizes:
        batch_size = batch_sizes[0]
        if not all([size == batch_size for size in batch_sizes]):
            raise ValueError(SAME_VMAP_DIM_MSG.format(batch_sizes))
    return batch_size


def _make_batched(args, dims, level):
    return [torch._make_batched(arg, dim, level)
            if isinstance(arg, Tensor) and dim is not None else arg
            for arg, dim in zip(args, dims)]


def _unwrap_batched_single(output, level, batch_size):
    assert isinstance(output, Tensor)
    return torch._unwrap_batched(output, level, batch_size, 0)


def _unwrap_batched(batched_outputs, vmap_level, batch_size):
    if isinstance(batched_outputs, Tensor):
        return _unwrap_batched_single(batched_outputs, vmap_level, batch_size)
    return tuple(_unwrap_batched_single(out, vmap_level, batch_size)
                 for out in batched_outputs)


OUTPUT_MSG = (
    'vmap({fn}, ...): `{fn}` must return a Tensor or a flat sequence of tensors, got '
    '{out} for return {idx}.'
)

OUTPUT_MSG2 = (
    'vmap({fn}, ...): `{fn}` must return a Tensor or a flat sequence of tensors, got '
    '{out} as the return.'
)


def _validate_outputs(outputs, fn_name):
    """Checks if the output is a single Tensor or multiple Tensors"""
    if isinstance(outputs, Tensor):
        return
    if not hasattr(outputs, '__iter__'):
        raise ValueError(OUTPUT_MSG2.format(fn=fn_name, out=type(outputs)))
    for idx, output in enumerate(outputs):
        if isinstance(output, Tensor):
            continue
        raise ValueError(OUTPUT_MSG.format(fn=fn_name, out=type(output), idx=idx))


# Global vmap level. Used to keep track of how many nested vmaps we are in.
VMAP_LEVEL = 0


def vmap(fn, in_dims=0, out_dims=0):
    if out_dims != 0:
        raise NotImplementedError('NYI: vmap with out_dims. Please don\'t pass an out_dims argument.')

    @functools.wraps(fn)
    def wrapped(*args):
        _validate_in_dims(in_dims, args)
        actual_in_dims = _maybe_broadcast_in_dims(in_dims, args)
        batch_size = _get_batch_size(args, actual_in_dims)
        if batch_size is None:
            raise ValueError('Tried to use vmap without a dimension to map over. This is not supported.')

        global VMAP_LEVEL
        VMAP_LEVEL += 1
        try:
            batched_inputs = _make_batched(args, actual_in_dims, VMAP_LEVEL)
            batched_outputs = fn(*batched_inputs)
            _validate_outputs(batched_outputs, fn.__name__)
            return _unwrap_batched(batched_outputs, VMAP_LEVEL, batch_size)
        finally:
            VMAP_LEVEL -= 1
    return wrapped
