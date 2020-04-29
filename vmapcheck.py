from collections import namedtuple
import torch
from torch import vmap
import random
import itertools
import copy

OperatorMetadata = namedtuple('OperatorMetadata', [
    'has_single_output',
    'dim_indices',
    'is_view',
    'is_inplace',
    # If is_inplace, specifies which inputs to return.
    'inplace_returns',
])


def opmeta(has_single_output=True, dim_indices=None, is_view=False, is_inplace=False, inplace_returns=False):
    return OperatorMetadata(has_single_output, dim_indices, is_view, is_inplace, inplace_returns)


def unbind_bdim(inp, dim, batch_size):
    if dim is None:
        return [inp] * batch_size
    result = inp.unbind(dim)
    assert len(result) == batch_size
    return result


def maybe_size(inp, dim):
    if dim is None:
        return None
    return inp.size(dim)


def find_batch_size(inputs, dims):
    sizes = [maybe_size(inp, dim) for inp, dim in zip(inputs, dims)]
    sizes = [size for size in sizes if size is not None]
    assert len(sizes) > 0
    assert all(size == sizes[0] for size in sizes)
    return sizes[0]


def vmap_reference(func, in_dims, inputs, opmeta):
    if all(dim is None for dim in in_dims):
        return func(*inputs)

    # TODO: process dimensions

    batch_size = find_batch_size(inputs, in_dims)
    exploded_args = [unbind_bdim(inp, dim, batch_size) for inp, dim in zip(inputs, in_dims)]

    sharded_args = list(zip(*exploded_args))
    if opmeta.has_single_output:
        sharded_results = [[func(*args)] for args in sharded_args]
    else:
        sharded_results = [func(*args) for args in sharded_args]

    if opmeta.is_inplace:
        return [inputs[i] for i in zip(opmeta.inplace_returns)]

    exploded_results = list(zip(*sharded_results))
    results = tuple(torch.stack(exploded_result, 0)
                    for exploded_result in exploded_results)
    if opmeta.has_single_output:
        return results[0]
    else:
        return results


def run_check(func, in_dims, inputs, opmeta):
    invocation = 'vmap({}, {})(*inputs)'.format(func.__name__, in_dims)
    if opmeta.is_view:
        raise RuntimeError("We can't check this")

    test_output = vmap(func, in_dims)(*inputs)
    expected_output = vmap_reference(func, in_dims, inputs, opmeta)
    if opmeta.has_single_output:
        if not torch.allclose(test_output, expected_output):
            assert False, invocation
    else:
        for test_out, expected_out in zip(test_output, expected_output):
            assert torch.allclose(test_out, expected_out), invocation


def generate_input(examples, in_dim):
    if in_dim is None:
        return examples[0]
    if in_dim == 0:
        return examples
    # Move batch dimension to in_dim
    return torch.stack(examples.unbind(0), in_dim)


def generate_inputs(example_inputs, in_dims):
    return [generate_input(examples, in_dim)
            for examples, in_dim
            in zip(example_inputs, in_dims)]


def permutations_of(can_batch_inputs, example_inputs):
    def cases(can_batch, examples):
        return ([0, None, random.randrange(1, examples.dim())]
                if can_batch else [None])

    all_cases = [cases(can_batch, examples) for can_batch, examples
                 in zip(can_batch_inputs, example_inputs)]
    return itertools.product(*all_cases)


def vmap_test(func, example_inputs, can_batch_inputs, opmeta):
    for in_dims in permutations_of(can_batch_inputs, example_inputs):
        inputs = generate_inputs(example_inputs, in_dims)
        run_check(func, in_dims, inputs, opmeta)

def foo(x, y):
    return x + y, x * y


x = torch.randn(2, 3, 5)
y = torch.randn(2, 3, 5)
vmap_test(torch.add, [x, y], [True, True], opmeta())
# vmap_test(foo, [x, y], [True, True], opmeta(has_single_output=False))

def add_dim(shape, dim, size):
    result = list(shape)
    result.insert(dim, size)
    return tuple(result)

def maybe_broadcast(thing, size):
    if hasattr(thing, '__iter__'):
        assert len(thing) == size
    return [thing] * size

def should_create_tensor_from(arg):
    return isinstance(arg, tuple)

def get_inputs(bdim_size, bdims, inputs_or_shapes, dtypes, device, input_fns):
    result = []
    for bdim, input_or_shape, dtype, input_fn, in zip(bdims, inputs_or_shapes, dtypes, input_fns):
        if bdim is not None:
            assert should_create_tensor_from(input_or_shape)
            input_or_shape = add_dim(input_or_shape, bdim, bdim_size)
        if should_create_tensor_from(input_or_shape):
            inp = input_fn(input_or_shape, dtype=dtype, device=device)
        else:
            inp = input_or_shape
        result.append(inp)
    return tuple(result)

def slice_inputs(inputs, bdims, i):
    result = []
    for inp, bdim in zip(inputs, bdims):
        if bdim is None:
            result.append(inp)
        else:
            result.append(inp.select(bdim, i))
    return tuple(result)

def check_vmap(op, bdim_size, bdims, inputs_or_shapes,
               dtypes=torch.float, device='cpu', input_fns=torch.rand):
    num_inputs = len(bdims)
    assert len(inputs_or_shapes) == num_inputs
    dtypes = maybe_broadcast(dtypes, num_inputs)
    input_fns = maybe_broadcast(input_fns, num_inputs)

    inputs = get_inputs(bdim_size, bdims, inputs_or_shapes, dtypes, device, input_fns)
    output = vmap(op, bdims)(*inputs)
    expected = torch.stack([op(*slice_inputs(inputs, bdims, i)) for i in range(bdim_size)])
    assert torch.allclose(output, expected)

def check_vmap_inplace(op, bdim_size, bdims, inputs_or_shapes,
                       dtypes=torch.float, device='cpu', input_fns=torch.rand):
    num_inputs = len(bdims)
    assert len(inputs_or_shapes) == num_inputs
    dtypes = maybe_broadcast(dtypes, num_inputs)
    input_fns = maybe_broadcast(input_fns, num_inputs)

    inputs = get_inputs(bdim_size, bdims, inputs_or_shapes, dtypes, device, input_fns)
    inputs_clone = copy.deepcopy(inputs)

    output = vmap(op, bdims)(*inputs)

    # NB: The output of an in-place operation is usually the first argument.
    torch.stack([op(*slice_inputs(inputs_clone, bdims, i)) for i in range(bdim_size)])
    expected = inputs_clone[0]

    assert torch.allclose(output, expected)

from torch import Tensor
check_vmap(torch.add, 3, bdims=(0, 1), inputs_or_shapes=((5,), (5,)))
check_vmap(torch.add, 3, bdims=(None, 1), inputs_or_shapes=((5,), (5,)))
check_vmap(torch.add, 3, bdims=(1, None), inputs_or_shapes=((5,), (5,)))

check_vmap_nested(torch.add, outer_bdims, inner_bdims, outer_bdim_size, inner_bdim_size, examples_or_shapes)

check_vmap_inplace(Tensor.add_, 3, bdims=(0, 0), inputs_or_shapes=((5,), (5,)))

# requirements:
# - su
# check_vmap(torch.sum, 3, (0, None), ((5,), (5,), 1))
# check_vmap_inplace(op, bdim_size, bdims, shapes, dtypes, device, input_fn)
# check_vmap_view(op, bdim_size, bdims, shapes, dtypes, device, input_fn)
