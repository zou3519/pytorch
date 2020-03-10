from collections import namedtuple
import torch
from torch import vmap
import random
import itertools

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
