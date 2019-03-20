import torch
from typing import TypeVar
import warnings
from .checker import NameCheck
from .registry import Registry


def default_names(tensor):
    return (None,) * tensor.dim()


def set_names_(tensor, names):
    if names is None:
        names = default_names(tensor)

    if len(names) != tensor.dim():
        raise RuntimeError('Got {} names but tensor has {} dim',
                           len(names), tensor.dim())
    not_none_names = [n for n in names if n is not None]
    if len(set(not_none_names)) != len(not_none_names):
        raise RuntimeError('Cannot create tensor with duplicate names')

    tensor.names = tuple(names)
    return tensor


def tensor_ctor(op):
    def fn(*args, names=None, **kwargs):
        return set_names_(op(*args, **kwargs), names)

    fn.__name__ = op.__name__
    return fn


def safe_get_names(tensor):
    if not hasattr(tensor, 'names'):
        tensor.names = (None,) * tensor.dim()
    return tensor.names


def maybe_safe_get_names(thing):
    if not isinstance(thing, torch.Tensor):
        return None
    return safe_get_names(thing)


def fixme_unsafe_op(op):
    def fn(*args, **kwargs):
        warnings.warn('No names implemented for {}'.format(op.__name__))
        return op(*args, **kwargs)
    fn.__name__ = op.__name__
    return fn


def pointwise_unary_op(op):
    def fn(tensor, *args, **kwargs):
        output = op(tensor, *args, **kwargs)
        output.names = safe_get_names(tensor)
        return output
    fn.__name__ = op.__name__
    return fn


def duplicate_names(names):
    s = set()
    for name in names:
        if name is None:
            continue
        if name in s:
            return result
        s.add(name)
    return None


# They're labeled in reverse order
def get_typevars(n):
    return list(reversed([TypeVar('T' + str(i)) for i in range(1, n + 1)]))


def order(tensor, other):
    t_len = len(tensor.names)
    o_len = len(other.names)

    longer, shorter = ((tensor, other) if t_len > o_len else
                       (other, tensor))
    diff = len(longer.names) - len(shorter.names)
    return longer, shorter, diff


def pointwise_binary_op(op):
    def fn(tensor, other, *args, **kwargs):
        # TODO: this is complicated. It probably shouldn't be this complicated.
        if not isinstance(other, torch.Tensor):
            output_names = tensor.names
            output = op(tensor, other, *args, **kwargs)
            output.names = output_names
            return output

        safe_get_names(tensor)
        safe_get_names(other)
        longer, shorter, diff = order(tensor, other)

        tvars = get_typevars(longer.dim())
        output_names = NameCheck().match(longer, *tvars) \
                                  .match(shorter, *tvars[diff:]) \
                                  .lookup(*tvars)
        dup = duplicate_names(output_names)
        if dup:
            raise RuntimeError('Name \'{}\' does not match positionally'.format(dup))

        output = op(tensor, other, *args, **kwargs)
        output.names = output_names
        return output
    fn.__name__ = op.__name__
    return fn


def lookup_dim(tensor, dim):
    try:
        return tensor.names.index(dim)
    except ValueError:
        raise ValueError('Dimension \'{}\' not in tensor.names ({})'.format(
            dim, tensor.names))


# NB: torch.min is very overloaded. The following probably doesn't catch all cases.
def is_dim_arg(dim):
    if isinstance(dim, torch.Tensor):
        if dim.numel() is 1 and dim.dtype is torch.long:
            return True
        return False
    return True


def set_names(tensor, names):
    tensor.names = names
    return tensor


def reduction_op(op):
    def fn(input, dim=None, keepdim=False, **kwargs):
        assert isinstance(input, torch.Tensor)
        safe_get_names(input)

        # reduction to 1 element
        if dim is None:
            assert keepdim is False
            return op(input, **kwargs)

        # dimension reduction
        assert isinstance(dim, (int, str))
        if isinstance(dim, str):
            dim = lookup_dim(input, dim)
        if keepdim:
            output_names = input.names
        else:
            output_names = list(input.names)
            del output_names[dim]
            output_names = tuple(output_names)
        return set_names_(op(input, dim, keepdim, **kwargs), output_names)

    fn.__name__ = op.__name__
    return fn


def min_or_max_op(op):
    def fn(input, dim=None, keepdim=False, **kwargs):
        assert isinstance(input, torch.Tensor)
        safe_get_names(input)

        if dim is None:
            assert keepdim is False
            return op(input, **kwargs)

        if not is_dim_arg(dim):
            assert keepdim is False
            return pointwise_binary_op(op)(input, dim, **kwargs)

        if isinstance(dim, str):
            dim = lookup_dim(input, dim)

        if keepdim:
            output_names = input.names
        else:
            output_names = list(input.names)
            del output_names[dim]
            output_names = tuple(output_names)
        output = op(input, dim, keepdim, **kwargs)
        set_names_(output[0], output_names)
        set_names_(output[1], output_names)
        return output

    fn.__name__ = op.__name__
    return fn


old_softmax = torch.softmax


def softmax(tensor, dim, *args, **kwargs):
    safe_get_names(tensor)
    newdim = lookup_dim(tensor, dim)
    return set_names_(old_softmax(tensor, newdim, *args, **kwargs), tensor.names)


def annotate_names(tensor, *names):
    NameCheck().match(tensor, *names)
    return set_names_(tensor, *names)


def assert_all_named(tensor):
    assert hasattr(tensor, 'names')
    for name in tensor.names:
        if name is None:
            raise RuntimeError('Required names to be non-null, got {}'
                               .format(tensor.names))


def contract(tensor, other, dim):
    assert_all_named(tensor)
    assert_all_named(other)


old_mm = torch.mm


def mm(tensor, other):
    safe_get_names(tensor)
    safe_get_names(other)
    L, M, N = TypeVar('L'), TypeVar('M'), TypeVar('N')
    outnames = NameCheck().match(tensor, L, M) \
                          .match(tensor, M, N) \
                          .lookup(L, N)
    return old_mm(tensor, other).set_names_(outnames)


def match(name1, name2):
    if name1 is None:
        return name2
    if name2 is None:
        return name1
    if name1 == name2:
        return name1
    raise RuntimeError('Name mismatch: {}, {}', name1, name2)


old_bmm = torch.bmm


def bmm(tensor, other):
    safe_get_names(tensor)
    safe_get_names(other)

    def check_bmm_names(tensor, other):
        result = []
        for n1, n2 in zip(reversed(tensor.names[:-2]), reversed(other.names[:-2])):
            result.append(match(n1, n2))
        result.reverse()
        match(tensor.names[-1], other.names[-2])
        result.extend([tensor.names[-2], other.names[-1]])
        return result

    outnames = check_bmm_names(tensor, other)
    return old_bmm(tensor, other).set_names_(outnames)


def matmul(tensor, other):
    if tensor.dim() <= 2 and other.dim() <= 2:
        return mm(tensor, other)
    return bmm(tensor, other)


old_transpose = torch.transpose


def transpose(tensor, dim0, dim1):
    safe_get_names(tensor)
    dim0b = lookup_dim(tensor, dim0)
    dim1b = lookup_dim(tensor, dim1)
    outnames = list(tensor.names)
    outnames[dim1b] = tensor.names[dim0b]
    outnames[dim0b] = tensor.names[dim1b]
    outnames = tuple(outnames)

    return set_names_(old_transpose(tensor, dim0b, dim1b), outnames)


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
torch_registry.register(softmax)

torch_registry.register(min_or_max_op(TORCH.min))
torch_registry.register(min_or_max_op(TORCH.max))

torch_registry.register(reduction_op(TORCH.sum))
torch_registry.register(reduction_op(TORCH.mean))
torch_registry.register(reduction_op(TORCH.std))
torch_registry.register(reduction_op(TORCH.var))

torch_registry.register(fixme_unsafe_op(TORCH.masked_select))

torch_registry.register(annotate_names)
torch_registry.register(transpose)

# FIXME: The following probably dont work
torch_registry.register(bmm)
torch_registry.register(mm)
torch_registry.register(matmul)
