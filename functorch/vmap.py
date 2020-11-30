from core import Interpreter, InterpreterValue, dispatcher_singleton
import ops
import torch as th

batch_rules = {}


class VmapInterpreter(Interpreter):
    def process_primitive(self, func, *args, **kwargs):
        if func not in batch_rules.keys():
            raise RuntimeError(f'NYI: {func}')
        return batch_rules[func](*args, **kwargs)

    def lift_value(self, value):
        if isinstance(value, th.Tensor):
            return VmapInterpreterValue(value, None, self)
        if isinstance(value, InterpreterValue):
            if value.interpreter is self:
                return value
            assert value.interpreter.ilevel <= self.ilevel
            return VmapInterpreterValue(value, None, self)
        return value


class VmapInterpreterValue(InterpreterValue):
    def __init__(self, value, bdim, interpreter):
        super().__init__(self, interpreter)
        assert isinstance(value, InterpreterValue) or isinstance(value, th.Tensor)
        self.value = value
        self.bdim = bdim

    def __repr__(self):
        if isinstance(self.value, th.Tensor):
            val = 'Tensor'
        else:
            val = self.value
        return f'VmapIValue({val}, {self.bdim}, {self.interpreter})'

    # TODO: why am I overriding this here?
    def size(self):
        result = list(self.value.size())
        if self.bdim is not None:
            del result[self.bdim]
        return result


def ndim_batch_rule(x):
    result = x.value.dim()
    if x.bdim is not None:
        return result - 1
    return result


def move_bdim_to_front(x, bdim, result_ndim=None):
    if bdim is None:
        result = ops.unsqueeze(x, 0)
    else:
        result = ops.movedim(x, bdim, 0)
    if result_ndim is None:
        return result
    diff = result_ndim - result.dim()
    for _ in range(diff):
        return ops.unsqueeze(result, 1)
    return result


def __ndim(value, bdim):
    result = value.dim()
    if bdim is None:
        return result + 1
    return result


def binary_pw_batch_rule(func):
    def wrapper(x, y):
        result_ndim = max(__ndim(x.value, x.bdim), __ndim(y.value, y.bdim))
        x_ = move_bdim_to_front(x.value, x.bdim, result_ndim)
        y_ = move_bdim_to_front(y.value, y.bdim, result_ndim)
        z_ = func(x_, y_)
        return VmapInterpreterValue(z_, 0, x.interpreter)
    return wrapper

def _matmul_batch_rule(x, y):
    result_ndim = max(__ndim(x.value, x.bdim), __ndim(y.value, y.bdim))
    x_ = move_bdim_to_front(x.value, x.bdim, result_ndim)
    y_ = move_bdim_to_front(y.value, y.bdim, result_ndim)
    z_ = ops._matmul(x_, y_)
    return VmapInterpreterValue(z_, 0, x.interpreter)


def movedim_batch_rule(x, from_dim, to_dim):
    x_ = move_bdim_to_front(x.value, x.bdim)
    z_ = ops.movedim(x_, from_dim + 1, to_dim + 1)
    return VmapInterpreterValue(z_, 0, x.interpreter)


def unsqueeze_batch_rule(x, dim):
    x_ = move_bdim_to_front(x.value, x.bdim)
    if dim >= 0:
        z_ = ops.unsqueeze(x_, dim + 1)
    else:
        z_ = ops.unsqueeze(x_, dim)
    return VmapInterpreterValue(z_, 0, x.interpreter)

def unary_batch_rule(func):
    def wrapped(x):
        return VmapInterpreterValue(func(x.value), x.bdim, x.interpreter)
    return wrapped


def sum_batch_rule(x, dim=None):
    x_ = move_bdim_to_front(x.value, x.bdim)
    if dim is None:
        reduce_dims = list(range(1, x_.dim()))
        result = ops.sum(x_, reduce_dims)
    elif isinstance(dim, int):
        result = ops.sum(x_, dim + 1)
    else:
        raise NotImplementedError()
    return VmapInterpreterValue(result, 0, x.interpreter)

def squeeze_batch_rule(x, dim):
    x_ = move_bdim_to_front(x.value, x.bdim)
    if dim >= 0:
        result = ops.squeeze(x_, dim + 1)
    else:
        result = ops.squeeze(x_, dim)
    return VmapInterpreterValue(result, 0, x.interpreter)


def expand_batch_rule(x, shape):
    x_ = move_bdim_to_front(x.value, x.bdim)
    x_ = ops.movedim(x_, 0, -1)
    result = ops.expand(x_, list(shape) + [x_.size()[-1]])
    result = ops.movedim(result, -1, 0)
    return VmapInterpreterValue(result, 0, x.interpreter)

def transpose_batch_rule(x, dim0, dim1):
    assert dim0 >= 0
    assert dim1 >= 0
    x_ = move_bdim_to_front(x.value, x.bdim)
    result = ops.transpose(x_, dim0 + 1, dim1 + 1)
    return VmapInterpreterValue(result, 0, x.interpreter)


batch_rules[th.log] = unary_batch_rule(ops.log)
batch_rules[th.sin] = unary_batch_rule(ops.sin)
batch_rules[th.cos] = unary_batch_rule(ops.cos)
batch_rules[th.add] = binary_pw_batch_rule(ops.add)
batch_rules[th.sub] = binary_pw_batch_rule(ops.sub)
batch_rules[th.mul] = binary_pw_batch_rule(ops.mul)
batch_rules[th.div] = binary_pw_batch_rule(ops.div)
batch_rules[th.pow] = binary_pw_batch_rule(ops.pow)
batch_rules[th.gt] = binary_pw_batch_rule(ops.gt)
batch_rules[th.sum] = sum_batch_rule
batch_rules[th.Tensor.dim] = ndim_batch_rule
batch_rules[th.movedim] = movedim_batch_rule
batch_rules[th.unsqueeze] = unsqueeze_batch_rule
batch_rules[th.squeeze] = squeeze_batch_rule
batch_rules[th.matmul] = _matmul_batch_rule
batch_rules[th.Tensor.expand] = expand_batch_rule
batch_rules[th.transpose] = transpose_batch_rule


def vmap(fn, in_axes=0):
    def wrapped(*args):
        _in_axes = in_axes
        if _in_axes == 0:
            _in_axes = (0,) * len(args)
        interpreter = VmapInterpreter()
        dispatcher_singleton.push_interpreter(interpreter)
        vmap_inputs = [VmapInterpreterValue(arg, dim, interpreter)
                       for arg, dim in zip(args, _in_axes)]
        output = fn(*vmap_inputs)
        assert output.interpreter is interpreter
        dispatcher_singleton.pop_interpreter()
        return output.value
    return wrapped

