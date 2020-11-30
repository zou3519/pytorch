from core import dispatcher_singleton
import torch as th

# TODO: These should be probably be __torch_function__, but
# __torch_function__ can't override factory functions

def mul(x, y):
    return dispatcher_singleton.call_primitive(th.mul, (x, y))


def div(x, y):
    return dispatcher_singleton.call_primitive(th.div, (x, y))


def sub(x, y):
    return dispatcher_singleton.call_primitive(th.sub, (x, y))


def add(x, y):
    return dispatcher_singleton.call_primitive(th.add, (x, y))


def pow(x, y):
    return dispatcher_singleton.call_primitive(th.pow, (x, y))


def log(x):
    return dispatcher_singleton.call_primitive(th.log, (x,))


def sin(x):
    return dispatcher_singleton.call_primitive(th.sin, (x,))


def cos(x):
    return dispatcher_singleton.call_primitive(th.cos, (x,))


def neg(x):
    return dispatcher_singleton.call_primitive(th.neg, (x,))


def gt(x, y):
    return dispatcher_singleton.call_primitive(th.gt, (x, y))


def transpose(x, dim0, dim1):
    return dispatcher_singleton.call_primitive(th.transpose, (x, dim0, dim1))


def expand(x, shape):
    return dispatcher_singleton.call_primitive(th.Tensor.expand, (x, shape))


# Only handles matmul for x, y with rank >= 2
def _matmul(x, y):
    return dispatcher_singleton.call_primitive(th.matmul, (x, y))

def sum(x, dim=None):
    if dim is None:
        return dispatcher_singleton.call_primitive(th.sum, (x,))
    return dispatcher_singleton.call_primitive(th.sum, (x, dim))


def movedim(x, from_dim, to_dim):
    return dispatcher_singleton.call_primitive(th.movedim, (x, from_dim, to_dim))


def unsqueeze(x, dim):
    return dispatcher_singleton.call_primitive(th.unsqueeze, (x, dim))


def squeeze(x, dim):
    return dispatcher_singleton.call_primitive(th.squeeze, (x, dim))


def relu(x):
    return x * gt(x, th.tensor(0))


def matmul(x, y):
    if x.dim() == 1 and y.dim() == 1:
        result = _matmul(unsqueeze(x, -2), unsqueeze(y, -1))
        return squeeze(squeeze(result, -1), -1)
    if x.dim() >= 2 and y.dim() == 1:
        result = _matmul(x, unsqueeze(y, -1))
        return squeeze(result, -1)
    if x.dim() == 1 and y.dim() >= 2:
        result = _matmul(unsqueeze(x, -2), y)
        return squeeze(result, -2)
    if x.dim() >= 2 and y.dim() >= 2:
        return _matmul(x, y)
    raise RuntimeError()

