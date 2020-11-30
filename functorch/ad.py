from core import Interpreter, InterpreterValue, dispatcher_singleton
import ops
import torch as th

vjp_rules = {}

class ReverseADInterpreter(Interpreter):
    def lower_primitive(self, func, *args, **kwargs):
        if func == th.Tensor.dim:
            return args[0].value.dim()
        if func not in vjp_rules.keys():
            raise RuntimeError(f'NYI: {func}')
        return vjp_rules[func](*args, **kwargs)

    def lift_value(self, value):
        if isinstance(value, th.Tensor):
            return ReverseADInterpreterValue(self, value, None)
        if isinstance(value, InterpreterValue):
            if value.interpreter is self:
                return value
            return ReverseADInterpreterValue(self, value, None)
        return value

class ReverseADInterpreterValue(InterpreterValue):
    def __init__(self, interpreter, value, grad_fn):
        super().__init__(self, interpreter)
        self.value = value
        self.grad_fn = grad_fn

    def __repr__(self):
        if isinstance(self.value, th.Tensor):
            val = 'Tensor'
        else:
            val = self.value
        return f'ReverseADInterpreterValue({val}, {self.interpreter})'

class GradNode():
    def __init__(self):
        self.out_edges = {}

    def call(self, grad, i):
        raise RuntimeError("abstract")

    def set_next_edge(self, out_nr, grad_node):
        self.out_edges[out_nr] = grad_node

    # Assume single-valued output for now
    def execute(self, grad):
        for i, next_node in self.out_edges.items():
            grad_input = self.call(grad, i)
            next_node.execute(grad_input)

class AccumulateGrad(GradNode):
    def execute(self, grad):
        if not hasattr(self, 'grad'):
            self.grad = grad
        else:
            self.grad = self.grad + grad

class AddBackward(GradNode):
    def call(self, grad, i):
        return grad, grad

class SubBackward(GradNode):
    def call(self, grad, i):
        if i == 0:
            return grad
        return ops.mul(-th.tensor(1.), grad)

class MulBackward(GradNode):
    def __init__(self, x, y):
        super().__init__()
        self.x = x.value
        self.y = y.value

    def execute(self, grad):
        super().execute(grad)

    def call(self, grad, i):
        if i == 0:
            return ops.mul(grad, self.y)
        return ops.mul(grad, self.x)

class _MatmulBackward(GradNode):
    def __init__(self, x, y):
        super().__init__()
        self.x = x.value
        self.y = y.value

    def execute(self, grad):
        super().execute(grad)

    def call(self, grad, i):
        if i == 0:
            return ops._matmul(grad, ops.transpose(self.y, 0, 1))
        return ops._matmul(ops.transpose(self.x, 0, 1), grad)

class PowBackward(GradNode):
    def __init__(self, base, exp):
        super().__init__()
        self.base = base.value
        self.exp = exp.value

    def call(self, grad, i):
        if i == 0:
            return ops.pow(ops.mul(ops.mul(grad, self.exp), self.base), ops.sub(self.exp, th.tensor(1)))
        return ops.pow(ops.mul(ops.mul(grad, ops.log(base)), self.base), self.exp)

class SumBackward(GradNode):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape

    def execute(self, grad):
        super().execute(grad)

    def call(self, grad, i):
        return ops.expand(grad, self.input_shape)

class SinBackward(GradNode):
    def __init__(self, x):
        super().__init__()
        self.x = x.value

    def call(self, grad, i):
        return ops.mul(grad, ops.cos(self.x))

class CosBackward(GradNode):
    def __init__(self, x):
        super().__init__()
        self.x = x.value

    def call(self, grad, i):
        return ops.mul(-grad, ops.sin(self.x))

class NegBackward(GradNode):
    def call(self, grad, i):
        return -ops.grad

class UnsqueezeBackward(GradNode):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def call(self, grad, i):
        return ops.squeeze(grad, self.dim)

class SqueezeBackward(GradNode):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def call(self, grad, i):
        return ops.unsqueeze(grad, self.dim)

def add_vjp_rule(x, y):
    node = AddBackward()
    result = ops.add(x.value, y.value)
    if x.grad_fn:
        node.set_next_edge(0, x.grad_fn)
    if y.grad_fn:
        node.set_next_edge(1, y.grad_fn)
    return ReverseADInterpreterValue(x.interpreter, result, node)

def sub_vjp_rule(x, y):
    node = SubBackward()
    result = ops.sub(x.value, y.value)
    if x.grad_fn:
        node.set_next_edge(0, x.grad_fn)
    if y.grad_fn:
        node.set_next_edge(1, y.grad_fn)
    return ReverseADInterpreterValue(x.interpreter, result, node)

def pow_vjp_rule(x, y):
    node = PowBackward(x, y)
    result = ops.pow(x.value, y.value)
    if x.grad_fn:
        node.set_next_edge(0, x.grad_fn)
    if y.grad_fn:
        node.set_next_edge(1, y.grad_fn)
    return ReverseADInterpreterValue(x.interpreter, result, node)

def mul_vjp_rule(x, y):
    node = MulBackward(x, y)
    result = ops.mul(x.value, y.value)
    if x.grad_fn:
        node.set_next_edge(0, x.grad_fn)
    if y.grad_fn:
        node.set_next_edge(1, y.grad_fn)
    return ReverseADInterpreterValue(x.interpreter, result, node)

def _matmul_vjp_rule(x, y):
    node = _MatmulBackward(x, y)
    result = ops._matmul(x.value, y.value)
    if x.grad_fn:
        node.set_next_edge(0, x.grad_fn)
    if y.grad_fn:
        node.set_next_edge(1, y.grad_fn)
    return ReverseADInterpreterValue(x.interpreter, result, node)

def sum_vjp_rule(x, dim=None):
    assert dim is None
    node = SumBackward(x.value.size())
    result = ops.sum(x.value)
    if x.grad_fn:
        node.set_next_edge(0, x.grad_fn)
    return ReverseADInterpreterValue(x.interpreter, result, node)

def sin_vjp_rule(x):
    node = SinBackward(x)
    result = ops.sin(x.value)
    if x.grad_fn:
        node.set_next_edge(0, x.grad_fn)
    return ReverseADInterpreterValue(x.interpreter, result, node)

def cos_vjp_rule(x):
    node = CosBackward(x)
    result = ops.cos(x.value)
    if x.grad_fn:
        node.set_next_edge(0, x.grad_fn)
    return ReverseADInterpreterValue(x.interpreter, result, node)

def neg_vjp_rule(x):
    node = NegBackward()
    result = ops.neg(x.value)
    if x.grad_fn:
        node.set_next_edge(0, x.grad_fn)
    return ReverseADInterpreterValue(x.interpreter, result, node)

def unsqueeze_vjp_rule(x, dim):
    node = UnsqueezeBackward(dim)
    result = ops.unsqueeze(x.value, dim)
    if x.grad_fn:
        node.set_next_edge(0, x.grad_fn)
    return ReverseADInterpreterValue(x.interpreter, result, node)

def squeeze_vjp_rule(x, dim):
    node = SqueezeBackward(dim)
    result = ops.squeeze(x.value, dim)
    if x.grad_fn:
        node.set_next_edge(0, x.grad_fn)
    return ReverseADInterpreterValue(x.interpreter, result, node)

def gt_vjp_rule(x, y):
    result = ops.gt(x.value, y.value)
    return ReverseADInterpreterValue(x.interpreter, result, None)


vjp_rules[th.sub] = sub_vjp_rule
vjp_rules[th.pow] = pow_vjp_rule
vjp_rules[th.sum] = sum_vjp_rule
vjp_rules[th.sin] = sin_vjp_rule
vjp_rules[th.mul] = mul_vjp_rule
vjp_rules[th.cos] = cos_vjp_rule
vjp_rules[th.neg] = neg_vjp_rule
vjp_rules[th.gt] = gt_vjp_rule
vjp_rules[th.unsqueeze] = unsqueeze_vjp_rule
vjp_rules[th.squeeze] = squeeze_vjp_rule
vjp_rules[th.matmul] = _matmul_vjp_rule

def vjp(func, primals):
    def wrapped(*tangents):
        interpreter = ReverseADInterpreter()
        dispatcher_singleton.push_interpreter(interpreter)

        inputs = tuple(ReverseADInterpreterValue(interpreter, primal, AccumulateGrad())
                       for primal, tangent in zip(primals, tangents))
        result = func(*inputs)
        dispatcher_singleton.pop_interpreter()

        # TODO: assumed single output
        assert len(tangents) == 1
        result.grad_fn.execute(tangents[0])
        grads = tuple(i.grad_fn.grad for i in inputs)
        return grads
    return wrapped

def grad(func):
    def wrapped(primal, *args):
        interpreter = ReverseADInterpreter()
        dispatcher_singleton.push_interpreter(interpreter)
        variable = ReverseADInterpreterValue(interpreter, primal, AccumulateGrad())
        result = func(variable, *args)
        dispatcher_singleton.pop_interpreter()
        result.grad_fn.execute(th.tensor(1.))
        return variable.grad_fn.grad
    return wrapped


forward_formulas = {}

class ForwardADInterpreter(Interpreter):
    def lower_primitive(self, func, *args, **kwargs):
        if func not in forward_formulas.keys():
            raise RuntimeError(f'NYI: {func}')
        return forward_formulas[func](*args, **kwargs)

    def lift_value(self, value):
        if isinstance(value, th.Tensor):
            return ForwardADInterpreterValue(value, zeros_like(value), self)
        if isinstance(value, InterpreterValue):
            if value.interpreter is self:
                return value
            return ForwardADInterpreterValue(value, zeros_like(value), self)
        return value


class ForwardADInterpreterValue(InterpreterValue):
    def __init__(self, value, dual, interpreter):
        super().__init__(self, interpreter)
        self.value = value
        self.dual = dual

    def __repr__(self):
        if isinstance(self.value, th.Tensor):
            val = 'Tensor'
        else:
            val = self.value
        if isinstance(self.dual, th.Tensor):
            dual = 'Tensor'
        else:
            dual = self.dual
        return f'ForwardADInterpreterValue({val}, {dual}, {self.interpreter})'


def add_jvp_rule(x, y):
    return ForwardADInterpreterValue(
        x.value + y.value,
        x.dual + y.dual,
        x.interpreter)

def sub_jvp_rule(x, y):
    return ForwardADInterpreterValue(
        x.value - y.value,
        x.dual - y.dual,
        x.interpreter)

def mul_jvp_rule(x, y):
    return ForwardADInterpreterValue(
        x.value * y.value,
        y * x.dual + x * y.dual,
        x.interpreter)

def pow_jvp_rule(x, y):
    exp = x.value ** y.value
    return ForwardADInterpreterValue(
        exp,
        y.value * x.value ** (y.value - 1) * x.dual + exp * log(x.value) * y.dual,
        x.interpreter)

def log_jvp_rule(x):
    return ForwardADInterpreterValue(
        ops.log(x.value),
        ops.mul(ops.div(th.tensor(1.), x.value), x.dual),
        x.interpreter)

forward_formulas[th.add] = add_jvp_rule
forward_formulas[th.sub] = sub_jvp_rule
forward_formulas[th.mul] = mul_jvp_rule
forward_formulas[th.pow] = pow_jvp_rule
forward_formulas[th.log] = log_jvp_rule

def jvp(func, primals, tangents):
    interpreter = ForwardADInterpreter()
    dispatcher_singleton.push_interpreter(interpreter)
    values = tuple(ForwardADInterpreterValue(primal, tangent, interpreter)
                   for primal, tangent in zip(primals, tangents))
    result = func(*values)
    dispatcher_singleton.pop_interpreter()
    if isinstance(result, ForwardADInterpreterValue):
        return result.value, result.dual
    return tuple(val.value for val in result), tuple(val.dual for val in result)

