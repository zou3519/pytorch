import torch as th
from functools import partial
import copy


# The dispatcher holds a stack of interpreters.
# When a primitive operation is called:
# - the dispatcher sends the operation to the topmost interpreter
# - the interpreter will "lower the operation to the next interpreter" by
#   calling operations in the next interpreter.
# - This continues recursively until we run out of interpreters. The last
#   interpreter lowers the operations to standard pytorch operations.
class DispatcherSingleton():
    def __init__(self):
        self.interpreter_stack = []

    def push_interpreter(self, interpreter):
        self.interpreter_stack.append(interpreter)

    def pop_interpreter(self):
        return self.interpreter_stack.pop()

    def call_primitive(self, func, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # No interpreters in the stack: call standard pytorch
        if not self.interpreter_stack:
            return func(*args, **kwargs)

        interpreter = self.interpreter_stack.pop()

        if not isinstance(args, tuple):
            import pdb; pdb.set_trace()

        # "Lift" the values to be values understood by this interpreter.
        # E.g., a VmapInterpreter operates on VmapInterpreterValues
        args = tuple(interpreter.lift_value(arg) for arg in args)
        kwargs = {k: interpreter.lift_value(v) for k, v in kwargs.items()}

        result = interpreter.lower_primitive(func, *args, **kwargs)
        self.interpreter_stack.append(interpreter)
        return result

dispatcher_singleton = DispatcherSingleton()
ilevel = 0


# Base class for interpreters
class Interpreter():
    def __init__(self):
        global ilevel
        self.ilevel = ilevel
        ilevel += 1

    def lower_primitive(self, func, *args, **kwargs):
        raise NotImplementedError('abstract')

    def lift_value(self, value):
        raise NotImplementedError('abstract')

    def __repr__(self):
        return f'I{self.ilevel}'


# Base class for interpreter values. An interpreter can only operate on its
# correspondingly defined InterpreterValues.
class InterpreterValue():
    def __init__(self, value, interpreter):
        self.interpreter = interpreter
        self.value = value

    # TODO: Can __torch_function__ handle methods?
    def dim(self):
        return dispatcher_singleton.call_primitive(th.Tensor.dim, (self,))

    def sum(self, dim=None):
        if dim is None:
            return dispatcher_singleton.call_primitive(th.sum, (self,))
        else:
            return dispatcher_singleton.call_primitive(th.sum, (self, dim))

    def __mul__(self, other):
        return dispatcher_singleton.call_primitive(th.mul, (self, other))

    def __add__(self, other):
        return dispatcher_singleton.call_primitive(th.add, (self, other))

    def __sub__(self, other):
        return dispatcher_singleton.call_primitive(th.sub, (self, other))

    def __pow__(self, other):
        return dispatcher_singleton.call_primitive(th.pow, (self, other))

    def __neg__(self):
        return dispatcher_singleton.call_primitive(th.neg, (self,))

# ----------------------------- API ----------------------------------
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


def sum(x, dim=None):
    if dim is None:
        return dispatcher_singleton.call_primitive(th.sum, (x,))
    return dispatcher_singleton.call_primitive(th.sum, (x, dim))


def movedim(x, from_dim, to_dim):
    return dispatcher_singleton.call_primitive(th.movedim, (x, from_dim, to_dim))


def unsqueeze(x, dim):
    return dispatcher_singleton.call_primitive(th.unsqueeze, (x, dim))

# IR
# TODO: evaluate FX's IR to see if they handle our requirements
class AbstractValue(InterpreterValue):
    def __init__(self, name, interpreter):
        super().__init__(self, interpreter)
        self.name = name

    def __repr__(self):
        return self.name

    def __bool__(self):
        raise RuntimeError("Nope nope nope")


class ShapedArray(AbstractValue):
    def __init__(self, name, shape, interpreter):
        super().__init__(name, interpreter)
        self.shape = shape

    def size(self, dim=None):
        if dim is None:
            return self.shape
        return self.shape[dim]

_shape_rules = {}

def broadcast_dim(x, y):
    if x == y:
        return x
    if x == 1:
        return y
    if y == 1:
        return x
    if x != y:
        raise RuntimeError("shape mismatch")

def binary_pw_shape_rule(x, y):
    x_size = x.size() if isinstance(x, AbstractValue) else ()
    y_size = y.size() if isinstance(y, AbstractValue) else ()
    output_size = []
    max_rank = max(len(x_size), len(y_size))
    for i in range(-1, -max_rank, -1):
        output_size.append(broadcast_dim(x, y))
    output_size = list(reversed(output_size))
    return output_size


def sum_shape_rule(x_size):
    return []

_shape_rules[th.add] = binary_pw_shape_rule
_shape_rules[th.sub] = binary_pw_shape_rule
_shape_rules[th.mul] = binary_pw_shape_rule
_shape_rules[th.pow] = binary_pw_shape_rule
_shape_rules[th.sum] = sum_shape_rule


class Call():
    def __init__(self, outputs, prim, inputs):
        self.outputs = outputs
        self.prim = prim
        self.inputs = inputs

    def __repr__(self):
        if isinstance(self.outputs, InterpreterValue):
            outs = [self.outputs]
        else:
            outs = self.outputs
        out = [repr(out) for out in outs]
        ins = [repr(i) for i in self.inputs]
        return f"{' ,'.join(out)} = {self.prim.__name__}({', '.join(ins)})"

class Graph():
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.inputs = []
        self.calls = []
        self.outputs = []
        self.num_values = 0

    def _gen_name(self):
        self.num_values += 1
        if self.num_values <= 26:
            return chr(ord('`') + self.num_values)
        return f'_{self.num_values}'

    def def_input(self, shape):
        result = ShapedArray(self._gen_name(), shape, self.interpreter)
        self.inputs.append(result)
        return result

    def def_output(self, val):
        self.outputs.append(val)

    def call(self, prim, inputs):
        shape_rule = _shape_rules[prim]
        result = shape_rule(*inputs)
        # TODO: this is a hacky check
        if isinstance(result, list):
            result = ShapedArray(self._gen_name(), result, self.interpreter)
        else:
            result = [ShapedArray(self._gen_name(), r, self.interpreter) for r in result]

        self.calls.append(Call(result, prim, inputs))
        return result

    def __repr__(self):
        ins = ', '.join([repr(i) for i in self.inputs])
        outs = ', '.join([repr(i) for i in self.outputs])
        result = f"def graph({ins}):\n"
        for call in self.calls:
            result += f'    {call}\n'
        result += f'    return {outs}'
        return result

class SymbolicTraceInterpreter(Interpreter):
    def __init__(self):
        self.graph = Graph(self)

    def lower_primitive(self, func, *args, **kwargs):
        assert not kwargs
        return self.graph.call(func, args)

    def lift_value(self, value):
        if isinstance(value, th.Tensor):
            # what happens when the graph encounters a tensor?
            assert False
        if isinstance(value, InterpreterValue):
            # or another value?
            assert value.interpreter is self
        return value

def symbolic_trace(func, args):
    interpreter = SymbolicTraceInterpreter()
    dispatcher_singleton.push_interpreter(interpreter)

    # NB: only tensors for now
    values = tuple(interpreter.graph.def_input(arg.shape) for arg in args)
    result = func(*values)
    if isinstance(result, InterpreterValue):
        interpreter.graph.def_output(result)
    else:
        for result in result:
            interpreter.graph.def_output(result)

    dispatcher_singleton.pop_interpreter()
    return interpreter.graph

def mse(x, y):
    return (x - y) ** 2

x = th.rand(10)
y = th.rand(10)

graph = symbolic_trace(mse, (x, y))
expected = """
def graph(a, b):
    c = sub(a, b)
    d = pow(c, 2)
    return d
""".strip()
assert repr(graph) == expected

# grad
# vjp_rules = {}
# 
# def sum_backward(grad, tensor):
#     return grad.expand_as(tensor)
# 
# def mul_backward(grad, tensor, other):
#     return grad * other, grad * tensor
# 
# def add_backward(grad, tensor, other):
#     return grad, grad
# 
# def sub_backward(grad, tensor, other):
#     return grad, -grad
# 
# def relu_backward(grad, tensor):
#     return grad * (tensor >= 0)
# 
# vjp_rules[th.sum] = sum_backward
# vjp_rules[th.mul] = mul_backward
# vjp_rules[th.add] = add_backward
# vjp_rules[th.sub] = sub_backward
# vjp_rules[th.nn.functional.relu] = relu_backward
# 
# def vjp(func, *primals):
#     def wrapped(*tangents):
#         # TODO: there's a question of how symbolic this should be.
#         graph = symbolic_trace(func, primals)
# 
#         # How to duplicate?
#         vjp_graph = copy.deepcopy(graph)
# 
#         # TODO: should really clean up the abstractions
#         sym_tangents = [vjp_graph.def_input(tangent.shape) for tangent in tangents]
# 
#         grad_dict = {}
#         for tangent, output in zip(sym_tangents, graph.outputs):
#             grad_dict[output.name] = tangent.name
# 
#         for node in reversed(graph.calls):
#             grad_fn = vjp_rules[node.prim]
#             vjp_graph.call()
# 

vjp_rules = {}

class ReverseADInterpreter(Interpreter):
    def lower_primitive(self, func, *args, **kwargs):
        if func not in vjp_rules.keys():
            raise RuntimeError(f'NYI: {func}')
        return vjp_rules[func](*args, **kwargs)

    def lift_value(self, value):
        if isinstance(value, InterpreterValue):
            if value.interpreter is self:
                return value
        return ReverseADInterpreterValue(self, value, None)

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
        return -th.tensor(1.) * grad

class MulBackward(GradNode):
    def __init__(self, x, y):
        super().__init__()
        self.x = x.value
        self.y = y.value

    def call(self, grad, i):
        if i == 0:
            return mul(grad, self.y)
        return mul(grad, self.x)

class PowBackward(GradNode):
    def __init__(self, base, exp):
        super().__init__()
        self.base = base.value
        self.exp = exp.value

    def call(self, grad, i):
        if i == 0:
            return pow(mul(mul(grad, self.exp), self.base), sub(self.exp, th.tensor(1)))
        return pow(mul(mul(grad, log(base)), self.base), self.exp)

class SumBackward(GradNode):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape

    def call(self, grad, i):
        return grad.expand(self.input_shape)

class SinBackward(GradNode):
    def __init__(self, x):
        super().__init__()
        self.x = x.value

    def call(self, grad, i):
        return mul(grad, cos(self.x))

class CosBackward(GradNode):
    def __init__(self, x):
        super().__init__()
        self.x = x.value

    def call(self, grad, i):
        return mul(-grad, sin(self.x))

class NegBackward(GradNode):
    def call(self, grad, i):
        return -grad

def add_vjp_rule(x, y):
    node = AddBackward()
    result = x.value + y.value
    if x.grad_fn:
        node.set_next_edge(0, x.grad_fn)
    if y.grad_fn:
        node.set_next_edge(1, y.grad_fn)
    return ReverseADInterpreterValue(x.interpreter, result, node)

def sub_vjp_rule(x, y):
    node = SubBackward()
    result = x.value - y.value
    if x.grad_fn:
        node.set_next_edge(0, x.grad_fn)
    if y.grad_fn:
        node.set_next_edge(1, y.grad_fn)
    return ReverseADInterpreterValue(x.interpreter, result, node)

def pow_vjp_rule(x, y):
    node = PowBackward(x, y)
    result = x.value ** y.value
    if x.grad_fn:
        node.set_next_edge(0, x.grad_fn)
    if y.grad_fn:
        node.set_next_edge(1, y.grad_fn)
    return ReverseADInterpreterValue(x.interpreter, result, node)

def mul_vjp_rule(x, y):
    node = MulBackward(x, y)
    result = mul(x.value, y.value)
    if x.grad_fn:
        node.set_next_edge(0, x.grad_fn)
    if y.grad_fn:
        node.set_next_edge(1, y.grad_fn)
    return ReverseADInterpreterValue(x.interpreter, result, node)

def sum_vjp_rule(x, dim=None):
    assert dim is None
    node = SumBackward(x.value.size())
    result = sum(x.value)
    if x.grad_fn:
        node.set_next_edge(0, x.grad_fn)
    return ReverseADInterpreterValue(x.interpreter, result, node)

def sin_vjp_rule(x):
    node = SinBackward(x)
    result = sin(x.value)
    if x.grad_fn:
        node.set_next_edge(0, x.grad_fn)
    return ReverseADInterpreterValue(x.interpreter, result, node)

def cos_vjp_rule(x):
    node = CosBackward(x)
    result = cos(x.value)
    if x.grad_fn:
        node.set_next_edge(0, x.grad_fn)
    return ReverseADInterpreterValue(x.interpreter, result, node)

def neg_vjp_rule(x):
    node = NegBackward()
    result = -x.value
    if x.grad_fn:
        node.set_next_edge(0, x.grad_fn)
    return ReverseADInterpreterValue(x.interpreter, result, node)



vjp_rules[th.sub] = sub_vjp_rule
vjp_rules[th.pow] = pow_vjp_rule
vjp_rules[th.sum] = sum_vjp_rule
vjp_rules[th.sin] = sin_vjp_rule
vjp_rules[th.mul] = mul_vjp_rule
vjp_rules[th.cos] = cos_vjp_rule
vjp_rules[th.neg] = neg_vjp_rule

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

def mse_loss(x, t):
    return ((x - t) ** 2).sum()

x = th.tensor([[0., 1., 2.], [3., 4., 5.]])
t = th.tensor([[1., 2., 3.], [5., 6., 7.]])
g = th.tensor(1.)

grad_loss = vjp(partial(mse_loss, t=t), (x,))
result, = grad_loss(g)
expected = 2 * (x - t)
assert th.allclose(result, expected)

grad_loss = grad(mse_loss)
result = grad_loss(x, t)
expected = 2 * (x - t)
assert th.allclose(result, expected)

grad_grad_loss = grad(grad(mse_loss))
result = grad_loss(x, t)
expected = 2 * (x - t)
assert th.allclose(result, expected)

# double grad
x = th.randn([])
grad_sin = grad(sin)
assert th.allclose(grad_sin(x), cos(x))
grad_grad_sin = grad(grad(sin))
assert th.allclose(grad_grad_sin(x), -sin(x))

# ----------------------------- forward ad things -------------------------------

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
        log(x.value),
        div(th.tensor(1.), x.value) * x.dual,
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

primal = th.tensor(3.)
tangent = th.tensor(1.)
expected = 1 / primal * tangent
_, dual = jvp(log, [primal], [tangent])
assert th.allclose(dual, expected)

# ----------------------------- vmap things -------------------------------

batch_rules = {}


class VmapInterpreter(Interpreter):
    def lower_primitive(self, func, *args, **kwargs):
        if func not in batch_rules.keys():
            raise RuntimeError(f'NYI: {func}')
        return batch_rules[func](*args, **kwargs)

    def lift_value(self, value):
        if isinstance(value, th.Tensor):
            return VmapInterpreterValue(value, None, self)
        if isinstance(value, InterpreterValue):
            if value.interpreter is self:
                return value
            return VmapInterpreterValue(value, None, self)
        return value


class VmapInterpreterValue(InterpreterValue):
    def __init__(self, value, bdim, interpreter):
        super().__init__(self, interpreter)
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
    if x.bdim:
        return result - 1
    return result


def move_bdim_to_front(x, bdim, result_ndim=None):
    if bdim is None:
        result = unsqueeze(x, 0)
    else:
        result = movedim(x, bdim, 0)
    if result_ndim is None:
        return result
    diff = result_ndim - result.dim()
    for _ in range(diff):
        return result.unsqueeze(1)
    return result


def __ndim(value, bdim):
    result = value.dim()
    if bdim:
        result -= 1
    return result


def binary_pw_batch_rule(func):
    def wrapper(x, y):
        result_ndim = max(__ndim(x.value, x.bdim), __ndim(y.value, y.bdim))
        x_ = move_bdim_to_front(x.value, x.bdim, result_ndim)
        y_ = move_bdim_to_front(y.value, y.bdim, result_ndim)
        z_ = func(x_, y_)
        return VmapInterpreterValue(z_, 0, x.interpreter)
    return wrapper


def movedim_batch_rule(x, from_dim, to_dim):
    x_ = move_bdim_to_front(x.value, x.bdim)
    z_ = movedim(x_, from_dim + 1, to_dim + 1)
    return VmapInterpreterValue(z_, 0, x.interpreter)


def unsqueeze_batch_rule(x, dim):
    x_ = move_bdim_to_front(x.value, x.bdim)
    z_ = unsqueeze(x_, dim + 1)
    return VmapInterpreterValue(z_, 0, x.interpreter)

def unary_batch_rule(func):
    def wrapped(x):
        return VmapInterpreterValue(func(x.value), x.bdim, x.interpreter)
    return wrapped


def sum_batch_rule(x, dim=None):
    x_ = move_bdim_to_front(x.value, x.bdim)
    if dim is None:
        reduce_dims = list(range(1, x_.dim()))
        result = sum(x_, reduce_dims)
    elif isinstance(dim, int):
        result = sum(x_, dim + 1)
    else:
        raise NotImplementedError()
    return VmapInterpreterValue(result, 0, x.interpreter)


batch_rules[th.log] = unary_batch_rule(log)
batch_rules[th.sin] = unary_batch_rule(sin)
batch_rules[th.cos] = unary_batch_rule(cos)
batch_rules[th.add] = binary_pw_batch_rule(add)
batch_rules[th.sub] = binary_pw_batch_rule(sub)
batch_rules[th.mul] = binary_pw_batch_rule(mul)
batch_rules[th.div] = binary_pw_batch_rule(div)
batch_rules[th.pow] = binary_pw_batch_rule(pow)
batch_rules[th.sum] = sum_batch_rule
batch_rules[th.Tensor.dim] = ndim_batch_rule
batch_rules[th.movedim] = movedim_batch_rule
batch_rules[th.unsqueeze] = unsqueeze_batch_rule


def vmap(fn, in_axes):
    def wrapped(*args):
        interpreter = VmapInterpreter()
        dispatcher_singleton.push_interpreter(interpreter)
        vmap_inputs = [VmapInterpreterValue(arg, dim, interpreter)
                       for arg, dim in zip(args, in_axes)]
        output = fn(*vmap_inputs)
        assert output.interpreter is interpreter
        dispatcher_singleton.pop_interpreter()
        return output.value
    return wrapped


# ----------------------------- Tests -------------------------------
x = th.ones(2, 3)
y = th.ones(3)
expected = add(x, y)
assert th.allclose(expected, th.add(x, y))

result = vmap(add, in_axes=(0, None))(x, y)
assert th.allclose(result, expected)

x = th.ones(2, 3)
y = th.ones(5, 3)
expected = y.view(5, 1, 3) + x
result = vmap(vmap(add, in_axes=(0, None)), in_axes=(None, 0))(x, y)
assert th.allclose(result, expected)

x = th.rand(2, 3)
y = th.rand(2, 5, 3)
expected = x.view(2, 1, 3) + y
result = vmap(add, in_axes=(0, 0))(x, y)
assert th.allclose(result, expected)

def mse(x, t):
    diff = sub(x, t)
    result = pow(diff, th.tensor(2))
    return result

x = th.rand(2, 3)
t = th.rand(3)
expected = mse(x, t)
result = vmap(mse, in_axes=(0, None))(x, t)
assert th.allclose(result, expected)

def mse_loss(x, t):
    return ((x - t) ** th.tensor(2)).sum()

x = th.rand(2, 3)
t = th.rand(2, 3)
expected = mse(x, t).sum(-1)
result = vmap(mse_loss, in_axes=(0, 0))(x, t)
assert th.allclose(result, expected)

# per-sample-grad
x = th.rand(10)
jac = vmap(grad(sin), in_axes=(0,))(x)
expected = cos(x)
assert th.allclose(jac, expected)

x = th.rand(2, 3)
t = th.rand(2, 3)
result = vmap(grad(mse_loss), in_axes=(0, 0))(x, t)
expected = 2 * (x - t)
assert th.allclose(result, expected)
