import torch as th


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

    def __mul__(self, other):
        return dispatcher_singleton.call_primitive(th.mul, (self, other))

    def __add__(self, other):
        return dispatcher_singleton.call_primitive(th.add, (self, other))

    def __sub__(self, other):
        return dispatcher_singleton.call_primitive(th.sub, (self, other))

    def __pow__(self, other):
        return dispatcher_singleton.call_primitive(th.pow, (self, other))

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

    def def_input(self):
        result = AbstractValue(self._gen_name(), self.interpreter)
        self.inputs.append(result)
        return result

    def def_output(self, val):
        self.outputs.append(val)

    def call(self, prim, inputs):
        # How do we figure out how many outputs prim has?
        result = AbstractValue(self._gen_name(), self.interpreter)
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

    values = tuple(interpreter.graph.def_input() for arg in args)
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

def log_batch_rule(x):
    return VmapInterpreterValue(log(x.value), x.bdim, x.interpreter)


batch_rules[th.log] = binary_pw_batch_rule(log)
batch_rules[th.add] = binary_pw_batch_rule(add)
batch_rules[th.sub] = binary_pw_batch_rule(sub)
batch_rules[th.mul] = binary_pw_batch_rule(mul)
batch_rules[th.div] = binary_pw_batch_rule(div)
batch_rules[th.pow] = binary_pw_batch_rule(pow)
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
