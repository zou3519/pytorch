import torch


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
            func(*args, **kwargs)

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

    # TODO: this unfortunately doesn't handle factory functions
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return dispatcher_singleton.call_primitive(func, args, kwargs)

    # TODO: Can __torch_function__ handle methods?
    def dim(self):
        return dispatcher_singleton.call_primitive(torch.Tensor.dim, (self,))

    def __mul__(self, other):
        return dispatcher_singleton.call_primitive(torch.mul, (self, other))

    def __add__(self, other):
        return dispatcher_singleton.call_primitive(torch.add, (self, other))

    def __sub__(self, other):
        return dispatcher_singleton.call_primitive(torch.sub, (self, other))

    def __pow__(self, other):
        return dispatcher_singleton.call_primitive(torch.pow, (self, other))

# ----------------------------- forward ad things -------------------------------

forward_formulas = {}

class ForwardADInterpreter(Interpreter):
    def lower_primitive(self, func, *args, **kwargs):
        if func not in forward_formulas.keys():
            raise RuntimeError(f'NYI: {func}')
        return forward_formulas[func](*args, **kwargs)

    def lift_value(self, value):
        if isinstance(value, torch.Tensor):
            return ForwardADInterpreterValue(value, torch.zeros_like(value), self)
        if isinstance(value, InterpreterValue):
            if value.interpreter is self:
                return value
            return ForwardADInterpreterValue(value, torch.zeros_like(value), self)
        return value


class ForwardADInterpreterValue(InterpreterValue):
    def __init__(self, value, dual, interpreter):
        super().__init__(self, interpreter)
        self.value = value
        self.dual = dual

    def __repr__(self):
        if isinstance(self.value, torch.Tensor):
            val = 'Tensor'
        else:
            val = self.value
        if isinstance(self.dual, torch.Tensor):
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
        y.value * x.value ** (y.value - 1) * x.dual + exp * torch.log(x.value) * y.dual,
        x.interpreter)

def log_jvp_rule(x):
    return ForwardADInterpreterValue(
        torch.log(x.value),
        torch.div(torch.tensor(1.), x.value) * x.dual,
        x.interpreter)

forward_formulas[torch.add] = add_jvp_rule
forward_formulas[torch.sub] = sub_jvp_rule
forward_formulas[torch.mul] = mul_jvp_rule
forward_formulas[torch.pow] = pow_jvp_rule
forward_formulas[torch.log] = log_jvp_rule

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

primal = torch.tensor(3.)
tangent = torch.tensor(1.)
expected = 1 / primal * tangent
_, dual = jvp(torch.log, [primal], [tangent])
assert torch.allclose(dual, expected)

# ----------------------------- vmap things -------------------------------

batch_rules = {}


class VmapInterpreter(Interpreter):
    def lower_primitive(self, func, *args, **kwargs):
        if func not in batch_rules.keys():
            raise RuntimeError(f'NYI: {func}')
        return batch_rules[func](*args, **kwargs)

    def lift_value(self, value):
        if isinstance(value, torch.Tensor):
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
        if isinstance(self.value, torch.Tensor):
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
        result = torch.unsqueeze(x, 0)
    else:
        result = torch.movedim(x, bdim, 0)
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
    z_ = torch.movedim(x_, from_dim + 1, to_dim + 1)
    return VmapInterpreterValue(z_, 0, x.interpreter)


def unsqueeze_batch_rule(x, dim):
    x_ = move_bdim_to_front(x.value, x.bdim)
    z_ = torch.unsqueeze(x_, dim + 1)
    return VmapInterpreterValue(z_, 0, x.interpreter)

def log_batch_rule(x):
    return VmapInterpreterValue(torch.log(x.value), x.bdim, x.interpreter)


batch_rules[torch.log] = binary_pw_batch_rule(torch.log)
batch_rules[torch.add] = binary_pw_batch_rule(torch.add)
batch_rules[torch.sub] = binary_pw_batch_rule(torch.sub)
batch_rules[torch.mul] = binary_pw_batch_rule(torch.mul)
batch_rules[torch.div] = binary_pw_batch_rule(torch.div)
batch_rules[torch.pow] = binary_pw_batch_rule(torch.pow)
batch_rules[torch.Tensor.dim] = ndim_batch_rule
batch_rules[torch.movedim] = movedim_batch_rule
batch_rules[torch.unsqueeze] = unsqueeze_batch_rule


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
x = torch.ones(2, 3)
y = torch.ones(3)
expected = torch.add(x, y)
assert torch.allclose(expected, torch.add(x, y))

result = vmap(torch.add, in_axes=(0, None))(x, y)
assert torch.allclose(result, expected)

x = torch.ones(2, 3)
y = torch.ones(5, 3)
expected = y.view(5, 1, 3) + x
result = vmap(vmap(torch.add, in_axes=(0, None)), in_axes=(None, 0))(x, y)
assert torch.allclose(result, expected)

x = torch.rand(2, 3)
y = torch.rand(2, 5, 3)
expected = x.view(2, 1, 3) + y
result = vmap(torch.add, in_axes=(0, 0))(x, y)
assert torch.allclose(result, expected)

def mse(x, t):
    diff = torch.sub(x, t)
    result = torch.pow(diff, torch.tensor(2))
    return result

x = torch.rand(2, 3)
t = torch.rand(3)
expected = mse(x, t)
result = vmap(mse, in_axes=(0, None))(x, t)
assert torch.allclose(result, expected)
