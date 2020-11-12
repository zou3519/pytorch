import torch


# The dispatcher holds a stack of interpreters
# When a primitive operation is called, it is sent to the topmost interpreter.
# That interpreter will lower the operation by calling operations in the next
# interpreter.
# This continues recursively until we run out of interpreters, at which point
# we run the operations in standard pytorch.
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

        result = interpreter.call_primitive(func, *args, **kwargs)
        self.interpreter_stack.append(interpreter)
        return result

dispatcher_singleton = DispatcherSingleton()
ilevel = 0


class Interpreter():
    def __init__(self):
        global ilevel
        self.ilevel = ilevel
        ilevel += 1

    def call_primitive(self, func, *args, **kwargs):
        raise NotImplementedError('abstract')

    def lift_value(self, value):
        raise NotImplementedError('abstract')

    def __repr__(self):
        return f'I{self.ilevel}'


class InterpreterValue():
    def __init__(self, value, interpreter):
        self.interpreter = interpreter
        self.value = value

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return dispatcher_singleton.call_primitive(func, args, kwargs)

    # TODO: Can __torch_function__ handle methods?
    def dim(self):
        return dispatcher_singleton.call_primitive(torch.Tensor.dim, (self,))

# ----------------------------- vmap things -------------------------------

batch_rules = {}


class VmapInterpreter(Interpreter):
    def call_primitive(self, func, *args, **kwargs):
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
        return f'VmapIValue(val, {self.bdim}, {self.interpreter})'


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
