import torch as th


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

    def call_primitive(self, opname, args=(), kwargs=None):
        if not isinstance(opname, str):
            raise RuntimeError(f"opname {type(opname)}")
        if kwargs is None:
            kwargs = {}

        # No interpreters in the stack: call standard pytorch
        if not self.interpreter_stack:
            if opname == 'ndim':
                return args[0].dim()
            op = getattr(th, opname)
            return op(*args, **kwargs)

        interpreter = self.interpreter_stack.pop()

        # "Lift" the values to be values understood by this interpreter.
        # E.g., a VmapInterpreter operates on VmapValues
        args = tuple(interpreter.lift_value(arg) for arg in args)
        kwargs = {k: interpreter.lift_value(v) for k, v in kwargs.items()}

        result = interpreter.call_primitive(opname, *args, **kwargs)
        self.interpreter_stack.append(interpreter)
        return result

ilevel = 0


class Interpreter():
    def __init__(self):
        global ilevel
        self.ilevel = ilevel
        ilevel += 1

    def call_primitive(self, opname, *args, **kwargs):
        raise NotImplementedError('abstract')

    def lift_value(self, value):
        raise NotImplementedError('abstract')


class InterpreterValue():
    def __init__(self, value, interpreter):
        self.interpreter = interpreter
        self.value = value

#     def __torch_function__(self, func, types, args=(), kwargs=None):
#         if kwargs is None:
#             kwargs = {}
#         return dispatcher.call_primitive(func, args, kwargs)
# 

dispatcher = DispatcherSingleton()


# TODO: Can we use __torch_function__ instead of having to write out each of these?
def add(x, y):
    return dispatcher.call_primitive('add', (x, y))


def sub(x, y):
    return dispatcher.call_primitive('sub', (x, y))


def mul(x, y):
    return dispatcher.call_primitive('mul', (x, y))


def div(x, y):
    return dispatcher.call_primitive('div', (x, y))


def unsqueeze(x, dim):
    return dispatcher.call_primitive('unsqueeze', (x, dim))


def movedim(x, from_dim, to_dim):
    return dispatcher.call_primitive('movedim', (x, from_dim, to_dim))


def ndim(x):
    return dispatcher.call_primitive('ndim', (x,))


batch_rules = {}


class VmapInterpreter(Interpreter):
    def call_primitive(self, opname, *args, **kwargs):
        if opname not in batch_rules.keys():
            raise RuntimeError(f'NYI: {opname}')
        return batch_rules[opname](*args, **kwargs)

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


def ndim_batch_rule(x):
    result = ndim(x.value)
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
    diff = result_ndim - ndim(result)
    for _ in range(diff):
        return result.unsqueeze(1)
    return result


def __ndim(value, bdim):
    result = ndim(value)
    if bdim:
        result -= 1
    return result


def add_batch_rule(x, y):
    result_ndim = max(__ndim(x.value, x.bdim), __ndim(y.value, y.bdim))
    x_ = move_bdim_to_front(x.value, x.bdim, result_ndim)
    y_ = move_bdim_to_front(y.value, y.bdim, result_ndim)
    z_ = add(x_, y_)
    return VmapInterpreterValue(z_, 0, x.interpreter)


def movedim_batch_rule(x, from_dim, to_dim):
    x_ = move_bdim_to_front(x.value, x.bdim)
    z_ = movedim(x_, from_dim + 1, to_dim + 1)
    return VmapInterpreterValue(z_, 0, x.interpreter)


def unsqueeze_batch_rule(x, dim):
    x_ = move_bdim_to_front(x.value, x.bdim)
    z_ = unsqueeze(x_, dim + 1)
    return VmapInterpreterValue(z_, 0, x.interpreter)


batch_rules['add'] = add_batch_rule
batch_rules['ndim'] = ndim_batch_rule
batch_rules['movedim'] = movedim_batch_rule
batch_rules['unsqueeze'] = unsqueeze_batch_rule


def vmap(fn, in_axes):
    def wrapped(*args):
        interpreter = VmapInterpreter()
        dispatcher.push_interpreter(interpreter)
        vmap_inputs = [VmapInterpreterValue(arg, dim, interpreter)
                       for arg, dim in zip(args, in_axes)]
        output = fn(*vmap_inputs)
        assert output.interpreter is interpreter
        dispatcher.pop_interpreter()
        return output.value
    return wrapped


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
