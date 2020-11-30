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
