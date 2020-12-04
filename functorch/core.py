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

        result = interpreter.process_primitive(func, *args, **kwargs)
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

    def process_primitive(self, func, *args, **kwargs):
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
    def __init__(self, interpreter):
        super().__init__(self, interpreter)
        assert interpreter is not None

    def __repr__(self):
        return "AbstractValue"

    def __bool__(self):
        raise RuntimeError("Nope nope nope")


class ShapedArray(AbstractValue):
    def __init__(self, interpreter, shape):
        super().__init__(interpreter)
        assert isinstance(shape, tuple)
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
        output_size.append(broadcast_dim(x_size[i], y_size[i]))
    output_size = tuple(reversed(output_size))
    return output_size

def tensor_shape_rule(*args):
    x = th.tensor(args)
    return x.size()

def sum_shape_rule(x_size):
    return ()

_shape_rules[th.tensor] = tensor_shape_rule
_shape_rules[th.add] = binary_pw_shape_rule
_shape_rules[th.sub] = binary_pw_shape_rule
_shape_rules[th.mul] = binary_pw_shape_rule
_shape_rules[th.pow] = binary_pw_shape_rule
_shape_rules[th.sum] = sum_shape_rule


class Variable():
    def __init__(self, id, abstract_value):
        self.id = id
        self.abstract_value = abstract_value

    def __repr__(self):
        if self.id <= 26:
            return chr(ord('`') + self.id)
        return f'_{self.id}'

class Literal():
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return repr(self.value)

class FunctionCall():
    def __init__(self, outputs, prim, inputs):
        self.outputs = outputs
        self.prim = prim
        self.inputs = inputs

    def __repr__(self):
        out = [repr(out) for out in self.outputs]
        ins = [repr(i) for i in self.inputs]
        return f"{' ,'.join(out)} = {self.prim.__name__}({', '.join(ins)})"


class Graph():
    def __init__(self):
        self.inputs = []
        self.function_calls = []
        self.outputs = []
        self.num_variables = 0

    def def_input(self, abstract_value):
        result = self.def_var(abstract_value)
        self.inputs.append(result)
        return result

    def def_output(self, variable):
        self.outputs.append(variable)

    def def_var(self, abstract_value):
        self.num_variables += 1
        return Variable(self.num_variables, abstract_value)

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
        self.graph = Graph()
        self.abstract_value_to_variable = {}

    def process_primitive(self, prim, *args, **kwargs):
        assert not kwargs
        shape_rule = _shape_rules[prim]
        result_shapes = shape_rule(*args)
        # NB: shape is always a tuple
        if not isinstance(result_shapes, list):
            result_shapes = [result_shapes]
        outputs = [ShapedArray(self, shape) for shape in result_shapes]

        # Write to the graph
        invars = [self.abstract_value_to_variable[arg]
                  if isinstance(arg, AbstractValue)
                  else Literal(arg)
                  for arg in args]
        outvars = [self.graph.def_var(out) for out in outputs]
        self.graph.function_calls.append(FunctionCall(outvars, prim, invars))

        for outvar, value in zip(outvars, outputs):
            self.abstract_value_to_variable[value] = outvar
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def def_input(self, value):
        assert isinstance(value, th.Tensor)
        result = ShapedArray(self, value.shape)
        invar = self.graph.def_input(result)
        self.abstract_value_to_variable[result] = invar
        return result

    def def_output(self, value):
        outvar = self.abstract_value_to_variable[value]
        self.graph.def_output(outvar)


    def lift_value(self, value):
        if isinstance(value, th.Tensor):
            # what happens when the graph encounters a tensor?
            # need to toss it into the "constants" section ?
            assert False
        if isinstance(value, InterpreterValue):
            # or another value?
            assert value.interpreter is self
        return value

def symbolic_trace(func, args):
    interpreter = SymbolicTraceInterpreter()
    dispatcher_singleton.push_interpreter(interpreter)

    # NB: only tensors for now
    values = tuple(interpreter.def_input(arg) for arg in args)
    result = func(*values)
    if isinstance(result, InterpreterValue):
        interpreter.def_output(result)
    else:
        for result in result:
            interpreter.def_output(result)

    dispatcher_singleton.pop_interpreter()
    return interpreter.graph

