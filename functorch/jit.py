from core import symbolic_trace, Literal

# maps f, arg_spec -> graph
cache = {}


def maybe_retrieve(f, arg_spec):
    key = (f, arg_spec)
    if key in cache:
        return cache[key]
    return None

def add_to_cache(f, arg_spec, graph):
    key = (f, arg_spec)
    cache[key] = graph

def jit(f):
    def wrapped(*args):
        arg_spec = tuple(arg.size() for arg in args)
        graph = maybe_retrieve(f, arg_spec)
        if graph is None:
            graph = symbolic_trace(f, args)
            add_to_cache(f, arg_spec, graph)

        # Now we interpret the graph...
        # NB: this actually holds on to things even if we've finished using them,
        # :(, probably need some sort of use count checking in the IR
        # Alternatively we can straight up create a python function
        var_map = {var: value for var, value in zip(graph.inputs, args)}

        def read(var):
            if isinstance(var, Literal):
                return var.value
            return var_map[var]

        def write(var, value):
            var_map[var] = value

        for call in graph.function_calls:
            args = tuple(read(var) for var in call.inputs)
            outs = call.prim(*args)
            if len(call.outputs) == 1:
                write(call.outputs[0], outs)
            else:
                for var, out in zip(call.outputs, outs):
                    write(var, out)
        assert len(graph.outputs) == 1
        return read(graph.outputs[0])

    return wrapped

