import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx.experimental.shape_prop
import torch.fx as fx
from torch.fx import Node, Proxy, symbolic_trace, Graph, GraphModule
from typing import List, Dict, Tuple
from torch import Tensor
import copy

# Gradient formulas.
# FIXME: There are a bunch of interesting problems with defining grad formulas.
# 1. What do we do with PyTorch ops that are composite w.r.t. autograd?
#    In FX we may need to write a symbolic gradient for *every* operator,
#    but that could quickly become intractable.
# 2. Some ops save intermediate values that are created inside the op for
#    backwards. How do we do that in FX?
#    One example is dropout. The dropout mask created inside dropout MUST
#    be saved for backwards so it can be re-applied.
#
# One potential workaround for both of the above is implement some decomposing
# logic for the autodiff pass to decompose operations like dropout into
# something more workable.
def sum_backward(grad, _, tensor):
    return grad.expand_as(tensor)

def mul_backward(grad, _, tensor, other):
    return grad * other, grad * tensor

def add_backward(grad, _, tensor, other):
    return grad, grad

def sub_backward(grad, _, tensor, other):
    return grad, -grad

def transpose_backward(grad, _, tensor, dim0, dim1):
    return grad.transpose(dim0, dim1)

def embedding_backward(grad, _, weight, input, *args, **kwargs):
    # NB: there is no out-of-place version of index_put_!
    # Also this doesn't support any *args or **kwargs :(
    grad_weight = torch.zeros_like(weight)
    grad_weight.index_put_([input], grad, accumulate=True)
    return grad_weight

def relu_backward(grad, _, tensor, inplace):
    assert not inplace
    return grad * (tensor >= 0)

def dot_backward(grad, _, tensor, other):
    return grad*other, grad*tensor

# TODO: default arg handling
def flatten_backward(grad, _, tensor, start_dim):
    return grad.unflatten(-1, tensor.shape[start_dim:])

def mean_backward(grad, _, tensor, dim):
    return grad.unsqueeze(dim).expand_as(tensor) / tensor.size(dim)

# Since modules are first-class citizens, they need their own gradient formulas
# This is not very ideal because:
# 1. many of our modules are composite (conv1d, conv2d, conv3d)
# 2. We may rely on intermediate values (see Dropout) to compute gradients.
# We can definitely work around (2), but (1) seems important.
def linear_backward(grad, _, tensor, weight, bias):
    # TODO: we're assuming that linear operates on 2D input here.
    return torch.matmul(grad, weight), torch.matmul(grad.t(), tensor), grad.sum(0)
    
    # for 1D input...
    return torch.matmul(grad, weight), torch.matmul(grad.t(), tensor), grad.sum(0)

def linear_backward_no_bias(grad, _, tensor, weight):
    return torch.matmul(grad, weight), torch.matmul(grad.t(), tensor)

def conv2d_backward(grad, _, tensor, weight, bias):
    # grad_weight computation is tricky.
    tensor_unfolded = (
        F.unfold(tensor, weight.shape[2:])
        .unflatten(1, [tensor.shape[1], weight.shape[2] * weight.shape[3]])
    )
    grad_weight = (
        torch.einsum('nckp,ndp->dck', tensor_unfolded, grad.flatten(-2, -1))
        .unflatten(-1, weight.shape[2:])
    )

    return (
        F.conv_transpose2d(grad, weight),
        grad_weight,
        grad.sum([0, 2, 3]),
    )

def log_softmax_backward(grad, result, input, dim, **kwargs):
    # NB: requires result, but I'm not sure why
    # NB: log_softmax_backward_data is not exposed
    return torch.log_softmax_backward_data(grad, result, dim, input)

def nll_loss_backward(grad, _, input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
    # NB: handling all of the following is tricky.
    assert reduce is None
    assert size_average is None
    assert reduction == 'mean'
    assert weight is None

    # NB: Hacked in backawrd operator that also works with no batch :D
    # Avoiding rank dependent control flow...
    batch_size = (input.dim() > 1) * (input.size(0) - 1) + 1
    return torch.nll_loss_bwd(grad, input, target, weight, 1, ignore_index, input.new_full([], batch_size))

def nothing(*args, **kwargs):
    pass

# Register gradient formulas. Feels really hacky to me.
vjp_map = {
    torch.sum: sum_backward,
    torch.mul: mul_backward,
    torch.sub: sub_backward,
    torch.add: add_backward,
    torch.dot: dot_backward,
    torch.mean: mean_backward,
    torch.flatten: flatten_backward,
    F.relu: relu_backward,
    torch.embedding: embedding_backward,
    F.linear: linear_backward,
    F.log_softmax: log_softmax_backward,
    F.nll_loss: nll_loss_backward,
    Tensor.transpose: transpose_backward,
    Tensor.mean: mean_backward,
    torch.transpose: transpose_backward,
    torch.mean: mean_backward,
    torch.ones: nothing,
}

# Register gradient formulae for modules here.
def module_backward_rule(module):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            return linear_backward
        return linear_backward_no_bias
    if isinstance(module, nn.Conv2d):
        return conv2d_backward
    assert False

# Gradient transformation logic
def get_params(module: nn.Module):
    return [f'{name}' for name, _ in module.named_parameters()]

GradDict = Dict[str, Proxy]

def update_grad(grad_dict, key: str, value: Proxy):
    if key not in grad_dict:
        grad_dict[key] = value
    else:
        grad_dict[key] = torch.add(grad_dict[key], value)

def function_backward(node: Node, grad_dict: GradDict):
    grad_fn = vjp_map[node.target]
    args = [Proxy(arg) if isinstance(arg, Node) else arg
            for arg in node.args]
    kwargs = {k: Proxy(v) if isinstance(v, Node) else v
              for k, v in node.kwargs.items()}
    # NB: assume only one output?
    output = Proxy(node)

    # TODO: does fx support nodes with multiple outputs?
    grad_output = grad_dict[node.name]
    assert grad_output is not None

    grad_inputs = grad_fn(grad_output, output, *args, **kwargs)
    if not isinstance(grad_inputs, tuple):
        grad_inputs = (grad_inputs,)

    for grad_inp, argname in zip(grad_inputs, map(lambda arg: arg.name, node.args)):
        update_grad(grad_dict, argname, grad_inp)

    # TODO: need elegant way to handle kwargs and args. Also why is bias a kwarg in FX?
    # TODO: zip doesn't check that everything has same number of elements...
    if grad_fn == linear_backward:
        update_grad(grad_dict, node.kwargs['bias'].name, grad_inputs[-1])

def module_backward(
        owning_module: nn.Module,
        node: Node,
        grad_dict: GradDict,
        leaf_attrs):
    # For each output, pull the grads. NB: assumes every module has a single output.
    grad: Proxy = grad_dict[node.name]

    # Pull the module parameters as proxies
    actual_module = getattr(owning_module, node.target)
    param_names = get_params(actual_module)
    leaf_attrs.update(param_names)
    params: List[Proxy] = [Proxy(node.graph.get_attr(param_name))
                           for param_name in param_names]

    # Perform the backward computation
    module_args = list(map(Proxy, node.args))
    result = module_backward_rule(actual_module)(grad, *module_args, *params)

    names = [arg.node.name for arg in module_args] + list(param_names)
    assert len(names) == len(result)
    for grad, name in zip(result, names):
        update_grad(grad_dict, name, grad)

# def grad(func, only_compute_param_grads=True):
def grad(func, argnums=0):
    gm = symbolic_trace(func)
    grad_graph = Graph()
    val_map = {}
    grad_graph.graph_copy(gm.graph, val_map)
    grad_dict: GradDict = {}
    leaf_attrs = set({})

    # TODO: Is there a way to create a Tensor in FX?
    ones = grad_graph.create_node('call_function', torch.ones, ([],))
    grad_dict[list(grad_graph.nodes)[-2].name] = Proxy(ones)

    # In reverse topological order, update `grad_dict` and `leaf_attrs`.
    nodes = list(grad_graph.nodes)[:-1]  # don't include "ones"
    for node in reversed(nodes):
        if node.op == 'call_function':
            function_backward(node, grad_dict)
        elif node.op == 'call_module':
            raise RuntimeError('NYI')
            module_backward(module, node, grad_dict, leaf_attrs)
        elif node.op == 'call_method':
            raise RuntimeError('NYI')
        else:
            continue

    # Set the outputs, based on argnums
    nodes = list(grad_graph.nodes)
    if isinstance(argnums, int):
        grad_graph.output(grad_dict[nodes[argnum].name].node)
    elif isinstance(argnums, tuple):
        grad_graph.output(tuple(grad_dict[nodes[argnum].name].node for argnum in argnums))
    else:
        raise RuntimeError("NYI")

    # Set the outputs to be either:
    # - [all parameters]
    # - [all inputs] + [all parameters]
    # TODO: This doesn't actually catch all the parameers.
    # output = {leaf: grad_dict[leaf].node for leaf in leaf_attrs}
    # for inp, target in [(node.name, node.target) for node in grad_graph.nodes if node.target in get_params(module)]:
    #     output[target] = grad_dict[inp].node
    # grad_graph.output(output)
    return GraphModule(gm, grad_graph)

import types
import torch.fx
from torch.fx.node import Node
from torch.fx import Tracer

from typing import Dict

class ShapeProp:
    def __init__(self, mod):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())
        self.shape_env = {}

    def propagate(self, *args):
        args_iter = iter(args)
        env : Dict[str, Node] = {}

        def load_arg(a):
            return torch.fx.node.map_arg(a, lambda n: env[n.name])

        def fetch_attr(target : str):
            target_atoms = target.split('.')
            attr_itr = self.mod
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
                attr_itr = getattr(attr_itr, atom)
            return attr_itr

        for node in self.graph.nodes:
            if node.op == 'placeholder':
                result = next(args_iter)
            elif node.op == 'get_attr':
                result = fetch_attr(node.target)
            elif node.op == 'call_function':
                result = node.target(*load_arg(node.args), **load_arg(node.kwargs))
            elif node.op == 'call_method':
                self_obj, *args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                result = getattr(self_obj, node.target)(*args, **kwargs)
            elif node.op == 'call_module':
                result = self.modules[node.target](*load_arg(node.args), **load_arg(node.kwargs))
            elif node.op == 'output':
                return self.shape_env

            if isinstance(result, torch.Tensor):
                self.shape_env[node.name] = result.shape
                node.shape = result.shape
                node.dtype = result.dtype

            env[node.name] = result

        return self.shape_env



def lazify(transform):
    def lazy_transform(model, in_axes):
        if not hasattr(model, 'transforms'):
            model.transforms = []
            model.in_axes = []
        model.transforms.append(transform)
        model.in_axes.append(in_axes)
        def apply_transforms(model, *args, **kwargs):
            shape_prop = ShapeProp(fx.symbolic_trace(model))
            orig_args = args
            for in_axes in model.in_axes:
                orig_args = [a.select(bdim, 0) if bdim is not None else a for a, bdim in zip(orig_args, in_axes)]
            shape_env = shape_prop.propagate(*orig_args)
            old_transforms = model.transforms
            old_axes = model.in_axes
            model = model.transforms[0](model, model.in_axes[0], shape_env)
            model.transforms = old_transforms[1:]
            model.in_axes = old_axes[1:]
            return model

        def new_model_f(model, *args, **kwargs):
            while len(model.transforms) > 0:
                model = apply_transforms(model, *args, **kwargs)

            cur_module = model
            return cur_module(*args, **kwargs)
        model.forward = types.MethodType(new_model_f, model)
        return model
    return lazy_transform


def move_bdim_to_front(x, result_ndim = None):
    x_dim = len(x.shape)
    x_bdim = x.node.bdim
    if x_bdim is None:
        x = torch.unsqueeze(x, 0)
    else:
        x = torch.movedim(x, x_bdim, 0)
    if result_ndim is None:
        return x
    diff = result_ndim - x_dim - (x_bdim is None)
    for _ in range(diff):
        x = torch.unsqueeze(x, 1)
    return x

batching_rules = {}
def gen_binary_op_batching_rule(op):
    def binary_op_batching_rule(a, b):
        a_ndim = len(a.shape)
        b_ndim = len(b.shape)
        result_ndim = max(a_ndim, b_ndim)
        a = move_bdim_to_front(a, result_ndim)
        b = move_bdim_to_front(b, result_ndim)
        res = op(a, b)
        return res, 0
    return binary_op_batching_rule

def unsqueeze_batching_rule(x, dim):
    x = move_bdim_to_front(x)
    if dim >= 0:
        return torch.unsqueeze(x, dim+1), 0
    else:
        return torch.unsqueeze(x, dim), 0

def movedim_batching_rule(x, from_dim, to_dim):
    x = move_bdim_to_front(x)
    return torch.movedim(x, from_dim+1, to_dim+1), 0

batching_rules[torch.mul] = gen_binary_op_batching_rule(torch.mul)
batching_rules[torch.unsqueeze] = unsqueeze_batching_rule
batching_rules[torch.movedim] = movedim_batching_rule

class ShapedProxy(Proxy):
    def __init__(self, i, tracer, shape):
        super().__init__(i, tracer=tracer)
        self.shape = shape


def gen_batching_rule_function(target, shape_env, new_graph, *args):
    vmap_tracer = Tracer()
    vmap_tracer.graph = new_graph
    proxy_args = [ShapedProxy(i, tracer=vmap_tracer, shape=shape_env[i.name]) if isinstance(i, fx.Node) else i for i in args]
    out, bdim = batching_rules[target](*proxy_args)
    out_node = out.node
    out_node.bdim = bdim
    return out_node

def update_args(x, old_name, new_node):
    if x.name == old_name:
        return new_node
    return x

def vmap(model: torch.nn.Module, in_axes, shape_env) -> torch.nn.Module:
    fx_model = fx.symbolic_trace(model)
    new_graph: fx.Graph = fx.Graph()
    new_graph._used_names = fx_model.graph._used_names
    inp_count = 0
    env = {}
    print(fx_model.graph)
    def change_name(node, new_name):
        del new_graph._used_names[node.name]
        node.name = new_name
    for node in fx_model.graph.nodes:
        if node.op == 'placeholder':
            new_node = new_graph.placeholder(node.name)
            change_name(new_node, node.name)
            new_node.bdim = in_axes[inp_count]
            env[node.name] = new_node
            inp_count += 1
        elif node.op == 'call_function':
            new_args = [env[x.name] if isinstance(x, fx.Node) else x for x in node.args]
            if any([x.bdim is not None for x in new_args if isinstance(x, fx.Node)]):
                new_graph._used_names[node.name] = None
                new_node = gen_batching_rule_function(node.target, shape_env, new_graph, *new_args)
            else:
                new_node = new_graph.node_copy(node, lambda x: env[x.name])
                new_node.bdim = None
            change_name(new_node, node.name)
            env[new_node.name] = new_node
        elif node.op == 'output':
            new_graph.output(env[node.args[0].name])

    res = fx.GraphModule(fx_model, new_graph)
    print(res.graph)
    res.graph.lint()
    print(res.code)
    return res

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx


x = torch.randn(3, 5)
y = torch.randn(2)
class M(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a, b):
        return torch.mul(a, b)
vmap_lazy = lazify(vmap)

model = vmap_lazy(vmap_lazy(M(), in_axes=(0, None)), in_axes=(0, None))
print(model(x, y).shape)


# ==================================================

# Utilities to make nn.Module "functional"
# In particular the goal is to be able to provide a function that takes as input
# the parameters and evaluate the nn.Module using fixed inputs.
def _del_nested_attr(obj: nn.Module, names: List[str]) -> None:
    """
    Deletes the attribute specified by the given list of names.
    For example, to delete the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'])
    """
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        _del_nested_attr(getattr(obj, names[0]), names[1:])

def _set_nested_attr(obj: nn.Module, names: List[str], value: Tensor) -> None:
    """
    Set the attribute specified by the given list of names to value.
    For example, to set the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'], value)
    """
    if len(names) == 1:
        setattr(obj, names[0], value)
    else:
        _set_nested_attr(getattr(obj, names[0]), names[1:], value)

def extract_weights(mod: nn.Module) -> Tuple[Tuple[Tensor, ...], List[str]]:
    """
    This function removes all the Parameters from the model and
    return them as a tuple as well as their original attribute names.
    The weights must be re-loaded with `load_weights` before the model
    can be used again.
    Note that this function modifies the model in place and after this
    call, mod.parameters() will be empty.
    """
    orig_params = tuple(mod.parameters())
    # Remove all the parameters in the model
    names = []
    for name, p in list(mod.named_parameters()):
        _del_nested_attr(mod, name.split("."))
        names.append(name)

    # Make params regular Tensors instead of nn.Parameter
    params = tuple(p.detach().requires_grad_() for p in orig_params)
    return params, names

def load_weights(mod: nn.Module, names: List[str], params: Tuple[Tensor, ...]) -> None:
    """
    Reload a set of weights so that `mod` can be used again to perform a forward pass.
    Note that the `params` are regular Tensors (that can have history) and so are left
    as Tensors. This means that mod.parameters() will still be empty after this call.
    """
    for name, p in zip(names, params):
        _set_nested_attr(mod, name.split("."), p)


# ==================================================

def test_vjp_rule(prim, *args, **kwargs):
    result = prim(*args, **kwargs)
    grad = torch.randn_like(result)
    diff_args = (arg for arg in args if torch.is_floating_point(arg))
    expected = torch.autograd.grad(result, diff_args, grad)

    vjp_fun = vjp_map[prim]
    result = vjp_fun(grad, result, *args, **kwargs)
    if isinstance(result, torch.Tensor):
        result = (result,)

    for r, e in zip(result, expected):
        if torch.allclose(r, e):
            continue
        import pdb; pdb.set_trace()

x = torch.randn(4, 3, requires_grad=True)
w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)

test_vjp_rule(F.linear, x, w, b)
test_vjp_rule(torch.mean, x, dim=-1)
test_vjp_rule(F.relu, x, inplace=False)
test_vjp_rule(torch.transpose, x, dim0=-1, dim1=-2)

x = torch.randn(4, 3, requires_grad=True)
test_vjp_rule(F.log_softmax, x, dim=-1)

x = torch.randn(4, 3, requires_grad=True)
t = torch.randint(0, 3, (4,))
test_vjp_rule(F.nll_loss, x, t)

# TODO: get this rule straight
indices = torch.randint(0, 4, (2, 3))
weight = torch.randn(4, 5, requires_grad=True)
test_vjp_rule(torch.embedding, weight, indices)


# =========================== End to end test ==============================

# NB: Assumes the model takes ONE input only
def make_functional(model: nn.Module):
    weights, descriptors = extract_weights(model)

    # NB: Exploded weights is a workaround for "Proxy object doesn't work with *args"
    def fun(w0, w1, w2, w3, w4, data):
        mutable_model = copy.deepcopy(model)
        load_weights(mutable_model, descriptors, [w0, w1, w2, w3, w4])
        return mutable_model(data)

    return weights, fun


class SampleNet(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 16)
        self.fc1 = nn.Linear(16, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.emb(x)
        x = torch.transpose(x, -1, -2)
        x = torch.mean(x, -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def name(self):
        return "SampleNet"


# Create our inputs...
vocab_size = 1000
batch_shape = [64]
words_per_sentence = 5
data = torch.randint(0, vocab_size, (*batch_shape, words_per_sentence))
targets = torch.randint(0, 1, (*batch_shape,))

# Construct our module
net = SampleNet(vocab_size)
criterion = nn.CrossEntropyLoss()

weights, func = make_functional(net)

output = func(*weights, data=data)
loss = criterion(output, targets)

def compute_loss(w0, w1, w2, w3, w4, data, target):
    output = func(w0, w1, w2, w3, w4, data)
    return criterion(output, target)

graph_func = symbolic_trace(compute_loss)
print(graph_func.code)

grad_func = grad(compute_loss, (0, 1, 2, 3, 4))
print(grad_func.code)

result = grad_func(*weights, data=data, target=targets)

loss = compute_loss(*weights, data=data, target=targets)
expected = torch.autograd.grad(loss, weights, allow_unused=True)

for r, e in zip(result, expected):
    if torch.allclose(r, e):
        continue
    import pdb; pdb.set_trace()

print("Everything's OK!")
