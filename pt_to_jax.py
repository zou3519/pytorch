import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import jax
import jax.numpy as jnp
import numpy as np
from foo import make_functional
import torch.fx as fx
from torch.fx import Node, Proxy, symbolic_trace, Graph, GraphModule
from typing import List, Dict, Tuple

# ======================= PT-to-JAX lowering rules ======================

def linear(tensor, weight, bias):
    return jnp.matmul(tensor, jnp.swapaxes(weight, 0, 1)) + bias

def transpose(tensor, dim0, dim1):
    return jnp.swapaxes(tensor, dim0, dim1)

def mean(tensor, dim):
    return jnp.mean(tensor, dim)

def relu(tensor, inplace):
    assert not inplace
    return jax.nn.relu(tensor)

def embedding(weight, input, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    assert padding_idx == -1
    return weight[input]

def log_softmax(x, dim, _stacklevel, dtype):
    assert _stacklevel == 3
    assert dtype is None
    return jax.nn.log_softmax(x, dim)

def nll_loss(x, t, weight, size_average, ignore_index, reduce, reduction):
    assert size_average is None
    assert ignore_index == -100
    assert reduce is None
    assert reduction == 'mean'
    # Lol, using vmap already
    if len(x.shape) == 1:
        return x[t]
    else:
        unreduced_loss = jax.vmap(lambda x, i: x[i])(x, t)
    return jnp.mean(unreduced_loss)

lowering_rules = {
    F.linear: linear,
    Tensor.transpose: transpose,
    torch.transpose: transpose,
    Tensor.mean: mean,
    torch.mean: mean,
    torch.embedding: embedding,
    torch.relu: relu,
    F.relu: relu,
    F.linear: linear,
    F.log_softmax: log_softmax,
    F.nll_loss: nll_loss,
}

# =============================  Some helper functions  ====================

def to_jax(maybe_tensor):
    if isinstance(maybe_tensor, torch.Tensor):
        # TODO: handle requires_grad
        return jnp.array(maybe_tensor.detach().numpy())
    return maybe_tensor

def to_pt(maybe_array):
    return torch.tensor(np.asarray(maybe_array))

def map_maybe_tuple(func, maybe_tuple):
    if isinstance(maybe_tuple, tuple):
        return tuple(map(func, maybe_tuple))
    else:
        return func(maybe_tuple)

def fx_to_jax_interpreter(fx_graph, args, kwargs):
    assert not kwargs  # TODO
    gen_args = iter(args)
    arg_dict = {}

    def retrieve(arg):
        if isinstance(arg, Node):
            return arg_dict[arg.name]
        return arg

    for node in fx_graph.nodes:
        if node.op == 'placeholder':
            arg_dict[node.name] = next(gen_args)
            continue
        if node.op == 'call_function':
            if node.target not in lowering_rules:
                raise RuntimeError(f'NYI: {node.target}')
            jax_fn = lowering_rules[node.target]
            args = tuple(retrieve(arg) for arg in node.args)
            kwargs = {k: retrieve(v) for k, v in node.kwargs.items()}
            result = jax_fn(*args, **kwargs)
            arg_dict[node.name] = result
            continue
        if node.op == 'output':
            # TODO: what does FX do with multiple outputs?
            return tuple(map(retrieve, node.args))[0]
        raise RuntimeError(f'NYI: Opcode {node.op}')
    pass

"""
Given a PyTorch function, JAXFunction lowers it to JAX. It does this by:
1. symbolic tracing `func`
2. When the JAXFunction gets called, we convert input tensors to JAX arrays
3. The JAXFunction interprets the FX graph using "lowering rules" that, for
   each op, calls the appropriate JAX functions.
4. Finally, all JAX array outputs are converted to PyTorch arrays

An interesting question is how this works with transforms. We've defined
vmap, grad, and jit transforms below. They essential work by calling
JAXFunction.transform, which transforms the function stored inside the
JAXFunction.
"""
class JAXFunction:
    def __init__(self, func):
        gm = symbolic_trace(func)

        def wrapped_fn(*args, **kwargs):
            return fx_to_jax_interpreter(gm.graph, args, kwargs)

        self.func = wrapped_fn

    def __call__(self, *args, **kwargs):
        args = tuple(map(to_jax, args))
        kwargs = {k: to_jax(v) for k, v in kwargs.items()}
        result = self.func(*args, **kwargs)
        return map_maybe_tuple(to_pt, result)

    def transform(self, transform_api, *args, **kwargs):
        self.func = transform_api(self.func, *args, **kwargs)
        return self

def vmap(func, *args, **kwargs):
    if not isinstance(func, JAXFunction):
        func = JAXFunction(func)
    return func.transform(jax.vmap, *args, **kwargs)

def grad(func, *args, **kwargs):
    if not isinstance(func, JAXFunction):
        func = JAXFunction(func)
    return func.transform(jax.grad, *args, **kwargs)

def jit(func, *args, **kwargs):
    if not isinstance(func, JAXFunction):
        func = JAXFunction(func)
    return func.transform(jax.jit, *args, **kwargs)

# ===================== Per sample grad computation example ======================


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
targets = torch.randint(0, 2, (*batch_shape,))

# Construct our module
net = SampleNet(vocab_size)
criterion = nn.CrossEntropyLoss()

# Extract the state (weights) from the network.
weights, func = make_functional(net)

def compute_loss(w0, w1, w2, w3, w4, data, target):
    output = func([w0, w1, w2, w3, w4], data)
    result = criterion(output, target)
    return result

grad_loss = grad(compute_loss, (0, 1, 2, 3, 4))
per_sample_grads = vmap(grad_loss, (None,) * 5 + (0, 0))
result = jit(per_sample_grads)(*weights, data, targets)
