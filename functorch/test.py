import torch as th
import api as functorch
from api import vmap, grad, vjp, jvp, jit
from functools import partial

def mse_loss(x, t):
    return functorch.pow(functorch.sub(x, t), th.tensor(2)).sum()

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
grad_sin = grad(functorch.sin)
assert th.allclose(grad_sin(x), functorch.cos(x))
grad_grad_sin = grad(grad(functorch.sin))
assert th.allclose(grad_grad_sin(x), -functorch.sin(x))

# ----------------------------- forward ad things -------------------------------

primal = th.tensor(3.)
tangent = th.tensor(1.)
expected = 1 / primal * tangent
_, dual = jvp(functorch.log, [primal], [tangent])
assert th.allclose(dual, expected)

# ----------------------------- vmap things -------------------------------

# ----------------------------- Tests -------------------------------
x = th.ones(2, 3)
y = th.ones(3)
expected = functorch.add(x, y)
assert th.allclose(expected, th.add(x, y))

result = vmap(functorch.add, in_axes=(0, None))(x, y)
assert th.allclose(result, expected)

x = th.ones(2, 3)
y = th.ones(5, 3)
expected = y.view(5, 1, 3) + x
result = vmap(vmap(functorch.add, in_axes=(0, None)), in_axes=(None, 0))(x, y)
assert th.allclose(result, expected)

x = th.rand(2, 3)
y = th.rand(2, 5, 3)
expected = x.view(2, 1, 3) + y
result = vmap(functorch.add)(x, y)
assert th.allclose(result, expected)

def mse(x, t):
    diff = functorch.sub(x, t)
    result = functorch.pow(diff, th.tensor(2))
    return result

x = th.rand(2, 3)
t = th.rand(3)
expected = mse(x, t)
result = vmap(mse, in_axes=(0, None))(x, t)
assert th.allclose(result, expected)

def mse_loss(x, t):
    return functorch.pow(functorch.sub(x, t), th.tensor(2)).sum()

x = th.rand(2, 3)
t = th.rand(2, 3)
expected = mse(x, t).sum(-1)
result = vmap(mse_loss, in_axes=(0, 0))(x, t)
assert th.allclose(result, expected)

# per-sample-grad
x = th.rand(10)
jac = vmap(grad(functorch.sin), in_axes=(0,))(x)
expected = functorch.cos(x)
assert th.allclose(jac, expected)

x = th.rand(2, 3)
t = th.rand(2, 3)
result = vmap(grad(mse_loss))(x, t)
expected = 2 * (x - t)
assert th.allclose(result, expected)

# matmul tests
x = th.rand(2, 5, 3)
y = th.rand(2, 3, 5)
result = vmap(functorch.matmul)(x, y)
expected = th.einsum('nhi,nij->nhj', x, y)
assert th.allclose(result, expected)

x = th.rand(2, 3)
y = th.rand(2, 3)
result = vmap(functorch.matmul)(x, y)
expected = th.einsum('ni,ni->n', x, y)
assert th.allclose(result, expected)

x = th.rand(2, 3)
y = th.rand(2, 3, 5)
result = vmap(functorch.matmul)(x, y)
expected = th.einsum('ni,nij->nj', x, y)
assert th.allclose(result, expected)

x = th.rand(2, 5, 3)
y = th.rand(2, 3)
result = vmap(functorch.matmul)(x, y)
expected = th.einsum('nhi,ni->nh', x, y)
assert th.allclose(result, expected)

# more per-sample-grads
B = 64
x = th.rand(B, 10)
t = th.rand(B)
weight = th.rand(10, 5)

def model(weight, x):
    x = functorch.matmul(x, weight)
    x = functorch.relu(x)
    return functorch.sum(x)

def loss(weight, x, t):
    y = model(weight, x)
    return mse_loss(y, t)

expected = [grad(loss)(weight, x[i], t[i]) for i in range(B)]
expected = th.stack(expected)
assert expected.shape == th.Size([B, 10, 5])
weight_grad_sample = vmap(grad(loss), in_axes=(None, 0, 0))(weight, x, t)
assert th.allclose(weight_grad_sample, expected)

# def mse(x, y):
#     return (x - y) ** 2
# 
# x = th.rand(10)
# y = th.rand(10)
# 
# graph = symbolic_trace(mse, (x, y))
# expected = """
# def graph(a, b):
#     c = sub(a, b)
#     d = pow(c, 2)
#     return d
# """.strip()
# assert repr(graph) == expected

def mse_loss(x, t):
    return functorch.pow(functorch.sub(x, t), functorch.tensor(2.)).sum()

x = th.rand(2, 3)
t = th.rand(2, 3)
expected = mse_loss(x, y)
result = jit(mse_loss)(x, y)
assert th.allclose(result, expected)

