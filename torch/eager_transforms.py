import torch
from torch import vmap
from functools import partial
import collections
import torch.nn as nn
import torch.nn.functional as F
from torch.make_functional import make_functional, make_functional_with_buffers
from torch._vmap_internals import vmap
import gc

# x = torch.ones(2, 3)
# y = torch.ones(2, 3)
# # result = vmap(torch.add)(x, y)
# result = vmap(vmap(torch.add))(x, y)

# assert torch.allclose(result, x + y)

def _create_differentiable(tensor_or_tuple_of_tensors, level=None):
    if isinstance(tensor_or_tuple_of_tensors, torch.Tensor):
        tensor = tensor_or_tuple_of_tensors
        # if tensor.requires_grad:
        #     return tensor
        # assert not tensor.requires_grad
        # # NB: view is needed because autograd is silly.
        # # autograd saved the variable before executing the op, which matters...
        # aliased = tensor.view_as(tensor)
        # return aliased.requires_grad_()
        aliased = torch._wrap_for_grad(tensor, level)
        return aliased.requires_grad_()
    if isinstance(tensor_or_tuple_of_tensors, tuple):
        return tuple(map(partial(_create_differentiable, level=level), tensor_or_tuple_of_tensors))
    if isinstance(tensor_or_tuple_of_tensors, list):
        return tuple(map(partial(_create_differentiable, level=level), tensor_or_tuple_of_tensors))
    raise ValueError(f'Thing passed to transform API must be Tensor, List or Tuple, '
                     f'got {type(tensor_or_tuple_of_tensors)}')

def _undo_create_differentiable(tensor_or_tuple_of_tensors, level=None):
    if isinstance(tensor_or_tuple_of_tensors, torch.Tensor):
        tensor = tensor_or_tuple_of_tensors
        return torch._unwrap_for_grad(tensor, level)
    if isinstance(tensor_or_tuple_of_tensors, tuple):
        return tuple(map(partial(_undo_create_differentiable, level=level), tensor_or_tuple_of_tensors))
    if isinstance(tensor_or_tuple_of_tensors, list):
        return tuple(map(partial(_undo_create_differentiable, level=level), tensor_or_tuple_of_tensors))
    assert False

def _any_differentiable(tensor_or_tuple_of_tensors):
    if isinstance(tensor_or_tuple_of_tensors, torch.Tensor):
        tensor = tensor_or_tuple_of_tensors
        return tensor.requires_grad
    if isinstance(tensor_or_tuple_of_tensors, tuple):
        return any(tuple(map(_any_differentiable, tensor_or_tuple_of_tensors)))
    if isinstance(tensor_or_tuple_of_tensors, list):
        return any(tuple(map(_any_differentiable, tensor_or_tuple_of_tensors)))
    return False


# How do we increment and decrement the nesting? I don't think we can.
def vjp(f, *primals):
    level = torch._C._grad_increment_nesting()
    try:
        diff_primals = _create_differentiable(primals, level)
        primals_out = f(*diff_primals)
        results = _undo_create_differentiable(primals_out, level)

        def wrapper(*cotangents, retain_graph=True, create_graph=True):
            result = torch.autograd.grad(primals_out, diff_primals, cotangents,
                                         retain_graph=retain_graph, create_graph=create_graph)
            return result

    finally:
        torch._C._grad_decrement_nesting()

    return results, wrapper
# 
# 
# def jacrev(f, diff_argnums=(0,)):
#     def wrapper(*args):
#         torch._C._grad_increment_nesting()
#         output = None
#         grad_outputs = None
#         try:
#             args = [_create_differentiable(arg) if i in diff_argnums else arg
#                     for i, arg in enumerate(args)]
#             output = f(*args)
#             # Only support single tensor output for now
#             assert isinstance(output, torch.Tensor)
#             output_numel = output.numel()
#             if output_numel != 0:
#                 grad_output = torch.eye(output_numel).view(output_numel, *output.shape)
# 
#             diff_args = [args[i] for i in diff_argnums]
#             single_diff_arg = isinstance(diff_args[0], torch.Tensor) and len(diff_args) == 1
#             # TODO: quick hack...
#             if len(diff_args) == 1 and isinstance(diff_args[0], tuple):
#                 diff_args = diff_args[0]
#             # NB: need create_graph so that backward pass isn't run in no_grad mode
# 
#             def compute_vjp(v):
#                 return torch.autograd.grad(output, diff_args, v, create_graph=True)
# 
#             if output_numel == 0:
#                 grad_input = compute_vjp(grad_output)
#             else:
#                 grad_input = vmap(compute_vjp)(grad_output)
# 
#             if single_diff_arg:
#                 grad_input = grad_input[0]
#         finally:
#             _undo_create_differentiable(args)
#             torch._C._grad_decrement_nesting()
#         return grad_input, output
#     return wrapper

def grad_with_value(f, diff_argnums=(0,), has_aux=False):
    def wrapper(*args):
        level = torch._C._grad_increment_nesting()
        output, aux, grad_input = None, None, None
        try:
            args = [_create_differentiable(arg, level) if i in diff_argnums else arg
                    for i, arg in enumerate(args)]
            # print("calling f(*args)")
            output = f(*args)
            # print("done with f(*args)")
            if has_aux:
                output, aux = output
            # print("calling output.dim()")
            assert output.dim() == 0
            diff_args = [args[i] for i in diff_argnums]
            single_diff_arg = isinstance(diff_args[0], torch.Tensor) and len(diff_args) == 1
            # TODO: quick hack...
            if len(diff_args) == 1 and isinstance(diff_args[0], tuple):
                diff_args = diff_args[0]
            # NB: need create_graph so that backward pass isn't run in no_grad mode
            # import torchviz; import graphviz
            # graph = torchviz.make_dot(output)
            # graph.save("inner.dot")
            # print("calling autograd.grad")
            grad_input = torch.autograd.grad(
                output, diff_args, create_graph=True)
            # print("done-ish!")
            if single_diff_arg:
                grad_input = grad_input[0]
        finally:
            if grad_input is not None:
                grad_input = _undo_create_differentiable(grad_input, level)
            torch._C._grad_decrement_nesting()
        if has_aux:
            return grad_input, output, aux
        return grad_input, output
    return wrapper

def grad(f, diff_argnums=(0,), has_aux=False):
    def wrapper(*args):
        results = grad_with_value(f, diff_argnums, has_aux=has_aux)(*args)
        if has_aux:
            return results[0], results[2]
        return results[0]
    return wrapper

