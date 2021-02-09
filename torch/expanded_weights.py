import torch
import functools

HANDLED_FUNCTIONS = {}

# ExpandedWeight represents a weight (parameter) Tensor that has an expanded
# batch dimension. Operations on the ExpandedWeight Tensor take advantage of
# how the batch dimension is expanded by de-expanding the weight before
# computation. A subsequent call to .backward() computes gradients for
# ExpandedWeight. Those gradients are equivalent to per-sample-grads for the
# unexpanded weight Tensors.
#
# ExpandedWeight has a fallback that does the forward + backward computation.
# The backward computation is not optimized: it runs torch.autograd.grad in
# a loop. To optimize the backward computation further, we must register
# overrides for specific operators.
#
# This is a __torch_function__ object but it could have also been a Tensor Extension
# with a dispatch key.
class ExpandedWeight(object):
    handled_functions = HANDLED_FUNCTIONS

    def __init__(self, weight, batch_size):
        self.weight = weight
        self.batch_size = batch_size
        self.requires_grad = weight.requires_grad
        self.grad_fn = None

    def __torch_function__(self, func, _, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in self.handled_functions:
            return expanded_weight_fallback(func, *args, **kwargs)
        return self.handled_functions[func](*args)

    @property
    def grad(self):
        return self.weight.grad

    @grad.setter
    def grad(self, value):
        self.weight.grad = value

# Fallback:
# - forward pass: run operation with de-expanded input
# - backward pass: run autograd.grad in a for-loop to compute per-sample-grads.
#                  This is NOT something the vmap API can handle, something more
#                  low-level is at work here.
def expanded_weight_fallback(func, *args, **kwargs):
    # Ugh, hacky kwargs handling for F.linear.
    if kwargs:
        args = args + tuple(kwargs.values())
    unwrapped_args = tuple(arg.weight if isinstance(arg, ExpandedWeight) else arg for arg in args)
    unexpanded_args = tuple(arg.weight[0] if isinstance(arg, ExpandedWeight) else arg for arg in args)
    output = func(*unexpanded_args)
    outputs = output.unbind(0)
    output_detached = output.detach()

    class MyFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args):
            # NB: Doesn't support multiple tensor outputs
            assert isinstance(output, torch.Tensor)
            ctx.unexpanded_args = unexpanded_args
            ctx.outputs = outputs
            return output_detached

        @staticmethod
        def backward(ctx, grad_output):
            unexpanded_args = ctx.unexpanded_args
            diff_args = tuple(arg for arg in unexpanded_args
                              if isinstance(arg, torch.Tensor) and arg.requires_grad)
            batch_size = grad_output.shape[0]
            per_sample_grads = tuple(torch.autograd.grad(ctx.outputs[i], diff_args, grad_output[i],
                                                         retain_graph=(i != batch_size - 1))
                                     for i in range(batch_size))
            per_sample_grads = zip(*per_sample_grads)
            per_sample_grads = (torch.stack(shards) for shards in per_sample_grads)
            result = []
            for arg in args:
                if isinstance(arg, ExpandedWeight):
                    result.append(next(per_sample_grads))
                elif isinstance(arg, torch.Tensor) and arg.requires_grad:
                    result.append(next(per_sample_grads).sum(0))
                else:
                    result.append(None)
            return tuple(result)

    result = MyFunction.apply(*unwrapped_args)
    assert result.requires_grad
    return result

# We can override the fallback by implementing efficient "per-sample-grad" rules
# for the backward pass. The forward pass should still be the same, though.
# TODO: No examples are implemented so far...
def implements_per_sample_grads(torch_function):
    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function] = func
        return func
    return decorator



