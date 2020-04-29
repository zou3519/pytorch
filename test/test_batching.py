from torch.testing._internal.common_utils import TestCase, run_tests
import torch
from torch import vmap, Tensor
from torch.autograd import gradcheck
import torch.nn.functional as F
import unittest
from collections import namedtuple
import itertools
import copy

def move_bdim(tensor, from_dim, to_dim):
    return torch.stack(tensor.unbind(from_dim), to_dim)


def add_dim(shape, dim, size):
    result = list(shape)
    result.insert(dim, size)
    return tuple(result)


def maybe_broadcast(thing, size):
    if hasattr(thing, '__iter__'):
        assert len(thing) == size
    return [thing] * size


def add_bdims(bdim_size, bdims, inputs_or_example_shapes):
    return tuple(example_or_shape if bdim is None else add_dim(example_or_shape, bdim, bdim_size)
                 for bdim, example_or_shape in zip(bdims, inputs_or_example_shapes))


def get_inputs(outer_bdim_size, outer_bdims,
               inner_bdim_size, inner_bdims,
               inputs_or_example_shapes, dtypes, device, input_fns):
    if inner_bdim_size is not None:
        inputs_or_example_shapes = add_bdims(inner_bdim_size, inner_bdims, inputs_or_example_shapes)
    inputs_or_shapes = add_bdims(outer_bdim_size, outer_bdims, inputs_or_example_shapes)
    result = []
    for input_or_shape, dtype, input_fn, in zip(inputs_or_shapes, dtypes, input_fns):
        if isinstance(input_or_shape, tuple):
            inp = input_fn(input_or_shape, dtype=dtype, device=device)
        else:
            inp = input_or_shape
        result.append(inp)
    return tuple(result)


def slice_inputs(inputs, bdims, i):
    result = []
    for inp, bdim in zip(inputs, bdims):
        if bdim is None:
            result.append(inp)
        else:
            result.append(inp.select(bdim, i))
    return tuple(result)


def fake_vmap(op, bdims, bdim_size, *inputs):
    return torch.stack([op(*slice_inputs(inputs, bdims, i)) for i in range(bdim_size)])


def fake_vmapvmap(op, outer_bdims, outer_bdim_size,
                  inner_bdims, inner_bdim_size, *inputs):
    results = []
    for outer in range(outer_bdim_size):
        for inner in range(inner_bdim_size):
            examples = slice_inputs(slice_inputs(inputs, outer_bdims, outer), inner_bdims, inner)
            results.append(op(*examples))
    flat_result = torch.stack(results)
    return flat_result.view(outer_bdim_size, inner_bdim_size, *flat_result.shape[1:])


def randp1(shape, dtype, device):
    return torch.rand(shape, dtype=dtype, device=device) + 1


class TestBatching(TestCase):

    def check_vmap(self, op, bdims, inputs_or_example_shapes,
                   bdim_size=3, inplace=False,
                   dtypes=torch.float, device='cpu', input_fns=torch.rand):
        """Tests vmap(op, bdims)(*inputs).

        [NOTE: input generation]
        We generate one input for each element of `inputs_or_example_shapes`.
        For each element:
        - If it is a tuple, then we treat it as an "example shape", add the
          bdims to the shape, and then create a tensor from said shape,
          using the corresponding `input_fn`, device, and dtype.
        - Otherwise, it is used directly as the input.
        """
        num_inputs = len(bdims)
        assert len(inputs_or_example_shapes) == num_inputs
        dtypes = maybe_broadcast(dtypes, num_inputs)
        input_fns = maybe_broadcast(input_fns, num_inputs)

        inputs = get_inputs(bdim_size, bdims,
                            None, None,
                            inputs_or_example_shapes, dtypes, device, input_fns)
        if inplace:
            inputs_clone = copy.deepcopy(inputs)
            output = vmap(op, bdims)(*inputs)
            # NB: The output of an in-place operation is usually the first argument.
            fake_vmap(op, bdims, bdim_size, *inputs_clone)
            expected = inputs_clone[0]
            self.assertEqual(output, expected)
            self.assertEqual(output.data_ptr(), inputs[0].data_ptr())
        else:
            output = vmap(op, bdims)(*inputs)
            expected = fake_vmap(op, bdims, bdim_size, *inputs)
            self.assertEqual(output, expected)

    def check_vmapvmap(self, op, outer_bdims, inner_bdims, inputs_or_example_shapes,
                       outer_bdim_size=3, inner_bdim_size=5, inplace=False,
                       dtypes=torch.float, device='cpu', input_fns=torch.rand):
        """Tests vmap(vmap(op, inner_bdims), outer_bdims)(*inputs).
        See [NOTE: input generation] for how we generate the inputs.
        """
        num_inputs = len(outer_bdims)
        assert len(inner_bdims) == num_inputs
        assert len(inputs_or_example_shapes) == num_inputs
        dtypes = maybe_broadcast(dtypes, num_inputs)
        input_fns = maybe_broadcast(input_fns, num_inputs)

        inputs = get_inputs(outer_bdim_size, outer_bdims,
                            inner_bdim_size, inner_bdims,
                            inputs_or_example_shapes, dtypes, device, input_fns)
        if inplace:
            inputs_clone = copy.deepcopy(inputs)
            output = vmap(vmap(op, inner_bdims), outer_bdims)(*inputs)
            # NB: The output of an in-place operation is usually the first argument.
            fake_vmapvmap(op, outer_bdims, outer_bdim_size,
                          inner_bdims, inner_bdim_size, *inputs_clone)
            expected = inputs_clone[0]
            self.assertEqual(output, expected)
            self.assertEqual(output.data_ptr(), inputs[0].data_ptr())
        else:
            output = vmap(vmap(op, inner_bdims), outer_bdims)(*inputs)
            expected = fake_vmapvmap(op, outer_bdims, outer_bdim_size,
                                     inner_bdims, inner_bdim_size, *inputs)
            self.assertEqual(output, expected)


    def test_defaults(self):
        x23 = torch.randn(2, 3)
        output = vmap(torch.add)(x23, x23)
        self.assertEqual(output, x23 + x23)

        with self.assertRaises(ValueError):
            output = vmap(torch.sum)(x23, 0)

    def test_aligned_broadcasting(self):
        x23 = torch.randn(2, 3)
        x573 = torch.randn(5, 7, 3)
        output = vmap(torch.mul, (0, None))(x23, x573)
        self.assertEqual(output, x23.view(2, 1, 1, 3) * x573)

    def test_double_nest(self):
        x = torch.randn(2, 3)
        result = vmap(vmap(torch.relu, (0,)), (0,))(x)
        self.assertEqual(result, x.relu())

    def test_nested_multiple(self):
        x = torch.rand(2, 3)
        y = torch.rand(2, 3)
        result = vmap(vmap(torch.mul, (0, 0)), (0, 0))(x, y)
        self.assertEqual(result, x * y)

    def test_nested_multiple_not_aligned(self):
        x = torch.rand(5, 2, 3)
        y = torch.rand(3, 5, 2)
        result = vmap(vmap(vmap(torch.mul, (0, 0)), (1, 0)), (1, 2))(x, y)
        self.assertEqual(result, x.permute(1, 2, 0) * y.permute(2, 0, 1))

    def test_fallback_not_aligned(self):
        x = torch.rand(2, 3)
        y = torch.rand(3, 2)
        result = vmap(torch.sub, (0, 1))(x, y)
        self.assertEqual(result, x - y.t())

    def test_nested_multiple_fallback(self):
        x = torch.rand(2, 3)
        y = torch.rand(2, 3)
        result = vmap(vmap(torch.sub, (0, 0)), (0, 0))(x, y)
        self.assertEqual(result, x - y)

    def test_nested_multiple_not_aligned_fallback_simple(self):
        x = torch.rand(2, 3)
        y = torch.rand(3, 2)
        result = vmap(vmap(torch.sub, (0, 0)), (0, 1))(x, y)
        self.assertEqual(result, x - y.t())

    def test_nested_multiple_not_aligned_fallback_complex(self):
        x = torch.rand(5, 2, 3)
        y = torch.rand(3, 5, 2)
        result = vmap(vmap(vmap(torch.sub, (0, 0)), (1, 0)), (1, 2))(x, y)
        self.assertEqual(result, x.permute(1, 2, 0) - y.permute(2, 0, 1))

    def test_nested(self):
        x23 = torch.randn(2, 3)
        x53 = torch.randn(5, 3)
        output = vmap(lambda xx: vmap(lambda yy: torch.add(xx, yy), (0,))(x53), (0,))(x23)
        self.assertEqual(output, x23.view(2, 1, 3) + x53)

    def test_nested_three_layers(self):
        x23 = torch.ones(2, 3)
        x53 = torch.ones(5, 3)
        x73 = torch.ones(7, 3)
        output = (vmap(lambda x:
                       vmap(lambda y:
                            vmap(lambda z:
                                 torch.add(torch.add(x, z), y),
                                 (0,))(x73),
                            (0,))(x53),
                       (0,))(x23))
        expected = x23.view(2, 1, 1, 3) + x53.view(5, 1, 3) + x73
        self.assertEqual(output, expected)

    def test_batched_batched_fallback(self):
        # NB: sub is not implemented. TODO: test fallback warning
        x23 = torch.randn(2, 3)
        output = vmap(torch.sub, (0, 0))(x23, x23)
        self.assertEqual(output, x23 - x23)

    def test_fallback(self):
        # NB: sum is not implemented. TODO: test fallback warning
        x23 = torch.randn(2, 3)
        output = vmap(torch.sum, (0,))(x23)
        self.assertEqual(output, x23.sum(-1))

    def test_independent_output(self):
        x23 = torch.randn(2, 3)
        output = vmap(lambda x: torch.tensor(1.), (0,))(x23)
        self.assertEqual(output, torch.ones(2))

    def test_batched_jacobian(self):
        # TODO: we probably want an API so the user isn't using BatchedTensor directly.
        x3 = torch.randn(3, requires_grad=True)
        y3 = torch.randn(3)
        batched_grad = torch._make_batched(torch.eye(3), 0, 1)
        result = torch.autograd.grad([x3 * y3], [x3], grad_outputs=[batched_grad])
        jacobian = torch._unwrap_batched(result[0], 0)
        self.assertEqual(jacobian, torch.diagflat(y3))

    def test_hessian(self):
        # TODO: we probably want an API so the user isn't using BatchedTensor directly.
        def jacobian_ref(y, x, create_graph=False):
            jac = []
            flat_y = y.reshape(-1)
            grad_y = torch.zeros_like(flat_y)
            for i in range(len(flat_y)):
                grad_y[i] = 1.
                grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True,
                                              create_graph=create_graph)
                jac.append(grad_x.reshape(x.shape))
                grad_y[i] = 0.
            return torch.stack(jac).reshape(y.shape + x.shape)

        def hessian_ref(y, x):
            return jacobian_ref(jacobian_ref(y, x, create_graph=True), x)

        def f(x):
            return x * x * torch.arange(4, dtype=torch.float)

        x = torch.ones(4, requires_grad=True)
        jac = jacobian_ref(f(x), x)
        hes = hessian_ref(f(x), x)

        def jacobian(y, x, create_graph=False):
            flat_y = y.reshape(-1)
            grad_y = torch._make_batched(torch.eye(flat_y.numel()), 0, 1)
            grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True,
                                          create_graph=create_graph)
            grad_x = torch._unwrap_batched(grad_x, 1)
            return grad_x.reshape(y.shape + x.shape)

        def hessian(y, x):
            return jacobian(jacobian(y, x, create_graph=True), x)


        x = torch.ones(4, requires_grad=True)
        jac_bat = jacobian(f(x), x)
        hes_bat = hessian(f(x), x)
        self.assertEqual(jac_bat, jac)
        self.assertEqual(hes_bat, hes)

    def test_batched_batched_inplace_fallback(self):
        inp = torch.randint(0, 1, [2, 3], dtype=torch.bool)
        out = torch.randint(0, 1, [2, 3], dtype=torch.bool)
        expected = torch.logical_xor(inp, out)
        vmap(Tensor.logical_xor_)(out, inp)
        self.assertEqual(out, expected)

    def test_batched_batched_inplace_fallback_unaligned(self):
        inp = torch.randint(0, 1, [2, 3], dtype=torch.bool)
        out = torch.randint(0, 1, [3, 2], dtype=torch.bool)
        expected = torch.logical_xor(inp.t(), out)
        vmap(Tensor.logical_xor_, (1, 0))(out, inp)
        self.assertEqual(out, expected)

    def test_fallback_scalar(self):
        y23 = torch.randn(2, 3)
        out = y23.clone()
        vmap(Tensor.mul_, (0, None))(out, 2)
        self.assertEqual(out, y23 * 2)

    def test_comparison_scalar(self):
        y23 = torch.randn(2, 3)
        out = y23.clone()
        vmap(Tensor.lt_, (0, None))(out, 0)
        self.assertEqual(out, y23 < 0)

    def test_pow_scalar(self):
        y23 = torch.randn(2, 3)
        out = y23.clone()
        vmap(Tensor.pow_, (0, None))(out, 2)
        self.assertEqual(out, y23 ** 2)

    def test_batched_batched_inplace(self):
        y23 = torch.randn(2, 3)
        out = y23.clone()
        vmap(Tensor.mul_)(out, y23)
        self.assertEqual(out, y23 * y23)

    def test_batched_unbatched_inplace(self):
        y23 = torch.randn(2, 3)
        y3 = torch.randn(3)
        out = y23.clone()
        vmap(Tensor.mul_, (0, None))(out, y3)
        self.assertEqual(out, y23 * y3)

    def test_aligned_broadcasting_inplace(self):
        y12 = torch.randn(1, 2)
        y23 = torch.randn(2, 3)
        out = y23.clone()
        vmap(Tensor.mul_, (0, 1))(out, y12)
        self.assertEqual(out, y23 * y12.t())

    def test_nested_inplace(self):
        y573 = torch.randn(5, 7, 3)
        out = y573.clone()
        vmap(vmap(Tensor.mul_))(out, y573)
        self.assertEqual(out, y573 * y573)

    def test_vmap_conv2d(self):
        imgs = torch.randn(7, 3, 5, 5)
        weight = torch.randn(3, 3, 2, 2)
        expected = F.conv2d(imgs, weight)
        output = vmap(F.conv2d, (0, None))(imgs, weight)
        self.assertEqual(output, expected)

        imgs = torch.randn(3, 7, 5, 5)
        weight = torch.randn(3, 3, 2, 2)
        expected = F.conv2d(imgs.transpose(0, 1), weight)
        output = vmap(F.conv2d, (1, None))(imgs, weight)
        self.assertEqual(output, expected)

    def test_vmap_conv2d_two_batch_dims(self):
        y25739 = torch.randn(2, 5, 7, 3, 9)
        weight = torch.randn(13, 7, 2, 2, requires_grad=True)
        bias = torch.randn(13, requires_grad=True)

        output = vmap(F.conv2d, (0, None, None))(y25739, weight, bias)
        expected = F.conv2d(y25739.view(10, 7, 3, 9), weight, bias).view(2, 5, 13, 2, 8)
        self.assertEqual(output, expected)

    def test_vmap_conv2d_batched_weight(self):
        imgs = torch.randn(5, 7, 3, 9)
        weight = torch.randn(3, 13, 7, 2, 2, requires_grad=True)
        bias = torch.randn(13, requires_grad=True)

        output = vmap(F.conv2d, (None, 0, None))(imgs, weight, bias)
        expected = torch.stack([
            F.conv2d(imgs, weight[0], bias),
            F.conv2d(imgs, weight[1], bias),
            F.conv2d(imgs, weight[2], bias),
        ])
        self.assertEqual(output, expected)

    def test_vmap_conv2d_autograd(self):
        imgs = torch.randn(2, 5, 3, 3, 3, dtype=torch.double)
        weight = torch.randn(2, 3, 2, 2, requires_grad=True, dtype=torch.double)
        bias = torch.randn(2, requires_grad=True, dtype=torch.double)
        func = vmap(F.conv2d, (0, None, None))
        gradcheck(func, [imgs, weight, bias])

    def test_vmap_batch_norm(self):
        N, C, H, W = (7, 3, 5, 5)
        B = 2
        imgs = torch.randn(N, C, H, W)
        running_mean = torch.randn(C)
        running_var = torch.randn(C)
        # NB: Using "None" because we're not vectorizing over a dimension.
        output = vmap(F.batch_norm, (None, None, None))(imgs, running_mean, running_var)
        self.assertEqual(output, F.batch_norm(imgs, running_mean, running_var))

        # batchbatchnorm
        imgs = torch.randn(B, N, C, H, W)
        output = vmap(F.batch_norm, (0, None, None))(imgs, running_mean, running_var)
        self.assertEqual(output[0], F.batch_norm(imgs[0], running_mean, running_var))
        self.assertEqual(output[1], F.batch_norm(imgs[1], running_mean, running_var))

    def test_vmap_batch_norm_autograd(self):
        B, N, C, H, W = (5, 3, 2, 1, 1)
        imgs = torch.randn(B, N, C, H, W, requires_grad=True, dtype=torch.double)
        running_mean = torch.zeros(C, dtype=torch.double)
        running_var = torch.ones(C, dtype=torch.double)
        batched_batch_norm = vmap(F.batch_norm, (0, None, None))
        gradcheck(batched_batch_norm, [imgs, running_mean, running_var])

    def test_dropout(self):
        N, C, H, W = (2, 3, 5, 7)
        imgs = torch.randn(N, C, H, W)
        output = vmap(F.dropout, (0, None))(imgs, 1.0)
        self.assertEqual(output, torch.zeros_like(imgs))

    def test_dropout_(self):
        N, C, H, W = (2, 3, 5, 7)
        imgs = torch.randn(N, C, H, W)
        output = vmap(F.dropout, (0, None, None, None))(imgs, 1.0, True, True)
        self.assertEqual(output, torch.zeros_like(imgs))
        self.assertEqual(imgs, torch.zeros_like(imgs))

    def test_clamp_min_(self):
        N, C, H, W = (2, 3, 5, 7)
        imgs = torch.randn(N, C, H, W)
        expected = imgs.clamp_min_(0.5)
        output = vmap(Tensor.clamp_min_, (0, None))(imgs, 0)
        self.assertEqual(output, expected)

    def test_relu(self):
        N, C, H, W = (2, 3, 5, 7)
        imgs = torch.randn(N, C, H, W)
        output = vmap(F.relu, (0,))(imgs)
        self.assertEqual(output, imgs.relu())

    def test_relu_(self):
        N, C, H, W = (2, 3, 5, 7)
        imgs = torch.randn(N, C, H, W)
        expected_result = imgs.relu()
        output = vmap(F.relu, (0, None))(imgs, True)
        self.assertEqual(imgs, expected_result)

    def test_transpose(self):
        x235 = torch.randn(2, 3, 5)
        output = vmap(torch.transpose, (1, None, None))(x235, 0, 1)
        self.assertEqual(output, x235.permute(1, 2, 0))
        self.assertEqual(output.data_ptr(), x235.data_ptr())

        output = vmap(torch.t, (1,))(x235)
        self.assertEqual(output, x235.permute(1, 2, 0))
        self.assertEqual(output.data_ptr(), x235.data_ptr())

    def test_detach(self):
        N, C, H, W = (2, 3, 5, 7)
        imgs = torch.randn(N, C, H, W, requires_grad=True)
        output = vmap(Tensor.detach, (0,))(imgs)
        self.assertEqual(output, imgs)
        self.assertEqual(output.data_ptr(), imgs.data_ptr())
        self.assertFalse(output.requires_grad)

    def test_squeeze(self):
        N, C, H, W = (2, 1, 5, 7)
        imgs = torch.randn(N, C, H, W, requires_grad=True)
        output = vmap(torch.squeeze, (0, None))(imgs, 0)
        self.assertEqual(output, imgs.squeeze(1))
        self.assertEqual(output.data_ptr(), imgs.data_ptr())

    def test_unsqueeze(self):
        N, C, H, W = (2, 3, 5, 7)
        imgs = torch.randn(N, C, H, W, requires_grad=True)
        output = vmap(torch.unsqueeze, (0, None))(imgs, 0)
        self.assertEqual(output, imgs.unsqueeze(1))
        self.assertEqual(output.data_ptr(), imgs.data_ptr())

    def test_permute(self):
        N, C, H, W = (2, 3, 5, 7)
        imgs = torch.randn(N, C, H, W, requires_grad=True)
        output = vmap(Tensor.permute, (0, None))(imgs, [1, 2, 0])
        self.assertEqual(output, imgs.permute(0, 2, 3, 1))
        self.assertEqual(output.data_ptr(), imgs.data_ptr())

    def test_T(self):
        def call_T(x):
            return x.T

        x235 = torch.randn(2, 3, 5)
        output = vmap(call_T, (1,))(x235)
        self.assertEqual(output, x235.permute(1, 2, 0))
        self.assertEqual(output.data_ptr(), x235.data_ptr())

    def test_view(self):
        N, C, H, W = (2, 3, 5, 7)
        imgs = torch.randn(N, C, H, W, requires_grad=True)
        output = vmap(Tensor.reshape, (0, None))(imgs, [3 * 5 * 7])
        self.assertEqual(output, imgs.reshape(2, 3 * 5 * 7))
        self.assertEqual(output.data_ptr(), imgs.data_ptr())
        # TODO: should test some view edge cases

    def test_reshape(self):
        N, C, H, W = (2, 3, 5, 7)
        imgs = torch.randn(N, C, H, W, requires_grad=True)
        output = vmap(Tensor.reshape, (0, None))(imgs, [3 * 5 * 7])
        self.assertEqual(output, imgs.reshape(2, 3 * 5 * 7))
        self.assertEqual(output.data_ptr(), imgs.data_ptr())

    def test_slice(self):
        def slice_fn(self, start, end):
            return self[start:end]

        N, C, H, W = (2, 11, 5, 7)
        imgs = torch.randn(N, C, H, W)
        output = vmap(slice_fn, (0, None, None, None))(imgs, 0, 4)
        self.assertEqual(output, imgs[:, 0:4])
        self.assertEqual(output.data_ptr(), imgs.data_ptr())

    def test_select(self):
        N, C, H, W = (2, 3, 5, 7)
        imgs = torch.randn(N, C, H, W)
        output = vmap(Tensor.select, (0, None, None, None))(imgs, 0, 0)
        self.assertEqual(output, imgs.select(1, 0))
        self.assertEqual(output.data_ptr(), imgs.data_ptr())

    def test_chunk(self):
        N, C, H, W = (2, 12, 5, 7)
        imgs = torch.randn(N, C, H, W)
        output = vmap(torch.chunk, (0, None, None))(imgs, 4, 0)
        expected = imgs.chunk(4, 1)
        self.assertEqual(output, expected)
        self.assertEqual(output[0].data_ptr(), imgs.data_ptr())

    def test_adv_index(self):
        N, C, H, W = (3, 5, 7, 2)
        imgs = torch.randn(N, C, H, W)

        def idx(x):
            return x[[0, 1], [2, 3]]

        output = vmap(idx, (0,))(imgs)
        self.assertEqual(output, imgs[:, [0, 1], [2, 3]])

        output = vmap(idx, (1,))(imgs)
        self.assertEqual(output, imgs.permute(1, 0, 2, 3)[:, [0, 1], [2, 3]])

    def test_vmap_sum(self):
        x235 = torch.randn(2, 3, 5)
        self.assertEqual(vmap(torch.sum, (0, None))(x235, 0), x235.sum(1))
        self.assertEqual(vmap(torch.sum, (0, None))(x235, [0, 1]), x235.sum([1, 2]))
        self.assertEqual(vmap(torch.sum, (1, None))(x235, 0), move_bdim(x235, 0, 1).sum(1))
        self.assertEqual(vmap(torch.sum, (1, None))(x235, 1), move_bdim(x235, 0, 1).sum(2))
        # NB: full-reduce sum is pretty broken. It's a long story.

    def test_vmap_sum_autograd(self):
        x235 = torch.randn(2, 3, 5, requires_grad=True)
        output = vmap(torch.sum, (0, None))(x235, 0)
        grad_output = torch.rand_like(output)
        output.backward(grad_output)
        self.assertEqual(x235.grad, grad_output.view(2, 1, 5).expand(2, 3, 5))

    def _test_binary_pointwise(self, op, input_fn):
        shape = (2, 7)

        # Basic vmap test
        self.check_vmap(op, (0, 2), (shape, shape), input_fns=input_fn)
        self.check_vmap(op, (1, None), (shape, shape), input_fns=input_fn)
        self.check_vmap(op, (None, 1), (shape, shape), input_fns=input_fn)

        # vmap alignment test
        self.check_vmap(op, (1, 1), ((2, 11, 7), (7,)), input_fns=input_fn)

        # Nested vmap test
        self.check_vmapvmap(op, (1, 1), (0, 2), (shape, shape), input_fns=input_fn)
        self.check_vmapvmap(op, (1, None), (None, 1), (shape, shape), input_fns=input_fn)

        # Nested vmap alignment test
        self.check_vmapvmap(op, (1, 1), (0, 1), ((2, 11, 7), (7,)), input_fns=input_fn)

    def test_binary_pointwise(self):
        table = [
            (torch.add, torch.rand),
            (torch.sub, torch.rand),
            (torch.mul, torch.rand),
            (torch.div, randp1),
        ]
        for op, input_fn in table:
            self._test_binary_pointwise(op, input_fn)


def V(name, op, bdims, inputs_or_example_shapes,
      bdim_size=3, inplace=False, dtypes=torch.float, device='cpu',
      input_fns=torch.rand, xfail=False):
    def fn(self):
        return self.check_vmap(op, bdims, inputs_or_example_shapes,
                               bdim_size, inplace, dtypes, device,
                               input_fns)
    fn.__name__ == name
    if xfail:
        fn = unittest.expectedFailure(fn)
    setattr(TestBatching, name, fn)

def VV(name, op, outer_bdims, inner_bdims, inputs_or_example_shapes,
       outer_bdim_size=3, inner_bdim_size=4, inplace=False,
       dtypes=torch.float, device='cpu',
       input_fns=torch.rand, xfail=False):
    def fn(self):
        return self.check_vmapvmap(op, outer_bdims, inner_bdims,
                                   inputs_or_example_shapes,
                                   outer_bdim_size, inner_bdim_size,
                                   inplace, dtypes, device,
                                   input_fns)
    fn.__name__ == name
    if xfail:
        fn = unittest.expectedFailure(fn)
    setattr(TestBatching, name, fn)


coverage_tests = [
    VV('test_vmap_abs0', torch.abs, (1,), (1,), ((2,),)),
    VV('test_vmap_abs1', Tensor.abs, (1,), (1,), ((2,),)),
    VV('test_vmap_abs2', Tensor.abs_, (1,), (1,), ((2,),), inplace=True),
    VV('test_vmap_acos0', torch.acos, (1,), (1,), ((2,),)),
    VV('test_vmap_acos1', Tensor.acos, (1,), (1,), ((2,),)),
    VV('test_vmap_acos2', Tensor.acos_, (1,), (1,), ((2,),), inplace=True),
    V('test_vmap_acos3', Tensor.acos_, (1,), ((2,),), inplace=True),

    V('test_vmap_add_all', torch.add, (0, 2), ((2, 7), (2, 7))),
    V('test_vmap_add_lhs', torch.add, (1, None), ((2, 7), (2, 7))),
    V('test_vmap_add_rhs', torch.add, (None, 1), ((2, 7), (2, 7))),
    V('test_vmap_add_alignment', torch.add, (1, 1), ((2, 7), (7,))),
]


if __name__ == '__main__':
    run_tests()
