from torch.testing._internal.common_utils import TestCase, run_tests
import torch
from torch import vmap, Tensor
from torch.autograd import gradcheck
import torch.nn.functional as F
import unittest


def move_bdim(tensor, from_dim, to_dim):
    return torch.stack(tensor.unbind(from_dim), to_dim)


class TestBatching(TestCase):

    def test_batched_batched(self):
        x23 = torch.randn(2, 3)
        output = vmap(torch.add, [0, 0])(x23, x23)
        self.assertEqual(output, x23 + x23)

    def test_add_0_2(self):
        x2357 = torch.randn(2, 3, 5, 7)
        y3527 = torch.randn(3, 5, 2, 7)
        output = vmap(torch.add, [0, 2])(x2357, y3527)
        self.assertEqual(output, x2357 + move_bdim(y3527, 2, 0))

    def test_batched_unbatched(self):
        x3 = torch.randn(3)
        x23 = torch.randn(2, 3)
        output = vmap(torch.add, [0, None])(x23, x3)
        self.assertEqual(output, x23 + x3)

    def test_aligned_broadcasting(self):
        x23 = torch.randn(2, 3)
        x573 = torch.randn(5, 7, 3)
        output = vmap(torch.mul, [0, None])(x23, x573)
        self.assertEqual(output, x23.view(2, 1, 1, 3) * x573)

    def test_double_nest(self):
        x = torch.randn(2, 3)
        result = vmap(vmap(torch.relu, [0]), [0])(x)
        self.assertEqual(result, x.relu())

    def test_nested_multiple(self):
        x = torch.rand(2, 3)
        y = torch.rand(2, 3)
        result = vmap(vmap(torch.mul, [0, 0]), [0, 0])(x, y)
        self.assertEqual(result, x * y)

    def test_nested_multiple_not_aligned(self):
        x = torch.rand(5, 2, 3)
        y = torch.rand(3, 5, 2)
        result = vmap(vmap(vmap(torch.mul, [0, 0]), [1, 0]), [1, 2])(x, y)
        self.assertEqual(result, x.permute(1, 2, 0) * y.permute(2, 0, 1))

    def test_nested_multiple_fallback(self):
        x = torch.rand(2, 3)
        y = torch.rand(2, 3)
        result = vmap(vmap(torch.sub, [0, 0]), [0, 0])(x, y)
        self.assertEqual(result, x - y)

    def test_nested_multiple_not_aligned_fallback_simple(self):
        x = torch.rand(2, 3)
        y = torch.rand(3, 2)
        result = vmap(vmap(torch.sub, [0, 0]), [0, 1])(x, y)
        self.assertEqual(result, x - y.t())

    def test_nested_multiple_not_aligned_fallback_complex(self):
        x = torch.rand(5, 2, 3)
        y = torch.rand(3, 5, 2)
        result = vmap(vmap(vmap(torch.sub, [0, 0]), [1, 0]), [1, 2])(x, y)
        self.assertEqual(result, x.permute(1, 2, 0) - y.permute(2, 0, 1))

    def test_nested(self):
        x23 = torch.randn(2, 3)
        x53 = torch.randn(5, 3)
        output = vmap(lambda xx: vmap(lambda yy: torch.add(xx, yy), [0])(x53), [0])(x23)
        self.assertEqual(output, x23.view(2, 1, 3) + x53)

    def test_nested_three_layers(self):
        x23 = torch.ones(2, 3)
        x53 = torch.ones(5, 3)
        x73 = torch.ones(7, 3)
        output = (vmap(lambda x:
                       vmap(lambda y:
                            vmap(lambda z:
                                 torch.add(torch.add(x, z), y),
                                 [0])(x73),
                            [0])(x53),
                       [0])(x23))
        expected = x23.view(2, 1, 1, 3) + x53.view(5, 1, 3) + x73
        self.assertEqual(output, expected)

    def test_batched_batched_fallback(self):
        # NB: sub is not implemented. TODO: test fallback warning
        x23 = torch.randn(2, 3)
        output = vmap(torch.sub, [0, 0])(x23, x23)
        self.assertEqual(output, x23 - x23)

    def test_fallback(self):
        # NB: sum is not implemented. TODO: test fallback warning
        x23 = torch.randn(2, 3)
        output = vmap(torch.sum, [0])(x23)
        self.assertEqual(output, x23.sum(-1))

    def test_independent_output(self):
        x23 = torch.randn(2, 3)
        output = vmap(lambda x: torch.tensor(1.), [0])(x23)
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

    @unittest.expectedFailure
    def test_batched_batched_inplace(self):
        y23 = torch.randn(2, 3)
        out = y23.clone()
        vmap(Tensor.mul_, [0, 0])(out, y23)
        self.assertEqual(out, y23 * y23)

    @unittest.expectedFailure
    def test_batched_unbatched_inplace(self):
        y23 = torch.randn(2, 3)
        y3 = torch.randn(3)
        out = y23.clone()
        vmap(Tensor.mul_, [0, None])(out, y3)
        self.assertEqual(out, y23 * y3)

    @unittest.expectedFailure
    def test_aligned_broadcasting_inplace(self):
        y12 = torch.randn(1, 2)
        y23 = torch.randn(2, 3)
        out = y23.clone()
        vmap(Tensor.mul_, [0, 1])(out, y12)
        self.assertEqual(out, y23 * y12.t())

    @unittest.expectedFailure
    def test_nested_inplace(self):
        y573 = torch.randn(5, 7, 3)
        out = y573.clone()
        vmap(vmap(Tensor.mul_, [0, 0]), [0, 0])(out, y573)
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


if __name__ == '__main__':
    run_tests()
