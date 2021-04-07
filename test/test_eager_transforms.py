from torch.testing._internal.common_utils import TestCase, run_tests
import torch
import torch.nn.functional as F
import functools
import itertools
import warnings
from torch.testing._internal.common_device_type import instantiate_device_type_tests, \
    skipCUDAIfNoMagma
import types
from torch.eager_transforms import grad


class TestGradTransform(TestCase):
    def test_primitive(self):
        x = torch.randn([])
        result = grad(torch.sin)(x)
        self.assertEqual(result, torch.cos(x))

    def test_composite_simple(self):
        x = torch.randn(2, 3, 4)
        result = grad(lambda x: torch.flatten(x).sum())(x)
        self.assertEqual(result, torch.ones_like(x))

    def test_composite_complicated(self):
        x = torch.randn(3)
        y = torch.randn(3, 5)

        def foo(x, y):
            result = x @ y
            return result.sum()

        result = grad(foo)(x, y)

        x.requires_grad_()
        out = foo(x, y)
        expected, = torch.autograd.grad(out, x)

        self.assertEqual(result, expected)

    def _test_attributes(self, get_attr_lambda):
        x = torch.randn(2, 3, 5, dtype=torch.double)
        expected = get_attr_lambda(x)

        def foo(x):
            self.assertEqual(get_attr_lambda(x), expected)
            return x.sum()

        grad(foo)(x)

    def test_shape(self):
        self._test_attributes(lambda x: x.shape)

    def test_dtype(self):
        self._test_attributes(lambda x: x.dtype)

    def test_is_cuda(self):
        self._test_attributes(lambda x: x.is_cuda)

    def test_numel(self):
        self._test_attributes(lambda x: x.numel())

    def test_inplace(self):
        x = torch.randn([])

        def foo(x):
            return x.clone().sin_()

        result = grad(foo)(x)
        self.assertEqual(result, x.cos())

    def test_inplace_on_view(self):
        x = torch.randn(3)

        def foo(x):
            y = x.clone()
            y0 = y[0]
            y0.sin_()
            return y.sum()

        result = grad(foo)(x)

        x.requires_grad_()
        out = foo(x)
        expected, = torch.autograd.grad(out, x)

        self.assertEqual(result, expected)

    def test_inplace_on_view_base(self):
        x = torch.randn(3)

        def foo(x):
            y = x.clone()
            y0 = y[0]
            y.sin_()
            return y0

        result = grad(foo)(x)

        x.requires_grad_()
        out = foo(x)
        expected, = torch.autograd.grad(out, x)

        self.assertEqual(result, expected)

    def test_nesting_simple(self):
        x = torch.randn([])
        result = grad(grad(torch.sin))(x)
        self.assertEqual(result, -torch.sin(x))


if __name__ == '__main__':
    run_tests()
