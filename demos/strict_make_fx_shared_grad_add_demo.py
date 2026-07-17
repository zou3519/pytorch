import contextlib
import sys
from pathlib import Path

import torch
from torch.fx.experimental.proxy_tensor import make_fx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strict_make_fx_prototype import print_strict_fx, strict_make_fx  # noqa: E402


lib = torch.library.Library("strict_demo", "FRAGMENT")
with contextlib.suppress(RuntimeError):
    lib.define("scaled_residual_fwd(Tensor skip, Tensor branch, Tensor alpha, Tensor beta) -> Tensor")
with contextlib.suppress(RuntimeError):
    lib.define("scaled_residual_bwd(Tensor grad_out, Tensor skip, Tensor branch, Tensor alpha, Tensor beta) -> (Tensor, Tensor)")
with contextlib.suppress(RuntimeError):
    lib.define("swiglu_fwd(Tensor packed) -> Tensor")
with contextlib.suppress(RuntimeError):
    lib.define("swiglu_bwd(Tensor grad_out, Tensor packed) -> Tensor")


class FusedScaledResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, skip, branch, alpha, beta):
        ctx.save_for_backward(skip, branch, alpha, beta)
        return torch.ops.strict_demo.scaled_residual_fwd.default(skip, branch, alpha, beta)

    @staticmethod
    def backward(ctx, grad_out):
        skip, branch, alpha, beta = ctx.saved_tensors
        grad_skip, grad_branch = torch.ops.strict_demo.scaled_residual_bwd.default(grad_out, skip, branch, alpha, beta)
        return grad_skip, grad_branch, None, None


class FusedSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, packed):
        ctx.save_for_backward(packed)
        return torch.ops.strict_demo.swiglu_fwd.default(packed)

    @staticmethod
    def backward(ctx, grad_out):
        (packed,) = ctx.saved_tensors
        return torch.ops.strict_demo.swiglu_bwd.default(grad_out, packed)


def residual_with_fused_boundary(x, w13, w2, alpha, beta):
    packed = torch.nn.functional.linear(x, w13)
    branch = torch.nn.functional.linear(FusedSwiGLU.apply(packed), w2)
    y = FusedScaledResidual.apply(x, branch, alpha, beta)
    (gx,) = torch.autograd.grad(y.sum(), (x,))
    return gx


def residual_with_accumulating_linear_backward(x, w13, w2, alpha, beta):
    packed = torch.nn.functional.linear(x, w13)
    act = torch.ops.strict_demo.swiglu_fwd.default(packed)
    branch = torch.nn.functional.linear(act, w2)
    y = FusedScaledResidual.apply(x, branch, alpha, beta)
    grad_out = torch.ones_like(y)
    grad_skip, grad_branch = torch.ops.strict_demo.scaled_residual_bwd.default(grad_out, x, branch, alpha, beta)
    grad_act = torch.mm(grad_branch, w2)
    grad_packed = torch.ops.strict_demo.swiglu_bwd.default(grad_act, packed)
    return torch.addmm(grad_skip, grad_packed, w13)


@torch.library.impl(lib, "scaled_residual_fwd", "CompositeExplicitAutograd")
def scaled_residual_fwd_impl(skip, branch, alpha, beta):
    return alpha * skip + beta * branch


@torch.library.impl(lib, "scaled_residual_fwd", "Meta")
def scaled_residual_fwd_meta(skip, branch, alpha, beta):
    return torch.empty_strided(skip.shape, skip.stride(), dtype=skip.dtype, device=skip.device)


@torch.library.impl(lib, "scaled_residual_bwd", "CompositeExplicitAutograd")
def scaled_residual_bwd_impl(grad_out, skip, branch, alpha, beta):
    return grad_out * alpha, grad_out * beta


@torch.library.impl(lib, "scaled_residual_bwd", "Meta")
def scaled_residual_bwd_meta(grad_out, skip, branch, alpha, beta):
    return (
        torch.empty_strided(skip.shape, skip.stride(), dtype=skip.dtype, device=skip.device),
        torch.empty_strided(branch.shape, branch.stride(), dtype=branch.dtype, device=branch.device),
    )


@torch.library.impl(lib, "swiglu_fwd", "CompositeExplicitAutograd")
def swiglu_fwd_impl(packed):
    gate, up = packed.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


@torch.library.impl(lib, "swiglu_fwd", "Meta")
def swiglu_fwd_meta(packed):
    shape = (*packed.shape[:-1], packed.shape[-1] // 2)
    return torch.empty_strided(shape, packed.stride(), dtype=packed.dtype, device=packed.device)


@torch.library.impl(lib, "swiglu_bwd", "CompositeExplicitAutograd")
def swiglu_bwd_impl(grad_out, packed):
    gate, up = packed.chunk(2, dim=-1)
    sigmoid = torch.sigmoid(gate)
    grad_gate = grad_out * up * sigmoid * (1 + gate * (1 - sigmoid))
    grad_up = grad_out * torch.nn.functional.silu(gate)
    return torch.cat([grad_gate, grad_up], dim=-1)


@torch.library.impl(lib, "swiglu_bwd", "Meta")
def swiglu_bwd_meta(grad_out, packed):
    return torch.empty_strided(packed.shape, packed.stride(), dtype=packed.dtype, device=packed.device)


def main():
    x = torch.randn(4, 5, requires_grad=True)
    w13 = torch.randn(14, 5)
    w2 = torch.randn(5, 7)
    alpha = torch.rand(5)
    beta = torch.rand(5)

    regular_gm = make_fx(residual_with_fused_boundary)(x, w13, w2, alpha, beta)
    gm = strict_make_fx(residual_with_fused_boundary)(x, w13, w2, alpha, beta)
    fused_gm = strict_make_fx(residual_with_accumulating_linear_backward)(x, w13, w2, alpha, beta)

    print("\n=== regular make_fx: fused residual boundary ===")
    print(regular_gm.code)

    print("\n=== strict_fx: fused residual boundary ===")
    before_text = print_strict_fx(gm)

    print("\n=== strict_fx: add fused into linear backward input grad ===")
    after_text = print_strict_fx(fused_gm)

    assert "strict_demo.scaled_residual_bwd" in before_text
    assert "strict_demo.swiglu_bwd" in before_text
    assert "aten.add.Tensor" in before_text
    assert "aten.addmm.default" in after_text
    assert "aten.add.Tensor" not in after_text
    torch.testing.assert_close(gm(x, w13, w2, alpha, beta), fused_gm(x, w13, w2, alpha, beta))


if __name__ == "__main__":
    main()
