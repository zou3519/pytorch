import copy
from dataclasses import dataclass
import sys
from pathlib import Path

import torch
from torch.fx.experimental.proxy_tensor import make_fx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strict_make_fx_prototype import (  # noqa: E402
    MUT_TENSOR,
    print_strict_fx,
    strict_fx_code,
    strict_make_fx,
    ownership_info,
    verify_strict_ownership,
)


def linear_relu_backward_with_preallocated_weight_grad(x, weight, weight_grad):
    weight.grad = weight_grad
    weight.grad.zero_()
    y = torch.nn.functional.linear(x, weight).relu().sum()
    y.backward()
    return weight.grad


def zero_add_pattern(grad, value):
    return grad.zero_().add_(value)


def zero_add_replacement(grad, value):
    return torch.ops.aten.copy_.default(grad, value)


def mm_out_replacement(grad, lhs, rhs):
    return torch.ops.aten.mm.out(lhs, rhs, out=grad)


def eliminate_zero_add_into_mm_out(gm):
    pattern_match = PatternMatch(
        zero_add_pattern,
        zero_add_replacement,
        (torch.empty(5, 3), torch.empty(5, 3)),
        mut_argnums=(0,),
    )
    changed = False
    for match_env in pattern_match.find(gm):
        grad, value = [match_env[p] for p in pattern_match.pattern_placeholders]
        replacement = mm_out_replacement_for(value, grad)
        if replacement is None:
            continue
        pattern_match.replace_one(gm, match_env, *replacement)
        changed = True
    if changed:
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
        verify_strict_ownership(gm)
        gm._code = strict_fx_code(gm)  # type: ignore[attr-defined]
    return gm


def mm_out_replacement_for(value, grad):
    if value.target is not torch.ops.aten.t.default:
        return None
    t_inner = value.args[0]
    if t_inner.target is not torch.ops.aten.t.default:
        return None
    mm = t_inner.args[0]
    if mm.target is not torch.ops.aten.mm.default:
        return None
    if len(value.users) != 1 or len(t_inner.users) != 1 or len(mm.users) != 1:
        return None
    args = (example_like(grad), example_like(mm.args[0]), example_like(mm.args[1]))
    replacement_gm = strict_make_fx(mm_out_replacement, mut_argnums=(0,))(*args)
    return replacement_gm, (grad, mm.args[0], mm.args[1])


def example_like(node):
    val = node.meta["val"]
    return torch.empty_strided(val.shape, val.stride(), dtype=val.dtype, device=val.device)


@dataclass(frozen=True)
class PatternSignature:
    input_kinds: tuple[str, ...]
    output_kind: str
    mutating_inputs: tuple[int, ...]


class PatternMatch:
    def __init__(self, pattern, replacement, example_args, *, mut_argnums):
        self.pattern_gm = strict_make_fx(pattern, mut_argnums=mut_argnums)(*example_args)
        self.replacement_gm = strict_make_fx(replacement, mut_argnums=mut_argnums)(*example_args)
        self.pattern_placeholders = placeholders(self.pattern_gm)
        self.replacement_placeholders = placeholders(self.replacement_gm)
        self.pattern_result = graph_result(self.pattern_gm)
        self.replacement_result = graph_result(self.replacement_gm)
        self.pattern_ops = [
            n for n in self.pattern_gm.graph.nodes if n.op == "call_function"
        ]
        self.validate()

    def validate(self):
        if strict_signature(self.pattern_gm) != strict_signature(self.replacement_gm):
            raise RuntimeError("pattern and replacement have different ownership signatures")

    def find(self, gm):
        for node in list(gm.graph.nodes):
            env = {}
            if not match(node, self.pattern_result, env):
                continue
            matched = {env[p] for p in self.pattern_ops}
            if any(user not in matched for n in matched - {node} for user in n.users):
                continue
            env["_root"] = node
            env["_matched"] = matched
            yield env

    def replace_one(self, gm, env, replacement_gm, replacement_inputs):
        self.validate_replacement(replacement_gm)
        node = env["_root"]
        matched = env["_matched"]
        replacement_placeholders = placeholders(replacement_gm)
        replacement_result = graph_result(replacement_gm)
        repl_env = dict(zip(replacement_placeholders, replacement_inputs))
        with gm.graph.inserting_before(node):
            for repl in replacement_gm.graph.nodes:
                if repl.op in {"placeholder", "output"}:
                    continue
                new = gm.graph.node_copy(repl, lambda n: repl_env[n])
                new.meta = copy.copy(
                    node.meta if repl is replacement_result else repl.meta
                )
                repl_env[repl] = new
        node.replace_all_uses_with(repl_env[replacement_result])
        for old in reversed(list(gm.graph.nodes)):
            if old in matched:
                gm.graph.erase_node(old)

    def validate_replacement(self, replacement_gm):
        pattern_sig = strict_signature(self.pattern_gm)
        replacement_sig = strict_signature(replacement_gm)
        if pattern_sig.output_kind != replacement_sig.output_kind:
            raise RuntimeError("pattern and replacement have different output ownership")
        if pattern_sig.input_kinds[0] != replacement_sig.input_kinds[0]:
            raise RuntimeError("pattern and replacement mutate different input ownership")
        if pattern_sig.mutating_inputs != replacement_sig.mutating_inputs:
            raise RuntimeError("pattern and replacement have different ownership signatures")


def strict_signature(gm):
    inputs = placeholders(gm)
    return PatternSignature(
        input_kinds=tuple(ownership_info(n).kind for n in inputs),
        output_kind=ownership_info(graph_result(gm)).kind,
        mutating_inputs=tuple(
            i for i, n in enumerate(inputs) if ownership_info(n).kind == MUT_TENSOR
        ),
    )


def placeholders(gm):
    return [n for n in gm.graph.nodes if n.op == "placeholder"]


def graph_result(gm):
    return next(n for n in gm.graph.nodes if n.op == "output").args[0]


def match(node, pattern, env):
    if pattern.op == "placeholder":
        previous = env.setdefault(pattern, node)
        return previous is node
    if not isinstance(node, torch.fx.Node):
        return False
    if node.op != pattern.op or node.target is not pattern.target:
        return False
    env[pattern] = node
    return match_value(node.args, pattern.args, env) and match_value(
        node.kwargs,
        pattern.kwargs,
        env,
    )


def match_value(node_value, pattern_value, env):
    if isinstance(pattern_value, torch.fx.Node):
        return match(node_value, pattern_value, env)
    if isinstance(pattern_value, tuple):
        return len(node_value) == len(pattern_value) and all(
            match_value(n, p, env) for n, p in zip(node_value, pattern_value)
        )
    if isinstance(pattern_value, dict):
        return node_value.keys() == pattern_value.keys() and all(
            match_value(node_value[k], p, env) for k, p in pattern_value.items()
        )
    return node_value == pattern_value


def main():
    x = torch.randn(4, 3)
    weight = torch.randn(5, 3, requires_grad=True)
    grad_before = torch.empty_like(weight)
    regular_gm = make_fx(linear_relu_backward_with_preallocated_weight_grad)(
        x,
        weight,
        grad_before,
    )
    gm = strict_make_fx(linear_relu_backward_with_preallocated_weight_grad, mut_argnums=(2,))(
        x,
        weight,
        grad_before,
    )

    print("\n=== regular make_fx ===")
    print(regular_gm.code)

    print("\n=== strict_fx before ===")
    before_text = print_strict_fx(gm)
    with torch.no_grad():
        before = gm(x, weight, grad_before)

    grad_after = torch.empty_like(weight)
    eliminate_zero_add_into_mm_out(gm)

    print("\n=== strict_fx after ===")
    after_text = print_strict_fx(gm)
    with torch.no_grad():
        after = gm(x, weight, grad_after)

    assert "aten.zero_.default" in before_text
    assert "aten.add_.Tensor" in before_text
    assert "aten.zero_.default" not in after_text
    assert "aten.add_.Tensor" not in after_text
    assert "aten.mm.out" in after_text
    assert before.data_ptr() == grad_before.data_ptr()
    assert after.data_ptr() == grad_after.data_ptr()
    torch.testing.assert_close(before, after)


if __name__ == "__main__":
    main()
