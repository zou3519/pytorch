from dataclasses import dataclass
import sys
from pathlib import Path

import torch
import torch.fx as fx
from torch import Tensor
from torch.func import functionalize
from torch.utils._ordered_set import OrderedSet
from torch.utils.checkpoint import CheckpointPolicy
from torch.fx.experimental.proxy_tensor import make_fx
from torch._functorch.partitioners import MinCutOptions, NodeInfo, _size_of, solve_min_cut

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strict_make_fx_prototype import (  # noqa: E402
    BORROW,
    END,
    op_effect,
    print_strict_fx,
    strict_make_fx,
)

CHEAP_FUNCTIONAL_OPS = {
    torch.ops.aten.add.Tensor,
    torch.ops.aten.cos.default,
    torch.ops.aten.expand.default,
    torch.ops.aten.mul.Tensor,
    torch.ops.aten.neg.default,
    torch.ops.aten.sin.default,
    torch.ops.aten.slice.Tensor,
}
CHEAP_OWNERSHIP_OPS = CHEAP_FUNCTIONAL_OPS | {
    torch.ops.aten.add_.Tensor,
    torch.ops.aten.cos_.default,
    torch.ops.aten.sin_.default,
}


@dataclass(frozen=True)
class Plan:
    saved: tuple[str, ...]
    recomputed: tuple[str, ...]
    bytes_saved: int
    recompute_cost: int


def tensor_node(node):
    return isinstance(node, fx.Node) and isinstance(node.meta.get("val"), Tensor)


def tensor_nodes(gm):
    return [n for n in gm.graph.nodes if tensor_node(n)]


def functional_min_cut_partition(gm):
    saved = min_cut_saved_values(gm)
    available, recomputed = replay_plan_from_saved(
        gm,
        joint_graph_demands(gm),
        saved,
        functional_recomputable,
    )
    if not all(n in available for n in joint_graph_demands(gm)):
        raise RuntimeError("functional min-cut picked values that cannot be replayed")
    return make_plan(gm, saved, recomputed)


def strict_fx_min_cut_partition(gm):
    saved = min_cut_saved_values(gm, dont_ban=strict_solver_dont_ban(gm))
    demands = joint_graph_demands(gm)
    plan = replay_plan_from_saved(gm, demands, saved, ownership_recomputable)
    if all(n in plan[0] for n in demands):
        return make_plan(gm, saved, plan[1])
    for node in demands:
        if node not in plan[0]:
            node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
    saved = min_cut_saved_values(gm, dont_ban=strict_solver_dont_ban(gm))
    plan = replay_plan_from_saved(gm, demands, saved, ownership_recomputable)
    return make_plan(gm, saved, plan[1])


def min_cut_saved_values(gm, dont_ban=None):
    node_info = actual_solver_node_info(gm)
    set_dist_from_bw(gm, node_info)
    saved, _ = solve_min_cut(
        gm.graph,
        node_info,
        MinCutOptions(False, False, False, True, True),
        dont_ban=dont_ban or OrderedSet(),
    )
    return [n for n in saved if n not in boundary_nodes(gm, node_info)]


def replay_plan_from_saved(gm, demands, saved, recomputable):
    available = {n for n in tensor_nodes(gm) if n.op == "placeholder"} | set(saved)
    recomputed = []
    recomputed_set = set()
    needed = backward_needed_nodes(tuple(n for n in demands if n not in saved))
    changed = True
    while changed:
        changed = False
        for node in forward_tensor_nodes(gm):
            if node not in needed or node in available:
                continue
            if recomputable(node, available, recomputed_set) and deps_available(node, available):
                available.add(node)
                recomputed.append(node)
                recomputed_set.add(node)
                changed = True
    return available, recomputed


def joint_graph_demands(gm):
    fw_nodes = forward_tensor_nodes(gm)
    bw_nodes = set(tensor_nodes(gm)) - set(fw_nodes)
    return tuple(
        n for n in fw_nodes
        if n.op != "placeholder"
        and n is not forward_output(gm)
        and any(user in bw_nodes for user in n.users)
    )


def backward_needed_nodes(demands):
    needed = set()
    stack = list(demands)
    while stack:
        node = stack.pop()
        if node in needed or not tensor_node(node):
            continue
        needed.add(node)
        stack.extend(node.all_input_nodes)
    return needed


def make_plan(gm, saved, recomputed):
    needed = backward_needed_nodes(tuple(joint_graph_demands(gm)))
    return Plan(
        tuple(n.name for n in saved),
        tuple(n.name for n in recomputed if n in needed and n not in saved),
        sum(_size_of(n) for n in saved),
        sum(
            pointwise_recompute_cost(n)
            for n in recomputed
            if n in needed and n not in saved
        ),
    )


def deps_available(node, available):
    return all(dep in available for dep in node.all_input_nodes if tensor_node(dep))


def forward_output(gm):
    return next(n for n in gm.graph.nodes if n.op == "output").args[0][0]


def forward_tensor_nodes(gm):
    return [n for n in gm.graph.nodes if n in backward_needed_nodes((forward_output(gm),))]


def pointwise_recompute_cost(node):
    if node.target in {torch.ops.aten.sin.default, torch.ops.aten.cos.default}:
        return 1
    if node.target in {torch.ops.aten.sin_.default, torch.ops.aten.cos_.default}:
        return 1
    return 0


def functional_recomputable(node, available, recomputed):
    schema = getattr(node.target, "_schema", None)
    return node.target in CHEAP_FUNCTIONAL_OPS and (schema is None or not schema.is_mutable)


def ownership_recomputable(node, available, recomputed):
    if not all(n in recomputed for n in strict_private_replay_inputs(node)):
        return False
    if node.target in {BORROW, END}:
        return True
    return node.target in CHEAP_OWNERSHIP_OPS


def strict_private_replay_inputs(node):
    if node.target in {BORROW, END} or op_effect(node.target) == "inplace":
        return (node.args[0],)
    return ()


def strict_solver_dont_ban(gm):
    return OrderedSet(
        n for n in forward_tensor_nodes(gm)
        if n.op == "call_function"
        and n.meta.get("recompute") != CheckpointPolicy.MUST_SAVE
        and (n.target in {BORROW, END} or n.target in CHEAP_OWNERSHIP_OPS)
    )


def actual_solver_node_info(gm):
    outs = next(n for n in gm.graph.nodes if n.op == "output").args[0]
    fw = OrderedSet(n for n in gm.graph.nodes if n in backward_needed_nodes((outs[0],)))
    bw = OrderedSet(n for n in gm.graph.nodes if n in backward_needed_nodes(outs[1:]))
    actual_bw = OrderedSet(n for n in bw if n not in fw)
    fw_order = {n: i for i, n in enumerate(fw)}
    inputs = [n for n in gm.graph.nodes if n.op == "placeholder"]
    unclaimed = OrderedSet(n for n in gm.graph.nodes if n not in fw and n not in bw)
    return NodeInfo(inputs, fw, bw, actual_bw, unclaimed, fw_order, OrderedSet())


def set_dist_from_bw(gm, node_info):
    for node in reversed(gm.graph.nodes):
        if node.op == "output":
            node.dist_from_bw = int(1e9)
        elif not node_info.is_required_fw(node):
            node.dist_from_bw = 0
        else:
            node.dist_from_bw = min(
                (user.dist_from_bw + 1 for user in node.users),
                default=int(1e9),
            )


def boundary_nodes(gm, node_info):
    return set(node_info.inputs) | {forward_output(gm)}


def show_plan(title, plan):
    print(f"\n-- {title} plan --")
    print(f"saved:      {plan.saved or '()'}")
    print(f"recomputed: {plan.recomputed or '()'}")
    print(f"cost:       saved_bytes={plan.bytes_saved}, recompute_cost={plan.recompute_cost}")


def show_demands(title, demands):
    print(f"\n-- {title} backward-demanded forward values --")
    print(tuple(n.name for n in demands) or "()")


def strict_joint_graph(f, args, *, mut_argnums=()):
    def joint(*xs):
        y = f(*xs)
        return (y, *torch.autograd.grad(y, xs))

    with torch.autograd._force_original_view_tracking(True):
        return strict_make_fx(joint, mut_argnums=mut_argnums)(*args)


def functional_joint_graph(f, args):
    ff = functionalize(f)

    def joint(*xs):
        y = ff(*xs)
        return (y, *torch.autograd.grad(y, xs))

    with torch.autograd._force_original_view_tracking(True):
        return make_fx(joint)(*args)


def compare(name, f, args, *, mut_argnums=()):
    print(f"\n\n######## {name} ########")
    strict_gm = strict_joint_graph(f, args, mut_argnums=mut_argnums)
    functional_gm = functional_joint_graph(f, args)
    strict_demands = joint_graph_demands(strict_gm)
    functional_demands = joint_graph_demands(functional_gm)

    print("\n-- ownership IR --")
    print_strict_fx(strict_gm)
    show_demands("ownership", strict_demands)
    show_plan(
        "ownership actual min-cut",
        strict_fx_min_cut_partition(strict_gm),
    )

    print("\n-- functionalized IR --")
    print(functional_gm.code)
    show_demands("functional", functional_demands)
    show_plan(
        "functional actual min-cut",
        functional_min_cut_partition(functional_gm),
    )


def plan_names(f, args, *, strict=True):
    gm = strict_joint_graph(f, args) if strict else functional_joint_graph(f, args)
    plan = strict_fx_min_cut_partition(gm) if strict else functional_min_cut_partition(gm)
    return plan.saved, plan.recomputed


def check_expected_plans():
    x = torch.randn(8, requires_grad=True)
    assert plan_names(aot_cos_chain_test, (x,)) == ((), ("cos", "cos_1"))
    assert plan_names(fresh_private_mutation_test, (x,)) == ((), ("add", "sin_"))
    assert plan_names(fresh_private_mutation_test, (x,), strict=False) == ((), ("add", "sin"))
    assert plan_names(view_mutation_test, (x,)) == (("end_borrow",), ())
    assert plan_names(view_mutation_test, (x,), strict=False) == (("slice_scatter",), ())


def aot_recompute_test(a, b):
    return (torch.sin(torch.sin(a)) + b).sum()


def aot_cos_chain_test(x):
    return x.cos().cos().cos().sum()


def view_mutation_test(x):
    z = x.clone()
    y = z[:4]
    y.add_(1)
    return z.cos().sum()


def fresh_private_mutation_test(x):
    y = x + 1
    y.sin_()
    y.cos_()
    return y.sum()


if __name__ == "__main__":
    check_expected_plans()
    compare(
        "AOT test_recompute_partitioning shape",
        aot_recompute_test,
        (torch.randn(4, 4, requires_grad=True), torch.randn(4, 4, requires_grad=True)),
    )
    compare(
        "AOT test_min_cut_partitioner cos chain shape",
        aot_cos_chain_test,
        (torch.randn(8, requires_grad=True),),
    )
    compare(
        "view mutation with autograd view replay",
        view_mutation_test,
        (torch.randn(8, requires_grad=True),),
    )
    compare(
        "fresh private mutation with autograd saves",
        fresh_private_mutation_test,
        (torch.randn(8, requires_grad=True),),
    )
