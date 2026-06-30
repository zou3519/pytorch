import contextlib
import operator
import unittest
from dataclasses import dataclass

import torch
import torch.fx as fx
import torch.utils._pytree as pytree
from torch import Tensor
from torch.fx.experimental.proxy_tensor import (
    _ProxyTensor,
    get_proxy_mode,
    get_proxy_slot,
    has_proxy_slot,
    make_fx,
    set_proxy_slot,
)
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.weak import WeakTensorKeyDictionary


class OwnershipError(RuntimeError):
    pass


@dataclass(frozen=True)
class OwnershipInfo:
    kind: str
    base: fx.Node | None = None
    view_recipe: object | None = None
    borrow_id: str | None = None


lib = torch.library.Library("strict", "FRAGMENT")
with contextlib.suppress(RuntimeError):
    lib.define("borrow_mut_view(Tensor(a) base, str view_recipe) -> Tensor(a)")
with contextlib.suppress(RuntimeError):
    lib.define("end_borrow(Tensor(a) mut_borrow) -> Tensor(a)")

BORROW = torch.ops.strict.borrow_mut_view.default
END = torch.ops.strict.end_borrow.default
MARKERS = {BORROW, END}
TENSOR = "tensor"
MUT_TENSOR = "mut_tensor"
READONLY_VIEW = "readonly_view"
MUT_BORROW_VIEW = "mut_borrow_view"
METADATA_MUTATION = {
    torch.ops.aten.resize_.default,
    torch.ops.aten.set_.default,
    torch.ops.aten.set_.source_Storage,
    torch.ops.aten.set_.source_Storage_storage_offset,
    torch.ops.aten.set_.source_Tensor,
    torch.ops.aten.set_.source_Tensor_storage_offset,
    torch.ops.aten.as_strided_.default,
}


def strict_make_fx(f, *, mut_argnums=(), allow_output_aliasing=False, **make_fx_kwargs):
    mut_argnums = tuple(mut_argnums)

    def strict_f(*args):
        with StrictOwnershipMode(args, mut_argnums):
            return f(*args)

    trace = make_fx(strict_f, **make_fx_kwargs)

    def wrapped(*args):
        check_mut_args_have_unique_ownership(args, mut_argnums)
        gm = trace(*args)
        if gm.graph.eliminate_dead_code():
            gm.graph.lint()
            gm.recompile()

        gm.meta["strict_ownership_signature"] = {
            "mut_argnums": mut_argnums,
            "allow_output_aliasing": allow_output_aliasing,
        }
        check_output_aliasing_invariant(gm, allow_output_aliasing)
        verify_strict_ownership(gm)
        gm._code = strict_fx_code(gm)  # type: ignore[attr-defined]
        return gm

    return wrapped


def print_strict_fx(gm):
    text = strict_fx_code(gm)
    print(text)
    return text


def strict_fx_code(gm):
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    args = ", ".join(
        f"%{n.name}: {ownership_type_str(n.meta.get('strict_ownership'))}"
        for n in placeholders
    )
    lines = [f"def forward({args}):"]

    for n in gm.graph.nodes:
        if n.op == "placeholder":
            continue
        if n.op == "output":
            lines.append(f"    return {format_output(n.args[0])}")
        elif n.op == "call_function":
            args = ", ".join(
                [format_arg(a) for a in n.args]
                + [f"{k}={format_arg(v)}" for k, v in n.kwargs.items()]
            )
            ty = ownership_type_str(n.meta.get("strict_ownership"))
            lines.append(f"    %{n.name}: {ty} = {op_name(n.target)}({args})")
        else:
            ty = ownership_type_str(n.meta.get("strict_ownership"))
            lines.append(f"    %{n.name}: {ty} = {n.op}[{n.target}]")

    text = "\n".join(lines)
    return text


def verify_strict_ownership(gm):
    ends = {}
    for node in gm.graph.nodes:
        if node.op == "call_function":
            verify_call_node(node, ends)
        if node_has_tensor_value(node):
            ownership_info(node)
        if node.op == "output":
            for out in graph_output_nodes(gm):
                if ownership_info(out).kind == MUT_BORROW_VIEW:
                    raise OwnershipError(
                        "mutable borrows cannot escape in graph outputs"
                    )
    for node in gm.graph.nodes:
        if not node_has_tensor_value(node):
            continue
        info = ownership_info(node)
        if info.kind == MUT_BORROW_VIEW and ends.get(info.borrow_id or "", 0) != 1:
            raise OwnershipError("every mutable borrow must end exactly once")
        if node.op == "call_function" and node.target is BORROW:
            end = next(
                (
                    n
                    for n in gm.graph.nodes
                    if n.op == "call_function"
                    and n.target is END
                    and isinstance(n.args[0], fx.Node)
                    and ownership_info(n.args[0]).borrow_id == info.borrow_id
                ),
                None,
            )
            if info.base is None or end is None:
                raise OwnershipError("every mutable borrow must end exactly once")
            for user in info.base.users:
                if user is not node and appears_between(user, node, end):
                    raise OwnershipError(
                        "base mut Tensor is used while an &mut Tensor borrow from it is live."
                    )

    allow = gm.meta.get("strict_ownership_signature", {}).get(
        "allow_output_aliasing", False
    )
    check_output_aliasing_invariant(gm, allow)


class StrictOwnershipMode(TorchDispatchMode):
    def __init__(self, args, mut_argnums):
        self.args = args
        self.mut_argnums = mut_argnums
        self.view_bases = WeakTensorKeyDictionary()
        self.borrow_count = 0

    def __enter__(self):
        out = super().__enter__()
        self.initialize_placeholder_ownership()
        return out

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in MARKERS:
            return func(*args, **kwargs)
        kind = op_effect(func)
        if kind == "metadata":
            raise metadata_mutation_error(func)
        if has_hidden_mutation(func):
            raise hidden_mutation_error(func)
        if kind == "view":
            return self.trace_view(func, args, kwargs)
        if kind == "inplace":
            return self.trace_inplace(func, args, kwargs)
        out = func(*args, **kwargs)
        if kind == "fresh":
            self.attach_ownership_to_tensor_outputs(out, OwnershipInfo(MUT_TENSOR))
        elif kind == "unknown" and pytree.tree_any_only(Tensor, lambda _: True, out):
            raise unknown_op_error(func)
        return out

    def trace_view(self, func, args, kwargs):
        out = func(*args, **kwargs)
        base = args[0]
        if isinstance(base, Tensor) and isinstance(out, Tensor):
            base_node, out_node = (
                self.proxy_node_for_tensor(base),
                self.proxy_node_for_tensor(out),
            )
            if out_node.target is torch.ops.aten.slice.Tensor:
                _, dim, start, end, *rest = out_node.args
                recipe = f"slice:{dim}:{start}:{end}:{rest[0] if rest else 1}"
            elif out_node.target is torch.ops.aten.as_strided.default:
                _, size, stride, *rest = out_node.args
                recipe = f"as_strided:{size}:{stride}:{rest[0] if rest else None}"
            else:
                recipe = str((str(out_node.target), out_node.args[1:], out_node.kwargs))
            out_node.meta["strict_ownership"] = OwnershipInfo(
                READONLY_VIEW,
                base=base_node,
                view_recipe=recipe,
            )
            self.view_bases[out] = base
        return out

    def trace_inplace(self, func, args, kwargs):
        x = mutated_arg_value(func, args, kwargs)
        if not isinstance(x, Tensor):
            raise OwnershipError(f"inplace op {func} does not mutate a tensor")
        info = ownership_info(self.proxy_node_for_tensor(x))
        if info.kind == TENSOR:
            raise self.immutable_tensor_mutation_error(self.proxy_node_for_tensor(x))
        if info.kind == READONLY_VIEW:
            return self.trace_view_mutation(func, args, kwargs, info)
        if info.kind in {MUT_TENSOR, MUT_BORROW_VIEW}:
            return self.trace_mut_tensor_mutation(func, args, kwargs, info)
        raise self.readonly_view_mutation_error(x)

    def trace_view_mutation(self, func, args, kwargs, info):
        view = args[0]
        base = self.view_bases.get(view)
        if (
            base is None
            or info.base is None
            or ownership_info(info.base).kind != MUT_TENSOR
        ):
            raise self.readonly_view_mutation_error(view)
        if str(info.view_recipe).startswith("as_strided"):
            raise OwnershipError(
                "as_strided view mutation is not supported in strict_make_fx MVP."
            )
        out = func(self.emit_mut_borrow_view(base, info), *args[1:], **kwargs)
        self.proxy_node_for_tensor(out).meta["strict_ownership"] = OwnershipInfo(
            MUT_BORROW_VIEW,
            base=info.base,
            view_recipe=info.view_recipe,
            borrow_id=f"borrow{self.borrow_count}",
        )
        self.emit_end_borrow(out, base)
        return out

    def trace_mut_tensor_mutation(self, func, args, kwargs, info):
        out = func(*args, **kwargs)
        self.proxy_node_for_tensor(out).meta["strict_ownership"] = OwnershipInfo(
            info.kind,
            base=info.base,
            view_recipe=info.view_recipe,
            borrow_id=info.borrow_id,
        )
        return out

    def emit_mut_borrow_view(self, base, info):
        self.borrow_count += 1
        out = torch.ops.strict.borrow_mut_view.default(base, info.view_recipe)
        self.proxy_node_for_tensor(out).meta["strict_ownership"] = OwnershipInfo(
            MUT_BORROW_VIEW,
            base=info.base,
            view_recipe=info.view_recipe,
            borrow_id=f"borrow{self.borrow_count}",
        )
        return out

    def emit_end_borrow(self, borrow, base):
        out = torch.ops.strict.end_borrow.default(borrow)
        proxy = get_proxy_slot(out, self.proxy_tracer()).proxy
        proxy.node.meta["strict_ownership"] = OwnershipInfo(MUT_TENSOR)
        set_proxy_slot(base, self.proxy_tracer(), _ProxyTensor(proxy, None))

    def initialize_placeholder_ownership(self):
        tracer = self.proxy_tracer()
        for i, arg in enumerate(self.args):
            for tensor in tensor_leaves(arg):
                if has_proxy_slot(tensor, tracer):
                    kind = MUT_TENSOR if i in self.mut_argnums else TENSOR
                    node = get_proxy_slot(tensor, tracer).proxy.node
                    node.meta["strict_ownership"] = OwnershipInfo(kind=kind)

    def attach_ownership_to_tensor_outputs(self, out, info):
        def mark(t):
            if has_proxy_slot(t, self.proxy_tracer()):
                self.proxy_node_for_tensor(t).meta["strict_ownership"] = info
            return t

        pytree.tree_map_only(Tensor, mark, out)

    def proxy_tracer(self):
        mode = get_proxy_mode()
        if mode is None:
            raise AssertionError("strict ownership mode must run under make_fx")
        return mode.tracer

    def proxy_node_for_tensor(self, tensor):
        return get_proxy_slot(tensor, self.proxy_tracer()).proxy.node

    def readonly_view_mutation_error(self, view):
        info = ownership_info(self.proxy_node_for_tensor(view))
        base = info.base
        if base is None:
            return OwnershipError(
                "cannot mutate readonly view because its base is not a mut Tensor.\n"
                "Only views borrowed from mut Tensor may be mutated."
            )
        return OwnershipError(
            f"cannot mutate view {base.name}[:4] because its base {base.name} "
            "is not a mut Tensor.\n"
            "Only views borrowed from mut Tensor may be mutated."
        )

    def immutable_tensor_mutation_error(self, node):
        name = node.name.removesuffix("_1")
        if node.op == "placeholder":
            return OwnershipError(
                f"cannot mutate input {name} because it is not a mut Tensor.\n"
                "strict_make_fx does not insert clones in strict mode.\n"
                f"Pass mut_argnums=(0,) if the caller guarantees {name} is exclusively owned."
            )
        return OwnershipError(
            "cannot mutate Tensor because it is not a mut Tensor.\n"
            "strict_make_fx does not insert clones in strict mode."
        )


@torch.library.impl(lib, "borrow_mut_view", "CompositeExplicitAutograd")
def borrow_mut_view_impl(base, recipe):
    parts = recipe.split(":")
    if len(parts) != 5 or parts[0] != "slice":
        raise OwnershipError(f"unsupported strict borrow view recipe: {recipe!r}")
    _, dim, start, end, step = (parts[0], *map(int, parts[1:]))
    return torch.ops.aten.slice.Tensor(base, dim, start, end, step)


@torch.library.impl(lib, "borrow_mut_view", "Meta")
def borrow_mut_view_meta(base, recipe):
    return borrow_mut_view_impl(base, recipe)


@torch.library.impl(lib, "end_borrow", "CompositeExplicitAutograd")
def end_borrow_impl(mut_borrow):
    base = getattr(mut_borrow, "_base", None)
    return mut_borrow if base is None else base


@torch.library.impl(lib, "end_borrow", "Meta")
def end_borrow_meta(mut_borrow):
    return mut_borrow


def schema_value_is_tensor(v):
    return "Tensor" in str(v.type)


def alias_sets(v):
    if v.alias_info is None:
        return set()
    return set(v.alias_info.before_set) | set(v.alias_info.after_set)


def returns_mutated_alias(schema):
    args = [
        a
        for a in schema.arguments
        if schema_value_is_tensor(a) and a.alias_info and a.alias_info.is_write
    ]
    rets = [
        r
        for r in schema.returns
        if schema_value_is_tensor(r) and r.alias_info and r.alias_info.is_write
    ]
    return len(args) == len(rets) == 1 and bool(
        alias_sets(args[0]) & alias_sets(rets[0])
    )


def op_effect(target):
    if target is operator.getitem:
        return "fresh"
    if target in METADATA_MUTATION:
        return "metadata"
    schema = getattr(target, "_schema", None)
    if schema is None:
        return "unknown"
    if schema.is_mutable:
        return "inplace" if returns_mutated_alias(schema) else "unknown"
    rets = [r for r in schema.returns if schema_value_is_tensor(r)]
    if not rets:
        return "unknown"
    return "view" if any(r.alias_info for r in rets) else "fresh"


def has_hidden_mutation(target):
    schema = getattr(target, "_schema", None)
    return (
        target not in METADATA_MUTATION
        and schema is not None
        and schema.is_mutable
        and not returns_mutated_alias(schema)
    )


def op_name(target):
    return {BORROW: "strict.borrow_mut_view", END: "strict.end_borrow"}.get(
        target, str(target)
    )


def unknown_op_error(target):
    return OwnershipError(
        f"cannot classify op {op_name(target)}.\n"
        "strict_make_fx requires every op to have an alias/effect schema."
    )


def metadata_mutation_error(target):
    name = target._schema.name.split("::")[-1]
    return OwnershipError(
        f"metadata mutation {name} is not supported in strict_make_fx MVP."
    )


def hidden_mutation_error(target):
    return OwnershipError(
        f"op {op_name(target)} mutates a tensor but does not return the "
        "updated tensor directly."
    )


def verify_call_node(node, ends):
    kind = op_effect(node.target)
    if kind == "metadata":
        raise metadata_mutation_error(node.target)
    if has_hidden_mutation(node.target):
        raise hidden_mutation_error(node.target)
    if node.target not in MARKERS and kind == "unknown" and node_has_tensor_value(node):
        raise unknown_op_error(node.target)
    if kind == "inplace":
        mutated = mutated_arg_node(node)
        if not isinstance(mutated, fx.Node):
            raise OwnershipError(
                f"inplace op {node.target} does not mutate a tensor node"
            )
        if ownership_info(mutated).kind not in {MUT_TENSOR, MUT_BORROW_VIEW}:
            raise OwnershipError(
                "Tensor and &Tensor cannot be passed to inplace mutation ops."
            )
        for user in mutated.users:
            if user is not node and appears_after(user, node):
                raise OwnershipError(
                    f"consumed mut Tensor value {mutated.name} is used after mutation."
                )
    if node.target is END:
        borrow = node.args[0]
        if not isinstance(borrow, fx.Node):
            raise OwnershipError(
                "strict.end_borrow argument must be a mutable borrow node"
            )
        info = ownership_info(borrow)
        if info.kind != MUT_BORROW_VIEW or info.borrow_id is None:
            raise OwnershipError("strict.end_borrow argument must be a mutable borrow")
        ends[info.borrow_id] = ends.get(info.borrow_id, 0) + 1


def mutated_arg_node(node):
    schema = getattr(node.target, "_schema", None)
    if schema is None:
        return node.args[0]
    writes = [
        (i, arg)
        for i, arg in enumerate(schema.arguments)
        if schema_value_is_tensor(arg) and arg.alias_info and arg.alias_info.is_write
    ]
    if len(writes) != 1:
        return node.args[0]
    i, arg = writes[0]
    return node.kwargs[arg.name] if arg.name in node.kwargs else node.args[i]


def mutated_arg_value(target, args, kwargs):
    schema = getattr(target, "_schema", None)
    if schema is None:
        return args[0]
    writes = [
        (i, arg)
        for i, arg in enumerate(schema.arguments)
        if schema_value_is_tensor(arg) and arg.alias_info and arg.alias_info.is_write
    ]
    if len(writes) != 1:
        return args[0]
    i, arg = writes[0]
    return kwargs[arg.name] if arg.name in kwargs else args[i]


def tensor_leaves(x):
    return [v for v in pytree.tree_leaves(x) if isinstance(v, Tensor)]


def shares_storage(a, b):
    return a.untyped_storage()._cdata == b.untyped_storage()._cdata


def check_mut_args_have_unique_ownership(args, mut_argnums):
    flat = tensor_leaves(args)
    by_arg = {i: tensor_leaves(arg) for i, arg in enumerate(args)}
    for i in mut_argnums:
        for mut_tensor in by_arg.get(i, []):
            for j, tensors in by_arg.items():
                for other in tensors:
                    if mut_tensor is other and i == j:
                        continue
                    if mut_tensor is other or shares_storage(mut_tensor, other):
                        raise OwnershipError(
                            "possible aliasing between mut args and other inputs."
                        )
            for other in flat:
                if mut_tensor is not other and shares_storage(mut_tensor, other):
                    raise OwnershipError(
                        "possible aliasing between mut args and other inputs."
                    )


def ownership_info(node):
    info = node.meta.get("strict_ownership")
    if not isinstance(info, OwnershipInfo):
        raise OwnershipError(f"node {node.name} is missing strict ownership metadata")
    return info


def node_has_tensor_value(node):
    return isinstance(node.meta.get("val"), Tensor)


def graph_output_nodes(gm):
    output = next(n for n in gm.graph.nodes if n.op == "output")
    return [n for n in pytree.tree_leaves(output.args[0]) if isinstance(n, fx.Node)]


def ownership_alias_root(node):
    info = ownership_info(node)
    if info.kind in {READONLY_VIEW, MUT_BORROW_VIEW} and info.base is not None:
        return ownership_alias_root(info.base)
    return node


def check_output_aliasing_invariant(gm, allow):
    roots = {}
    for i, node in enumerate(graph_output_nodes(gm)):
        root = ownership_alias_root(node)
        if not allow and root in roots:
            raise OwnershipError(
                f"outputs {roots[root]} and {i} alias.\n"
                "strict_make_fx requires non-aliasing outputs.\n"
                "Return only the base, clone the view manually in user code, "
                "or enable a future allow_output_aliasing mode."
            )
        roots[root] = i


def appears_between(candidate, start, end):
    active = False
    for node in start.graph.nodes:
        if node is start:
            active = True
        elif node is end:
            return False
        elif active and node is candidate:
            return True
    return False


def appears_after(candidate, start):
    seen_start = False
    for node in start.graph.nodes:
        if node is start:
            seen_start = True
        elif seen_start and node is candidate:
            return True
    return False


def ownership_type_str(info):
    names = {
        TENSOR: "Tensor",
        MUT_TENSOR: "mut Tensor",
        READONLY_VIEW: "&Tensor",
        MUT_BORROW_VIEW: "&mut Tensor",
    }
    return "Tensor" if info is None else names[info.kind]


def format_arg(arg):
    if isinstance(arg, fx.Node):
        return f"%{arg.name}"
    if isinstance(arg, str) and arg.startswith("slice:"):
        _, dim, start, end, step = arg.split(":")
        if step == "1":
            return f"slice({dim}, {start}, {end})"
        return f"slice({dim}, {start}, {end}, {step})"
    return repr(arg)


def format_output(arg):
    if isinstance(arg, fx.Node):
        return f"%{arg.name}"
    if isinstance(arg, tuple):
        return "(" + ", ".join(map(format_output, arg)) + ")"
    if isinstance(arg, list):
        return "[" + ", ".join(map(format_output, arg)) + "]"
    return repr(arg)


class TestStrictMakeFxPrototype(unittest.TestCase):
    def test_pure_readonly_input(self):
        def f(x):
            return x * 2

        text = print_strict_fx(strict_make_fx(f)(torch.randn(8)))
        self.assertIn("Tensor", text)
        self.assertIn("mut Tensor = aten.mul.Tensor", text)

    def test_fresh_intermediate_mutation(self):
        def f(x):
            y = x + 1
            y.add_(2)
            return y

        text = print_strict_fx(strict_make_fx(f)(torch.randn(8)))
        self.assertIn("Tensor", text)
        self.assertIn("mut Tensor = aten.add.Tensor", text)
        self.assertIn("mut Tensor = aten.add_.Tensor", text)

    def test_input_mutation_requires_mut_arg(self):
        def f(x):
            x.add_(1)
            return x

        with self.assertRaisesRegex(OwnershipError, "not a mut Tensor"):
            strict_make_fx(f)(torch.randn(8))
        text = print_strict_fx(strict_make_fx(f, mut_argnums=(0,))(torch.randn(8)))
        self.assertIn("mut Tensor", text)
        self.assertIn("aten.add_.Tensor", text)

    def test_view_mutation_uses_mut_borrow(self):
        def f(x):
            y = x[:4]
            y.add_(1)
            return x

        text = print_strict_fx(strict_make_fx(f, mut_argnums=(0,))(torch.randn(8)))
        self.assertIn("mut Tensor", text)
        self.assertIn("&mut Tensor", text)
        self.assertIn("strict.borrow_mut_view", text)
        self.assertIn("strict.end_borrow", text)

    def test_code_and_str_show_ownership(self):
        def f(x):
            y = x[:4]
            y.add_(1)
            return x.sum()

        gm = strict_make_fx(f, mut_argnums=(0,))(torch.randn(8))
        self.assertIn("mut Tensor", gm.code)
        self.assertIn("&mut Tensor", gm.code)
        self.assertIn("strict.borrow_mut_view", gm.code)
        self.assertIn("strict.end_borrow", gm.code)
        self.assertIn("mut Tensor", str(gm))
        self.assertIn("&mut Tensor", str(gm))

    def test_view_mutation_then_base_read(self):
        def f(x):
            y = x[:4]
            y.add_(1)
            return x.sum()

        text = print_strict_fx(strict_make_fx(f, mut_argnums=(0,))(torch.randn(8)))
        self.assertIn("strict.end_borrow", text)
        self.assertIn("aten.sum.default", text)

    def test_view_mutation_requires_mut_base(self):
        def f(x):
            y = x[:4]
            y.add_(1)
            return x

        with self.assertRaisesRegex(OwnershipError, "base .* is not a mut Tensor"):
            strict_make_fx(f)(torch.randn(8))

    def test_output_aliasing_errors(self):
        def f(x):
            y = x[:4]
            return x, y

        with self.assertRaisesRegex(OwnershipError, "outputs 0 and 1 alias"):
            strict_make_fx(f)(torch.randn(8))

    def test_metadata_mutation_errors(self):
        def f(x):
            x.resize_(20)
            return x

        with self.assertRaisesRegex(OwnershipError, "metadata mutation resize_"):
            strict_make_fx(f, mut_argnums=(0,))(torch.randn(8))

    def test_as_strided_view_mutation_errors(self):
        def f(x):
            y = x.as_strided((3, 3), (1, 1))
            y.add_(1)
            return x

        with self.assertRaisesRegex(OwnershipError, "as_strided view mutation"):
            strict_make_fx(f, mut_argnums=(0,))(torch.randn(12))

    def test_mut_args_have_unique_ownership(self):
        def f(x, y):
            x.add_(1)
            return x + y

        x = torch.randn(8)
        with self.assertRaisesRegex(OwnershipError, "possible aliasing"):
            strict_make_fx(f, mut_argnums=(0,))(x, x)

    def test_autograd_saved_clone_before_mutation(self):
        def f(x):
            y = x + 1
            y.sin_()
            y.cos_()
            return y.sum()

        def joint(x):
            y = f(x)
            (gx,) = torch.autograd.grad(y, (x,))
            return y, gx

        text = print_strict_fx(
            strict_make_fx(joint)(torch.randn(8, requires_grad=True))
        )
        self.assertIn("aten.clone.default(%add)", text)
        self.assertIn("aten.clone.default(%sin_)", text)
        self.assertIn("aten.sin_.default(%add)", text)
        self.assertIn("aten.cos_.default(%sin_)", text)


if __name__ == "__main__":
    unittest.main()
