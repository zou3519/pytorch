# Strict FX Design Notes

The least invasive framing is:

```text
strict ownership is a graph validity property, not necessarily a new IR
```

The graph can remain ordinary FX with a few explicit marker ops and node metadata. The important part is that a validator proves mutation is disciplined enough that most graph passes do not need to reason about arbitrary PyTorch aliasing.

## Contract

A graph satisfies strict ownership semantics if:

- Every tensor value is either read-only or the current mutable representative of some logical storage.
- In-place mutation consumes the previous mutable representative and returns the next one.
- A consumed mutable value is not used again.
- Mutation through a view is represented as a scoped mutable borrow.
- Mutable borrows do not escape and end exactly once.
- Unknown side-effecting ops and metadata mutation are rejected unless they are explicitly modeled.

Example:

```text
%x0: mut Tensor = placeholder
%x1: mut Tensor = aten.add_.Tensor(%x0, 1)
%y:  mut Tensor = aten.mul.Tensor(%x1, 2)
return %y
```

After `%x1` is created, a pass must not insert a new use of `%x0`. The validator catches that as a use of a consumed mutable value.

## View Mutation

View mutation needs explicit structure because the base is suspended while the mutable view borrow is live:

```text
%x0: mut Tensor = placeholder
%b0: &mut Tensor = strict.borrow_mut_view(%x0, slice(0, 0, 4))
%b1: &mut Tensor = aten.add_.Tensor(%b0, 1)
%x1: mut Tensor = strict.end_borrow(%b1)
%z:  mut Tensor = aten.sum.default(%x1)
return %z
```

This makes the ordering auditable without requiring every pass to rediscover that `x[:4]` aliases `x`.

## Normalizer And Validator

The implementation should be split into two roles:

```text
normalizer: inserts ownership marker ops and metadata
validator: checks that the resulting FX graph satisfies the strict ownership contract
```

The normalizer can be best-effort and fail closed. The validator is the part pass authors should trust. If a transform preserves the validated discipline, then mutation is explicit SSA state rather than hidden aliasing behavior.

## Pass-Facing IR

For optimization passes, the most practical surface is probably:

```text
outside functional islands: mutation names an explicit destination
inside functional islands: alias-heavy mutation has been made pure and local
```

Destination-passing examples should stay visible:

```text
%g0 = aten.zero_.default(%grad)
%tmp = aten.mm.default(%a, %b)
%g1 = aten.add_.Tensor(%g0, %tmp)
```

can become:

```text
%g1 = aten.mm.out(%a, %b, out=%grad)
```

and:

```text
%tmp = aten.mm.default(%lhs, %rhs)
%out = aten.add.Tensor(%skip, %tmp)
```

can become:

```text
%out = aten.addmm.default(%skip, %lhs, %rhs)
```

These are destination or accumulation rewrites. Ownership proves the writes are legal, but the rewrite itself is easier to express in terms of destinations and producer fusion.

## Functional Islands

Alias-heavy mutation can be locally functionalized instead of forcing every pass to handle borrow regions:

```text
%x1 = strict.functional_region(%x0):
    %v0 = aten.slice.Tensor(%x0, 0, 0, 4)
    %v1 = aten.add.Tensor(%v0, 1)
    %x1 = aten.slice_scatter.default(%x0, %v1, 0, 0, 4)
    return %x1
```

The island boundary says `%x0` becomes `%x1`. Inside the island, there is no mutation. Lowering can later choose borrow/view update, copy into view, a custom kernel, or `slice_scatter`.

## Memory SSA

Memory SSA is useful when we need precise scheduling around side effects:

```text
%x1, %m1 = aten.add_(%x0, 1) writes(%m0 -> %m1)
%z = aten.sum.default(%x1) reads(%m1)
```

It is strongest when we can prove regions are disjoint:

```text
a = x[:4]
b = x[4:]
a.add_(1)
b.mul_(2)
```

The hard part is region precision. Proving view overlap generally requires stride and storage-offset reasoning, and dynamic shapes make this conservative quickly. So Memory SSA seems better as a later precision tool than as the first pass-facing abstraction.

## Recommendation

Use strict ownership as the validator/legality layer, not as the only abstraction every pass must match on.

The practical stack should be:

```text
1. normalize mutation into explicit ownership/destination form
2. validate strict ownership semantics
3. let common passes match destination-passing and pure functional forms
4. add Memory SSA or region summaries only for passes that need precise reordering
```

The demos in this directory exercise those questions:

- `strict_make_fx_zero_add_elim_demo.py`: rewrites zeroed preallocated grad accumulation into `mm.out`.
- `strict_make_fx_shared_grad_add_demo.py`: shows residual-induced backward accumulation and an `addmm` fused form.
- `strict_make_fx_partition_experiment.py`: compares simple min-cut decisions on strict FX and functionalized FX.
