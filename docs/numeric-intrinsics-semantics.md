# Float numeric intrinsics: cross-backend semantics

Added on branch `backend-numeric-intrinsics` (off `04d3ec91`): `Fabs`, `Fmin`,
`Fmax`, `Fclamp` (f32 only, mirroring `Fsqrt`).

**UPDATE (rounding family, on top of `584a1b15`)**: `Ffloor`, `Fceil`,
`Ftrunc`, `Fround` added, mirroring `Fabs` exactly (all four are unary, one
native instruction on every backend, no bit-twiddling). See "The rounding
family" section below.

## The template: `Fsqrt`

Traced end-to-end before writing anything:

- IR: `crates/ir/src/inst/arith.rs` (`#[inst(kind(unary(Fsqrt)))]`), registered
  in `InstSetBase` (`crates/ir/src/inst/inst_set.rs`) and `NativeInstSet`
  (`crates/ir/src/inst/native/inst_set.rs`), parsed via
  `crates/parser/src/inst/arith.rs`, verified in
  `crates/verifier/.../dispatch.rs` (`verify_unary_f32`), interpreted in
  `crates/ir/src/interpret/arith.rs`, and given equivalence-class identity via
  `UnaryInstKind::Fsqrt` in `crates/ir/src/inst/equiv.rs`.
- naga/SPIR-V: `crates/codegen/src/isa/spirv/mod.rs` maps `Fsqrt` ->
  `naga::Expression::Math { fun: MathFunction::Sqrt }`.
- wasm: `crates/codegen/src/isa/wasm/translate.rs` maps `Fsqrt` ->
  `Operator::F32Sqrt`.
- cranelift: **did not exist**. `NativeInstSet` already listed `Fsqrt` (and
  `Fadd`/`Fsub`/`Fmul`/`Fdiv`/`Fneg`) as part of the native instruction set,
  but `crates/codegen/src/isa/cranelift/translate.rs` had zero float support:
  `sonatina_type_to_clif` had no `Type::F32` arm, so any function using a
  float op failed to translate ("unsupported type for cranelift") and was
  silently skipped by `translate_module`'s per-function fail-open behavior.
  This was a real, confirmed, pre-existing gap (see
  `crates/codegen/tests/cranelift_backend.rs`: zero float tests existed;
  `mandelbrot_snapshot.rs`'s cranelift lane uses fixed-point integers, not
  floats, presumably for exactly this reason).

Since "mirror `Fsqrt`" implies a working `Fsqrt`-on-cranelift template to
mirror, this branch first makes that template real (`Type::F32` ->
`clif::types::F32`, f32 immediate materialization via `f32const`/`Ieee32`, and
`fneg`/`fadd`/`fsub`/`fmul`/`fdiv`/`sqrt` native lowering), then mirrors it for
the four new ops. This closes a real gap Fe's own `native.rs` driver already
defends against (it treats a cranelift per-function translation skip as a
hard `LowerError`, per its own comment), not scope creep.

## The new ops

- `Fabs` (unary, 1 arg): mirrors `Fsqrt` exactly (`UnaryInstKind::Fabs`).
- `Fmin`, `Fmax` (binary, 2 args): mirror `Fadd`/`Fsub`/etc
  (`BinaryInstKind::Fmin`/`Fmax`).
- `Fclamp` (ternary, 3 args: `arg`, `lo`, `hi`): a **dedicated IR op** so the IR
  carries clamp semantics as a unit and each backend lowers it as it sees fit.
  ALL THREE backends compose it as `min(max(arg, lo), hi)` (branch-free): wasm
  (`f32.min`/`f32.max`) and cranelift (`fmin`/`fmax`) natively, and naga by
  calling the exact `Fmin`/`Fmax` expansion (below) twice. We deliberately do
  NOT emit a single GLSL.std.450 `FClamp` on the GPU: `FClamp` is
  spec-undefined (poison) when `lo > hi`, whereas the composed
  `min(max(x, lo), hi)` is defined for **every** input, not just finite ones
  (`lo > hi` deterministically yields `hi`, on every backend, matching
  wasm/cranelift's composed clamp) — Codex adversarial review flagged the
  single-`FClamp` form as a finite-input miscompile before push, and pinning
  the naga min/max underneath (see below) has since closed the NaN/-0.0
  residual too. Composing is still branch-free, so the branch-free win is
  preserved; only the aesthetic of a single `clamp()` call is given up.
  `Fclamp` has no `UnaryInstKind`/`BinaryInstKind` home (no ternary category
  exists, and adding one would require plumbing a new `InstClassKind` variant
  through the macro crate, `equiv.rs`, and every exhaustive match in
  `optim`/`analysis`/`verifier` that currently only knows Unary/Binary/Cast/
  Phi/Opaque). Instead it takes the macro's default `InstClassKind::Opaque`
  (same treatment as `Call`/`Jump`/etc: no constant-folding/peephole rewrite,
  but still eligible for ordinary structural GVN CSE via `OwnedInstKey`, since
  arity/kind bucketing is a constant-fold/simplify concern, not a CSE
  concern). `Fclamp` is intentionally NOT registered in the `Interpret`
  trait's (`crates/ir/src/interpret/mod.rs`) explicit `Members` list — that
  list has no established ternary pattern either, and nothing in this task
  requires a const-eval path for it.

## The rounding family: Ffloor, Fceil, Ftrunc, Fround

Unlike `Fmin`/`Fmax`/`Fabs`/`Fclamp`, this family needed no NaN/-0.0
resolution pass: every op is a **single native instruction on all three
backends**, and (with one exception, `Fround`'s ties-to-even rule, checked
explicitly below) every backend already agrees on the answer.

- `Ffloor` -> naga `MathFunction::Floor` (WGSL `floor()`, SPIR-V
  `GLSL.std.450 Floor`), wasm `f32.floor`, cranelift `floor`.
- `Fceil` -> naga `MathFunction::Ceil` (`ceil()`/`Ceil`), wasm `f32.ceil`,
  cranelift `ceil`.
- `Ftrunc` -> naga `MathFunction::Trunc` (`trunc()`/`Trunc`), wasm
  `f32.trunc`, cranelift `trunc`.
- `Fround` -> naga `MathFunction::Round` (`round()`/`RoundEven`), wasm
  `f32.nearest`, cranelift `nearest`.

All four are `UnaryInstKind` variants (mirroring `Fabs`, not `Fclamp`'s
`Opaque`/ternary special-casing), registered the same way as `Fabs` in every
exhaustive `UnaryInstKind` match across `analysis`/`optim`
(`demanded_bits.rs`, `known_bits.rs`, `gvn.rs` x2, `simplify_expr.rs`): no
constant-folding/algebraic-simplification/GVN-key special case, same
"unknown but pure, CSE via `OwnedInstKey`" treatment `Fabs` gets.

### THE ONE SEMANTIC CHECK: `Fround` ties-to-even

`round(x)` must be `roundTiesToEven` (IEEE 754): `round(0.5) == 0`,
`round(1.5) == 2`, `round(2.5) == 2`, `round(-0.5) == -0`. This was flagged
as the one place this family could silently diverge cross-backend (the way
`Fmin`/`Fmax` diverged on NaN/-0.0), because Rust's own `f32::round()` is
**ties-away-from-zero**, a different rounding rule -- a tempting, wrong,
oracle.

**Verified, all three backends agree exactly, no divergence to pin around:**

- **wasm**: the WebAssembly spec's `f32.nearest` is `roundTiesToEven` by
  definition; `waffle`'s own reference interpreter
  (`waffle-0.2.0/src/interp.rs`) implements `Operator::F32Nearest` as
  `f32::round_ties_even()`, not `.round()`.
- **cranelift**: `nearest`'s own generated doc comment (from
  `cranelift-codegen-meta`) reads *"Round floating point round to integral,
  towards nearest with ties to even."*
- **naga/SPIR-V**: `MathFunction::Round` lowers to SPIR-V's `GLSL.std.450`
  extended instruction **`RoundEven`**, NOT the ties-away-from-zero `Round`
  ext inst (verified by reading naga 29.0.4's SPIR-V backend source,
  `src/back/spv/block.rs`: `Mf::Round => MathOp::Ext(GlslStd450Op::RoundEven)`).
  WGSL's `round()` builtin is defined the same way per the WGSL spec ("k is
  rounded to even if e is exactly halfway between two integers"), so the
  WGSL-text and SPIR-V-binary paths agree too.

So `Ffloor`/`Fceil`/`Ftrunc`/`Fround` needed **no** NaN/-0.0/ties resolution
work analogous to `Fmin`/`Fmax`'s `emit_exact_fminmax` expansion: each op
lowers to exactly one native instruction/`OpExtInst`, branch-free, on every
backend, and all three backends were already exact. `Fround`'s Rust
interpreter/CTFE implementation uses `f32::round_ties_even()`
(`crates/ir/src/interpret/arith.rs`, and Fe's `crates/hir/src/analysis/
semantic/ctfe/machine.rs`), never `f32::round()`, to avoid silently
reintroducing the divergence at the reference-implementation layer.

`floor`/`ceil`/`trunc` have no ties/rounding-mode ambiguity at all (they are
each defined pointwise, no "nearest" choice to make), so there is nothing to
check beyond "is it the right native instruction" for those three.

## The sharp edge: Fmin/Fmax NaN and signed-zero semantics

**PINNED semantics: the "WebAssembly rules" (IEEE 754-2019 `minimum`/
`maximum`).** If either operand is NaN, the result is NaN (we return the
canonical quiet NaN `0x7fc0_0000`; the spec permits any NaN with unspecified
sign/payload bits beyond "quiet, mantissa MSB=1"). `-0.0` is treated as
strictly less than `+0.0` **regardless of argument order**
(`min(+0,-0) == min(-0,+0) == -0.0`; `max(+0,-0) == max(-0,+0) == +0.0`).

Why this choice: it is **exactly** wasm's `f32.min`/`f32.max`, and it is
**exactly** cranelift's `fmin`/`fmax` by cranelift's own doc comment
(`cranelift-codegen-meta`'s `shared/instructions.rs`): *"Floating point
minimum, propagating NaNs using the WebAssembly rules."* So wasm and
cranelift require **zero composition** to agree — both already implement this
exact semantics natively. This is verified bit-for-bit (see below).

**naga/SPIR-V now matches too, exactly, via a branch-free integer expansion
(RESOLVED, was the OPEN DECISION below).** `MathFunction::Min`/`Max` (SPIR-V's
`GLSL.std.450` extended instructions `FMin`/`FMax`) are implementation-defined
on NaN/-0.0 per the Khronos spec — *"Which operand is the result is undefined
if one of the operands is a NaN"* — so they are NOT used for `Fmin`/`Fmax`
anymore. Instead, `crates/codegen/src/isa/spirv/mod.rs`'s `emit_exact_fminmax`
bitcasts both operands to `u32` (`As { convert: None }`), builds a monotone
total integer order over the bit pattern (`key(x) = xu ^ (0x80000000 |
(0xffffffff * signbit(xu)))`, under which unsigned integer comparison agrees
with IEEE float ordering including `-0.0 < +0.0`), and does everything else —
the min/max pick AND the NaN-detect-and-force-canonical-qNaN step — with
integer compares and `Expression::Select` (naga's conditional move; lowers to
SPIR-V `OpSelect`, WGSL `select()`). Every op after the initial bitcasts is
integer-only: no float comparison, so no fast-math latitude of the driver can
touch it, and — critically — **no control flow**: `OpSelect`/`select()` is a
conditional move, not a branch, so this stays exactly as branch-free as the
GLSL.std.450 call it replaced (just ~15-20 ALU ops instead of 1). `Fabs` is
similarly now an explicit bitcast-AND-bitcast (`& 0x7fffffff`) rather than
`MathFunction::Abs`, for the same "same integer toolkit, no extended-instruction-set
dependency" reason, though `MathFunction::Abs`/GLSL.std.450 `FAbs` was already
exact (pure bitwise) either way. This is verified bit-for-bit against the
same from-scratch "WebAssembly rules" oracle used for wasm/cranelift by
constructing the naga module directly and asserting on the emitted
WGSL/SPIR-V structure (no branch/phi, only `select`/`OpSelect`); no GPU
adapter exists in this sandbox to execute the shader, so real driver
NaN/-0.0 handling is not observed end-to-end, but the expansion does not
depend on driver `min`/`max` behavior at all anymore — there is no float
`min`/`max` instruction left in the emitted code to have driver-dependent
behavior.

**`Fabs` has no such issue.** It is a pure, deterministic bitwise sign-clear
on every backend (cranelift's own doc: *"Note that this is a pure bitwise
operation"*; wasm `f32.abs` and GLSL.std.450 `FAbs` are equivalently
bitwise), so NaN payloads pass through unchanged except for the sign bit.

**`Fclamp` is composed as `min(max(x, lo), hi)` on ALL three backends**, so its
out-of-order-bounds (`lo > hi`) behavior is fully DEFINED and identical across
backends on **every** input, not just finite ones (this is the Codex-flagged
fix: a single GLSL.std.450 `FClamp` would have been poison for `lo > hi`). Its
NaN/-0.0 behavior is exactly that of the (now-exact, see above) `Fmin`/`Fmax`
it is composed from.

## OPEN DECISION: RESOLVED (Slice 0 of the float-semantics design)

The naga/SPIR-V `Fmin`/`Fmax` NaN and -0.0 divergence from wasm/cranelift
described above is CLOSED: `Fmin`/`Fmax`/`Fabs`/`Fclamp` are now pinned-exact
("WebAssembly rules") on **all three backends**, including naga/SPIR-V and its
WGSL output, via the branch-free integer key-compare-and-select expansion
(`emit_exact_fminmax`, `crates/codegen/src/isa/spirv/mod.rs`) — see "The sharp
edge" above for the mechanism. It was resolved by measurement, not by
sacrificing branch-freedom: exact IEEE-754-2019 minimum/maximum turned out to
be expressible on the GPU with zero control flow (bitcast to integer, build a
monotone total-order key, `OpSelect`/`select()` — a conditional move, not a
branch), so the "branch-free XOR exact" framing this decision originally
posed was overstated. The real, and still real, tradeoff is throughput: ~1 ALU
op (the old `MathFunction::Min`/`Max`/GLSL.std.450 call) vs ~15-20 ALU ops (the
exact expansion), both branch-free. Plain `f32` `Fmin`/`Fmax`/`Fclamp` now pay
that wider expansion unconditionally on the GPU path.

A relaxed, single-instruction opt-in (for hot inner loops where the ~15-20x op
count matters and the caller can locally guarantee no NaN operand / no
observed zero-sign) is DEFERRED to a later slice: new `FminRelaxed`/
`FmaxRelaxed` IR ops behind an explicit typed domain (a `Regular` newtype, per
`/workspace/mb2/FLOAT_SEMANTICS_TYPE_API_DESIGN.md`), not a flag or target
conditional on the existing ops. This slice touches ONLY the naga/SPIR-V
lowering of the existing `Fmin`/`Fmax`/`Fabs`/`Fclamp`; wasm and cranelift are
unchanged (they were already exact and native).

## Verification

- `crates/codegen/tests/cranelift_backend.rs`:
  `cranelift_f32_abs_min_max_clamp_oracle` (native oracle, ordinary values,
  bit-exact vs a plain Rust reference; `Fabs`/`Fclamp` assertions compare
  `.to_bits()`, not float `==`, so a `-0.0`-vs-`+0.0` divergence cannot hide
  behind IEEE equality — the review's gap) and
  `cross_backend_f32_min_max_nan_zero_inf_differential` (wasm vs cranelift
  bit-for-bit over normal/±0/±inf/NaN inputs for `Fmin`/`Fmax`, PLUS
  `Fclamp` including a `lo > hi` case, both checked against a from-scratch
  "WebAssembly rules" oracle written independently of
  `crates/ir/src/interpret/arith.rs`'s copy).
- `crates/codegen/tests/wasm_backend.rs` / the Fe-side
  `crates/codegen/tests/wasm_e2e.rs::f32_abs_min_max_clamp_intrinsics_execute_on_wasm`
  (Fe source -> wasm -> wasmtime, full pipeline, ordinary values).
- naga/SPIR-V: no GPU adapter exists in this sandbox, so the exact expansion
  is validated structurally (legal SPIR-V via `spirv-val`, legal WGSL via
  `naga::front::wgsl::parse_str` + `naga::valid::Validator`, and the emitted
  WGSL/SPIR-V contains no branch/phi for `Fmin`/`Fmax`/`Fclamp` — only
  `select`/`OpSelect`), not executed end-to-end against real driver behavior.

### Rounding family (`Ffloor`/`Fceil`/`Ftrunc`/`Fround`)

- `crates/codegen/tests/cranelift_backend.rs`: `cranelift_f32_rounding_oracle`
  (native oracle vs Rust's `f32::floor`/`ceil`/`trunc`/`round_ties_even`,
  bit-exact via `.to_bits()`, covering ties, negatives, `-0.0`, `+-inf`, and
  NaN-passthrough, plus the four `roundTiesToEven` answers spelled out:
  `round(0.5)==0`, `round(1.5)==2`, `round(2.5)==2`, `round(-0.5)==-0`) and
  `cross_backend_f32_rounding_differential` (wasm vs cranelift, bit-for-bit,
  same edge-input set, both checked against the Rust oracle).
- `crates/codegen/tests/spirv_backend.rs`:
  `spirv_f32_rounding_lowering_is_exact_and_branch_free` (structural: each op
  emits exactly one native WGSL call -- `floor(`/`ceil(`/`trunc(`/`round(` --
  no `if`/`else`/`loop {`, legal SPIR-V magic, reparses and validates via
  `naga::valid::Validator`; mirrors
  `spirv_f32_minmaxabsclamp_lowering_is_exact_and_branch_free`'s shape but
  does NOT assert a global "zero `select(`" count, since the kernel's
  trailing `F32ToI32` return-ABI conversion legitimately emits its own
  `select(`s for saturation, unrelated to this family).
- `crates/codegen/tests/wasm_backend.rs` / the Fe-side
  `crates/codegen/tests/wasm_e2e.rs::f32_rounding_intrinsics_execute_on_wasm`
  (Fe source -> wasm -> wasmtime, full pipeline, same value set as the
  cranelift oracle, plus the wasm opcode-presence and branch-freedom checks).
- Fe demo pipeline: `demos/sketches/fmath`'s `floor` (and `ceil`/`round`, now
  newly exposed) call the intrinsics directly; `fract`/`wrap_pi`/`sin` build
  on `floor` and go branch-free for free. Regenerating `demos/sketches/cga3d`
  and `demos/sketches/desargues`'s WGSL and diffing against the pre-change
  output confirms the hand-rolled `if (t > x) { t - 1.0 } else { t }` floor
  pattern is gone, replaced by a single `floor(...)` call, with the
  corresponding `phi_`/branch count dropping. See the top-level agent report
  for this task for the before/after WGSL snippet.
