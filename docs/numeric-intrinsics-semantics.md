# Float numeric intrinsics: cross-backend semantics

Added on branch `backend-numeric-intrinsics` (off `04d3ec91`): `Fabs`, `Fmin`,
`Fmax`, `Fclamp` (f32 only, mirroring `Fsqrt`).

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
- `Fclamp` (ternary, 3 args: `arg`, `lo`, `hi`): a **dedicated** op, not
  composed as `Fmin(Fmax(x, lo), hi)` at this layer. Rationale: naga's
  `MathFunction::Clamp` is a genuine single hardware instruction
  (`GLSL.std.450 FClamp` / WGSL `clamp()`), which is the actual motivating win
  ("branch-free single hardware instructions") for the GPU backend — composing
  it from two ops at the IR level would forever prevent naga from ever
  emitting a single `clamp()` call. wasm and cranelift have no native clamp
  instruction either way, so they compose it as `min(max(arg, lo), hi)`
  (documented at each backend's lowering site; both compose identically).
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

**naga/SPIR-V does NOT match, and cannot be made to.** `MathFunction::Min`/
`Max` lowers to SPIR-V's `GLSL.std.450` extended instructions `FMin`/`FMax`.
Per the Khronos GLSL.std.450 spec, *"Which operand is the result is undefined
if one of the operands is a NaN"* — the spec does not even guarantee
NaN-avoidance (unlike IEEE `minNum`/`maxNum`), let alone NaN-propagation. Real
GPU driver implementations vary. There is also no spec guarantee on
`-0.0`/`+0.0` ordering. **This is a genuine, spec-level divergence, not an
implementation gap** — we do not control the GPU driver's native `min`/`max`
instruction, and decomposing the naga lowering into explicit
sign/NaN-checking comparisons would reintroduce exactly the branchy, non-
"single hardware instruction" code this feature exists to eliminate. No GPU
adapter exists in this sandbox to even observe real driver behavior; the
naga/SPIR-V path is validated (`spirv-val`, legal module) but not executed,
and NOT asserted to bit-agree with wasm/cranelift on NaN/-0.0 inputs.

**`Fabs` has no such issue.** It is a pure, deterministic bitwise sign-clear
on every backend (cranelift's own doc: *"Note that this is a pure bitwise
operation"*; wasm `f32.abs` and GLSL.std.450 `FAbs` are equivalently
bitwise), so NaN payloads pass through unchanged except for the sign bit.

**`Fclamp`'s composition inherits the same caveat two levels down** on wasm/
cranelift (`min(max(x, lo), hi)`, both native, both pinned), but on naga it is
a single opaque `FClamp` call, whose own NaN/out-of-order-bounds behavior is
likewise implementation-defined (naga's own SPIR-V backend comment: *"Clamp is
undefined if min > max"*).

## OPEN DECISION (flagged for review before push)

The naga/SPIR-V Fmin/Fmax/Fclamp NaN and -0.0 divergence from wasm/cranelift
is real and, as far as this sandbox can determine, not fixable without
sacrificing the single-hardware-instruction win. It is a spec-level property
of GLSL.std.450, not a bug in this implementation. Flagging for Micah/Codex
review before the sonatina branch is pushed: is this divergence acceptable
given the demo's actual value domain (geometric-algebra distances/clamps,
which practically never produce NaN or -0.0), or does it need an explicit
guard/fallback for the GPU path?

## Verification

- `crates/codegen/tests/cranelift_backend.rs`:
  `cranelift_f32_abs_min_max_clamp_oracle` (native oracle, ordinary values,
  bit-exact vs a plain Rust reference) and
  `cross_backend_f32_min_max_nan_zero_inf_differential` (wasm vs cranelift
  bit-for-bit over normal/±0/±inf/NaN inputs, both checked against a
  from-scratch "WebAssembly rules" oracle written independently of
  `crates/ir/src/interpret/arith.rs`'s copy; naga/SPIR-V validated only).
- `crates/codegen/tests/wasm_backend.rs` / the Fe-side
  `crates/codegen/tests/wasm_e2e.rs::f32_abs_min_max_clamp_intrinsics_execute_on_wasm`
  (Fe source -> wasm -> wasmtime, full pipeline, ordinary values).
