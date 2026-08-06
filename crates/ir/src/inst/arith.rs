use macros::Inst;

use crate::ValueId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Inst)]
#[inst(kind(unary(Neg)))]
pub struct Neg {
    arg: ValueId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Inst)]
#[inst(kind(unary(Fneg)))]
pub struct Fneg {
    arg: ValueId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Inst)]
#[inst(kind(binary(Fadd)))]
pub struct Fadd {
    lhs: ValueId,
    rhs: ValueId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Inst)]
#[inst(kind(binary(Fsub)))]
pub struct Fsub {
    lhs: ValueId,
    rhs: ValueId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Inst)]
#[inst(kind(binary(Fmul)))]
pub struct Fmul {
    lhs: ValueId,
    rhs: ValueId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Inst)]
#[inst(kind(binary(Fdiv)))]
pub struct Fdiv {
    lhs: ValueId,
    rhs: ValueId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Inst)]
#[inst(kind(unary(Fsqrt)))]
pub struct Fsqrt {
    arg: ValueId,
}

/// Floating point absolute value. A pure bitwise sign-clear: deterministic on
/// every backend, including for NaN payloads (only the sign bit changes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Inst)]
#[inst(kind(unary(Fabs)))]
pub struct Fabs {
    arg: ValueId,
}

/// Floating point minimum. Semantics are PINNED to the "WebAssembly rules"
/// (IEEE 754-2019 `minimum`): NaN-propagating (either operand NaN => NaN
/// result), and -0.0 is treated as strictly less than +0.0 regardless of
/// argument order. This is exactly wasm's `f32.min`, (by its own docs)
/// cranelift's `fmin`, AND (via a branch-free integer key-compare-and-select
/// expansion; GLSL.std.450 `FMin` alone is not exact) naga/SPIR-V's lowering.
/// See `docs/numeric-intrinsics-semantics.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Inst)]
#[inst(kind(binary(Fmin)))]
pub struct Fmin {
    lhs: ValueId,
    rhs: ValueId,
}

/// Floating point maximum. See [`Fmin`] for the pinned semantics (the
/// maximum-side mirror of the same "WebAssembly rules").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Inst)]
#[inst(kind(binary(Fmax)))]
pub struct Fmax {
    lhs: ValueId,
    rhs: ValueId,
}

/// Floating point clamp: `clamp(arg, lo, hi)`. A dedicated ternary op (not
/// composed at this layer) so the IR carries clamp semantics as a unit and
/// each backend lowers it as it sees fit. All three backends currently
/// compose it as `min(max(arg, lo), hi)` (the textbook clamp order,
/// deliberately NOT `max(min(arg, hi), lo)`): wasm and cranelift from their
/// native min/max, and naga/SPIR-V from the exact branch-free min/max
/// expansion (NOT a single GLSL.std.450 `FClamp`, which is poison when
/// `lo > hi`) -- documented at each backend's lowering site. Deliberately `InstClassKind::Opaque`
/// (no `kind(...)` attribute): a ternary op has no `UnaryInstKind`/`BinaryInstKind`
/// home, and Opaque already gets correct, safe treatment everywhere (no constant
/// folding/peephole simplification, but still eligible for ordinary structural GVN
/// CSE since it is a pure op with the default `SideEffect::None`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Inst)]
pub struct Fclamp {
    arg: ValueId,
    lo: ValueId,
    hi: ValueId,
}

/// Floating point floor: round to integral, towards negative infinity.
/// Mirrors `Fsqrt`/`Fabs` exactly -- a single native instruction on every
/// backend (naga `MathFunction::Floor`, wasm `f32.floor`, cranelift
/// `floor`), no bit-twiddling, no NaN/-0 subtlety (floor is monotone and
/// sign-preserving at the boundary: `floor(-0.0) == -0.0`). See
/// `docs/numeric-intrinsics-semantics.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Inst)]
#[inst(kind(unary(Ffloor)))]
pub struct Ffloor {
    arg: ValueId,
}

/// Floating point ceiling: round to integral, towards positive infinity.
/// Mirrors `Ffloor` (see above).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Inst)]
#[inst(kind(unary(Fceil)))]
pub struct Fceil {
    arg: ValueId,
}

/// Floating point truncation: round to integral, towards zero. Mirrors
/// `Ffloor` (see above).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Inst)]
#[inst(kind(unary(Ftrunc)))]
pub struct Ftrunc {
    arg: ValueId,
}

/// Floating point round: round to integral, towards the nearest value, with
/// ties rounded to even (`roundTiesToEven`, IEEE 754). PINNED: this matches
/// wasm's `f32.nearest` and cranelift's `nearest` (both ties-to-even by
/// their own spec/doc comment) AND naga/SPIR-V, whose `MathFunction::Round`
/// lowers to GLSL.std.450 `RoundEven` (verified against naga 29.0.4's SPIR-V
/// backend source), not the ties-away-from-zero `Round` ext inst. All three
/// backends agree exactly; no divergence to pin around, unlike `Fmin`/
/// `Fmax`. See `docs/numeric-intrinsics-semantics.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Inst)]
#[inst(kind(unary(Fround)))]
pub struct Fround {
    arg: ValueId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Inst)]
#[inst(kind(binary(Add)))]
pub struct Add {
    lhs: ValueId,
    rhs: ValueId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Inst)]
#[inst(kind(binary(Uaddo)))]
pub struct Uaddo {
    lhs: ValueId,
    rhs: ValueId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Inst)]
#[inst(kind(binary(Uaddsat)))]
pub struct Uaddsat {
    lhs: ValueId,
    rhs: ValueId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Inst)]
#[inst(kind(binary(Saddo)))]
pub struct Saddo {
    lhs: ValueId,
    rhs: ValueId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Inst)]
#[inst(kind(binary(Saddsat)))]
pub struct Saddsat {
    lhs: ValueId,
    rhs: ValueId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Inst)]
#[inst(kind(binary(Mul)))]
pub struct Mul {
    lhs: ValueId,
    rhs: ValueId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Inst)]
#[inst(kind(binary(Sub)))]
pub struct Sub {
    lhs: ValueId,
    rhs: ValueId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Inst)]
#[inst(kind(binary(Usubo)))]
pub struct Usubo {
    lhs: ValueId,
    rhs: ValueId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Inst)]
#[inst(kind(binary(Usubsat)))]
pub struct Usubsat {
    lhs: ValueId,
    rhs: ValueId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Inst)]
#[inst(kind(binary(Ssubo)))]
pub struct Ssubo {
    lhs: ValueId,
    rhs: ValueId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Inst)]
#[inst(kind(binary(Ssubsat)))]
pub struct Ssubsat {
    lhs: ValueId,
    rhs: ValueId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Inst)]
#[inst(kind(binary(Umulo)))]
pub struct Umulo {
    lhs: ValueId,
    rhs: ValueId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Inst)]
#[inst(kind(binary(Umulsat)))]
pub struct Umulsat {
    lhs: ValueId,
    rhs: ValueId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Inst)]
#[inst(kind(binary(Smulo)))]
pub struct Smulo {
    lhs: ValueId,
    rhs: ValueId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Inst)]
#[inst(kind(binary(Smulsat)))]
pub struct Smulsat {
    lhs: ValueId,
    rhs: ValueId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Inst)]
#[inst(kind(unary(Snego)))]
pub struct Snego {
    arg: ValueId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Inst)]
#[inst(kind(binary(Sdiv)))]
pub struct Sdiv {
    lhs: ValueId,
    rhs: ValueId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Inst)]
#[inst(kind(binary(Udiv)))]
pub struct Udiv {
    lhs: ValueId,
    rhs: ValueId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Inst)]
#[inst(kind(binary(Umod)))]
pub struct Umod {
    lhs: ValueId,
    rhs: ValueId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Inst)]
#[inst(kind(binary(Smod)))]
pub struct Smod {
    lhs: ValueId,
    rhs: ValueId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Inst)]
#[inst(kind(binary(Shl)))]
pub struct Shl {
    bits: ValueId,
    value: ValueId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Inst)]
#[inst(kind(binary(Shr)))]
pub struct Shr {
    bits: ValueId,
    value: ValueId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Inst)]
#[inst(kind(binary(Sar)))]
pub struct Sar {
    bits: ValueId,
    value: ValueId,
}
