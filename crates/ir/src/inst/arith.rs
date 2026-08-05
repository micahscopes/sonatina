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
/// argument order. This is exactly wasm's `f32.min` and (by its own docs)
/// cranelift's `fmin`. See `docs/numeric-intrinsics-semantics.md` for the
/// naga/SPIR-V divergence this does NOT paper over.
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
/// composed at this layer) so the naga/SPIR-V backend can emit a single
/// native `clamp()`/`FClamp` instruction; the wasm and cranelift backends
/// compose it from their native min/max as `max(min(arg, hi), lo)` is NOT
/// used -- they use `min(max(arg, lo), hi)` (the textbook clamp order),
/// documented at each backend's lowering site. Deliberately `InstClassKind::Opaque`
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
