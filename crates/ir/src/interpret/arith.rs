use super::{Action, EvalValue, Interpret, State, single_result};
use crate::inst::arith::*;

fn unary_f32(value: EvalValue, op: impl FnOnce(f32) -> f32) -> EvalValue {
    match value {
        EvalValue::Imm(crate::Immediate::F32(bits)) => {
            EvalValue::Imm(crate::Immediate::F32(op(f32::from_bits(bits)).to_bits()))
        }
        _ => EvalValue::Undef,
    }
}

fn binary_f32(lhs: EvalValue, rhs: EvalValue, op: impl FnOnce(f32, f32) -> f32) -> EvalValue {
    match (lhs, rhs) {
        (EvalValue::Imm(crate::Immediate::F32(lhs)), EvalValue::Imm(crate::Immediate::F32(rhs))) => {
            EvalValue::Imm(crate::Immediate::F32(
                op(f32::from_bits(lhs), f32::from_bits(rhs)).to_bits(),
            ))
        }
        _ => EvalValue::Undef,
    }
}

impl Interpret for Fneg {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        state.set_action(Action::Continue);
        single_result(unary_f32(state.lookup_val(*self.arg()), |value| -value))
    }
}

macro_rules! impl_binary_f32 {
    ($inst:ty, $op:expr) => {
        impl Interpret for $inst {
            fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
                state.set_action(Action::Continue);
                single_result(binary_f32(
                    state.lookup_val(*self.lhs()),
                    state.lookup_val(*self.rhs()),
                    $op,
                ))
            }
        }
    };
}

impl_binary_f32!(Fadd, |lhs, rhs| lhs + rhs);
impl_binary_f32!(Fsub, |lhs, rhs| lhs - rhs);
impl_binary_f32!(Fmul, |lhs, rhs| lhs * rhs);
impl_binary_f32!(Fdiv, |lhs, rhs| lhs / rhs);

impl Interpret for Fsqrt {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        state.set_action(Action::Continue);
        single_result(unary_f32(state.lookup_val(*self.arg()), f32::sqrt))
    }
}

/// Bitwise sign-clear. Deterministic for every input, including NaN (only the
/// sign bit changes, the payload is untouched).
fn fabs_bits(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0x7fff_ffff)
}

impl Interpret for Fabs {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        state.set_action(Action::Continue);
        single_result(unary_f32(state.lookup_val(*self.arg()), fabs_bits))
    }
}

impl Interpret for Ffloor {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        state.set_action(Action::Continue);
        single_result(unary_f32(state.lookup_val(*self.arg()), f32::floor))
    }
}

impl Interpret for Fceil {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        state.set_action(Action::Continue);
        single_result(unary_f32(state.lookup_val(*self.arg()), f32::ceil))
    }
}

impl Interpret for Ftrunc {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        state.set_action(Action::Continue);
        single_result(unary_f32(state.lookup_val(*self.arg()), f32::trunc))
    }
}

/// `roundTiesToEven` (IEEE 754), NOT Rust's `f32::round()` (which is
/// ties-away-from-zero). This matches wasm's `f32.nearest`, cranelift's
/// `nearest`, and naga/SPIR-V's `MathFunction::Round` (-> GLSL.std.450
/// `RoundEven`) -- all three backends agree exactly, so there is nothing to
/// pin/approximate here, unlike `Fmin`/`Fmax`.
impl Interpret for Fround {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        state.set_action(Action::Continue);
        single_result(unary_f32(state.lookup_val(*self.arg()), f32::round_ties_even))
    }
}

/// The PINNED cross-backend semantics for float min/max: the "WebAssembly
/// rules" (IEEE 754-2019 `minimum`/`maximum`). NaN-propagating (if either
/// operand is NaN, the result is NaN -- we return the canonical quiet NaN,
/// which the spec permits since the choice of NaN payload/sign is
/// unspecified), and -0.0 is treated as strictly less than +0.0 regardless of
/// argument order. This matches wasm's `f32.min`/`f32.max` and cranelift's
/// `fmin`/`fmax` (whose own doc comment says "propagating NaNs using the
/// WebAssembly rules"), AND naga/SPIR-V's lowering (`emit_exact_fminmax` in
/// `isa/spirv/mod.rs`, a branch-free integer key-compare-and-select
/// expansion -- NOT `MathFunction::Min`/`Max`/GLSL.std.450 `FMin`/`FMax`,
/// whose NaN/-0 behavior is implementation-defined by spec and are no longer
/// used for these ops). See `docs/numeric-intrinsics-semantics.md`.
const CANONICAL_NAN: f32 = f32::from_bits(0x7fc0_0000);

fn wasm_rules_fmin(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        return CANONICAL_NAN;
    }
    if a == 0.0 && b == 0.0 {
        return if a.is_sign_negative() || b.is_sign_negative() {
            -0.0
        } else {
            0.0
        };
    }
    if a < b { a } else { b }
}

fn wasm_rules_fmax(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        return CANONICAL_NAN;
    }
    if a == 0.0 && b == 0.0 {
        return if a.is_sign_negative() && b.is_sign_negative() {
            -0.0
        } else {
            0.0
        };
    }
    if a > b { a } else { b }
}

impl Interpret for Fmin {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        state.set_action(Action::Continue);
        single_result(binary_f32(
            state.lookup_val(*self.lhs()),
            state.lookup_val(*self.rhs()),
            wasm_rules_fmin,
        ))
    }
}

impl Interpret for Fmax {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        state.set_action(Action::Continue);
        single_result(binary_f32(
            state.lookup_val(*self.lhs()),
            state.lookup_val(*self.rhs()),
            wasm_rules_fmax,
        ))
    }
}

/// `FminRelaxed`/`FmaxRelaxed` evaluate AS EXACT: the canonical refinement of
/// the relaxed "any latitude" contract, chosen so const-eval stays
/// deterministic and backend-independent (a program CTFE-folding a relaxed
/// min/max gets the same answer regardless of which backend eventually runs
/// it). Reuses the same `wasm_rules_fmin`/`wasm_rules_fmax` pinned functions
/// as `Fmin`/`Fmax` above.
impl Interpret for FminRelaxed {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        state.set_action(Action::Continue);
        single_result(binary_f32(
            state.lookup_val(*self.lhs()),
            state.lookup_val(*self.rhs()),
            wasm_rules_fmin,
        ))
    }
}

impl Interpret for FmaxRelaxed {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        state.set_action(Action::Continue);
        single_result(binary_f32(
            state.lookup_val(*self.lhs()),
            state.lookup_val(*self.rhs()),
            wasm_rules_fmax,
        ))
    }
}

impl Interpret for Neg {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        state.set_action(Action::Continue);

        let val = state.lookup_val(*self.arg());
        single_result(val.with_imm(|value| -value))
    }
}

impl Interpret for Add {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        state.set_action(Action::Continue);

        let lhs = state.lookup_val(*self.lhs());
        let rhs = state.lookup_val(*self.rhs());

        single_result(EvalValue::zip_with_imm(lhs, rhs, |lhs, rhs| lhs + rhs))
    }
}

impl Interpret for Uaddo {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        state.set_action(Action::Continue);

        let lhs = state.lookup_val(*self.lhs());
        let rhs = state.lookup_val(*self.rhs());
        let (EvalValue::Imm(lhs), EvalValue::Imm(rhs)) = (lhs, rhs) else {
            return smallvec::smallvec![EvalValue::Undef, EvalValue::Undef];
        };

        let (sum, overflow) = lhs.overflowing_uadd(rhs);
        smallvec::smallvec![EvalValue::Imm(sum), EvalValue::Imm(overflow.into())]
    }
}

impl Interpret for Uaddsat {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        state.set_action(Action::Continue);

        let lhs = state.lookup_val(*self.lhs());
        let rhs = state.lookup_val(*self.rhs());
        single_result(EvalValue::zip_with_imm(lhs, rhs, |lhs, rhs| {
            lhs.saturating_uadd(rhs)
        }))
    }
}

impl Interpret for Saddo {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        state.set_action(Action::Continue);

        let lhs = state.lookup_val(*self.lhs());
        let rhs = state.lookup_val(*self.rhs());
        let (EvalValue::Imm(lhs), EvalValue::Imm(rhs)) = (lhs, rhs) else {
            return smallvec::smallvec![EvalValue::Undef, EvalValue::Undef];
        };

        let (sum, overflow) = lhs.overflowing_sadd(rhs);
        smallvec::smallvec![EvalValue::Imm(sum), EvalValue::Imm(overflow.into())]
    }
}

impl Interpret for Saddsat {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        state.set_action(Action::Continue);

        let lhs = state.lookup_val(*self.lhs());
        let rhs = state.lookup_val(*self.rhs());
        single_result(EvalValue::zip_with_imm(lhs, rhs, |lhs, rhs| {
            lhs.saturating_sadd(rhs)
        }))
    }
}

impl Interpret for Sub {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        state.set_action(Action::Continue);

        let lhs = state.lookup_val(*self.lhs());
        let rhs = state.lookup_val(*self.rhs());

        single_result(EvalValue::zip_with_imm(lhs, rhs, |lhs, rhs| {
            EvalValue::Imm(lhs - rhs)
        }))
    }
}

impl Interpret for Usubo {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        state.set_action(Action::Continue);

        let lhs = state.lookup_val(*self.lhs());
        let rhs = state.lookup_val(*self.rhs());
        let (EvalValue::Imm(lhs), EvalValue::Imm(rhs)) = (lhs, rhs) else {
            return smallvec::smallvec![EvalValue::Undef, EvalValue::Undef];
        };

        let (diff, overflow) = lhs.overflowing_usub(rhs);
        smallvec::smallvec![EvalValue::Imm(diff), EvalValue::Imm(overflow.into())]
    }
}

impl Interpret for Usubsat {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        state.set_action(Action::Continue);

        let lhs = state.lookup_val(*self.lhs());
        let rhs = state.lookup_val(*self.rhs());
        single_result(EvalValue::zip_with_imm(lhs, rhs, |lhs, rhs| {
            lhs.saturating_usub(rhs)
        }))
    }
}

impl Interpret for Ssubo {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        state.set_action(Action::Continue);

        let lhs = state.lookup_val(*self.lhs());
        let rhs = state.lookup_val(*self.rhs());
        let (EvalValue::Imm(lhs), EvalValue::Imm(rhs)) = (lhs, rhs) else {
            return smallvec::smallvec![EvalValue::Undef, EvalValue::Undef];
        };

        let (diff, overflow) = lhs.overflowing_ssub(rhs);
        smallvec::smallvec![EvalValue::Imm(diff), EvalValue::Imm(overflow.into())]
    }
}

impl Interpret for Ssubsat {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        state.set_action(Action::Continue);

        let lhs = state.lookup_val(*self.lhs());
        let rhs = state.lookup_val(*self.rhs());
        single_result(EvalValue::zip_with_imm(lhs, rhs, |lhs, rhs| {
            lhs.saturating_ssub(rhs)
        }))
    }
}

impl Interpret for Mul {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        state.set_action(Action::Continue);

        let lhs = state.lookup_val(*self.lhs());
        let rhs = state.lookup_val(*self.rhs());
        state.set_action(Action::Continue);

        single_result(EvalValue::zip_with_imm(lhs, rhs, |lhs, rhs| lhs * rhs))
    }
}

impl Interpret for Umulo {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        state.set_action(Action::Continue);

        let lhs = state.lookup_val(*self.lhs());
        let rhs = state.lookup_val(*self.rhs());
        let (EvalValue::Imm(lhs), EvalValue::Imm(rhs)) = (lhs, rhs) else {
            return smallvec::smallvec![EvalValue::Undef, EvalValue::Undef];
        };

        let (product, overflow) = lhs.overflowing_umul(rhs);
        smallvec::smallvec![EvalValue::Imm(product), EvalValue::Imm(overflow.into())]
    }
}

impl Interpret for Umulsat {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        state.set_action(Action::Continue);

        let lhs = state.lookup_val(*self.lhs());
        let rhs = state.lookup_val(*self.rhs());
        single_result(EvalValue::zip_with_imm(lhs, rhs, |lhs, rhs| {
            lhs.saturating_umul(rhs)
        }))
    }
}

impl Interpret for Smulo {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        state.set_action(Action::Continue);

        let lhs = state.lookup_val(*self.lhs());
        let rhs = state.lookup_val(*self.rhs());
        let (EvalValue::Imm(lhs), EvalValue::Imm(rhs)) = (lhs, rhs) else {
            return smallvec::smallvec![EvalValue::Undef, EvalValue::Undef];
        };

        let (product, overflow) = lhs.overflowing_smul(rhs);
        smallvec::smallvec![EvalValue::Imm(product), EvalValue::Imm(overflow.into())]
    }
}

impl Interpret for Smulsat {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        state.set_action(Action::Continue);

        let lhs = state.lookup_val(*self.lhs());
        let rhs = state.lookup_val(*self.rhs());
        single_result(EvalValue::zip_with_imm(lhs, rhs, |lhs, rhs| {
            lhs.saturating_smul(rhs)
        }))
    }
}

impl Interpret for Snego {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        state.set_action(Action::Continue);

        let value = state.lookup_val(*self.arg());
        let EvalValue::Imm(value) = value else {
            return smallvec::smallvec![EvalValue::Undef, EvalValue::Undef];
        };

        let (negated, overflow) = value.overflowing_sneg();
        smallvec::smallvec![EvalValue::Imm(negated), EvalValue::Imm(overflow.into())]
    }
}

impl Interpret for Sdiv {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        state.set_action(Action::Continue);

        let lhs = state.lookup_val(*self.lhs());
        let rhs = state.lookup_val(*self.rhs());

        single_result(EvalValue::zip_with_imm(lhs, rhs, |lhs, rhs| {
            if rhs.is_zero() {
                return EvalValue::Undef;
            }
            lhs.sdiv(rhs).into()
        }))
    }
}

impl Interpret for Udiv {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        state.set_action(Action::Continue);

        let lhs = state.lookup_val(*self.lhs());
        let rhs = state.lookup_val(*self.rhs());

        single_result(EvalValue::zip_with_imm(lhs, rhs, |lhs, rhs| {
            if rhs.is_zero() {
                return EvalValue::Undef;
            }
            lhs.udiv(rhs).into()
        }))
    }
}

impl Interpret for Umod {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        state.set_action(Action::Continue);

        let lhs = state.lookup_val(*self.lhs());
        let rhs = state.lookup_val(*self.rhs());

        single_result(EvalValue::zip_with_imm(lhs, rhs, |lhs, rhs| {
            if rhs.is_zero() {
                return EvalValue::Undef;
            }
            lhs.urem(rhs).into()
        }))
    }
}

impl Interpret for Smod {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        state.set_action(Action::Continue);

        let lhs = state.lookup_val(*self.lhs());
        let rhs = state.lookup_val(*self.rhs());

        single_result(EvalValue::zip_with_imm(lhs, rhs, |lhs, rhs| {
            if rhs.is_zero() {
                return EvalValue::Undef;
            }
            lhs.srem(rhs).into()
        }))
    }
}

impl Interpret for Shl {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        state.set_action(Action::Continue);

        let bits = state.lookup_val(*self.bits());
        let value = state.lookup_val(*self.value());
        single_result(EvalValue::zip_with_imm(bits, value, |bits, value| {
            value << bits
        }))
    }
}

impl Interpret for Shr {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        state.set_action(Action::Continue);

        let bits = state.lookup_val(*self.bits());
        let value = state.lookup_val(*self.value());

        single_result(EvalValue::zip_with_imm(bits, value, |bits, value| {
            value >> bits
        }))
    }
}

impl Interpret for Sar {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        state.set_action(Action::Continue);

        let bits = state.lookup_val(*self.bits());
        let value = state.lookup_val(*self.value());

        single_result(EvalValue::zip_with_imm(bits, value, |bits, value| {
            value.ashr(bits)
        }))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::{
        DataFlowGraph, Immediate, Type,
        builder::test_util::test_isa,
        interpret::EvalResults,
        module::{FuncRef, ModuleCtx},
    };

    use super::*;

    use crate::inst::evm::inst_set::EvmInstSet;

    struct TestState {
        dfg: DataFlowGraph,
        values: HashMap<crate::ValueId, EvalValue>,
    }

    impl TestState {
        fn new(values: impl IntoIterator<Item = (crate::ValueId, EvalValue)>) -> Self {
            let isa = test_isa();
            let dfg = DataFlowGraph::new(ModuleCtx::new(&isa));
            Self {
                dfg,
                values: values.into_iter().collect(),
            }
        }
    }

    impl State for TestState {
        fn lookup_val(&mut self, value: crate::ValueId) -> EvalValue {
            self.values.get(&value).cloned().unwrap_or_default()
        }

        fn call_func(&mut self, _func: FuncRef, _args: Vec<EvalValue>) -> EvalResults {
            unreachable!()
        }

        fn set_action(&mut self, action: Action) {
            assert_eq!(action, Action::Continue);
        }

        fn prev_block(&mut self) -> crate::BlockId {
            unreachable!()
        }

        fn load(&mut self, _addr: EvalValue, _ty: Type) -> EvalValue {
            unreachable!()
        }

        fn store(&mut self, _addr: EvalValue, _value: EvalValue, _ty: Type) -> EvalValue {
            unreachable!()
        }

        fn alloca(&mut self, _ty: Type) -> EvalValue {
            unreachable!()
        }

        fn dfg(&self) -> &DataFlowGraph {
            &self.dfg
        }
    }

    #[test]
    fn div_mod_by_zero_returns_undef() {
        let hi = EvmInstSet::new();
        let lhs = crate::ValueId::from_u32(0);
        let rhs = crate::ValueId::from_u32(1);

        let mut state = TestState::new([
            (lhs, EvalValue::Imm(Immediate::I32(1))),
            (rhs, EvalValue::Imm(Immediate::I32(0))),
        ]);

        assert_eq!(
            Sdiv::new(&hi, lhs, rhs).interpret(&mut state),
            super::single_result(EvalValue::Undef)
        );
        assert_eq!(
            Udiv::new(&hi, lhs, rhs).interpret(&mut state),
            super::single_result(EvalValue::Undef)
        );
        assert_eq!(
            Umod::new(&hi, lhs, rhs).interpret(&mut state),
            super::single_result(EvalValue::Undef)
        );
        assert_eq!(
            Smod::new(&hi, lhs, rhs).interpret(&mut state),
            super::single_result(EvalValue::Undef)
        );
    }

    #[test]
    fn shift_right_uses_expected_signedness() {
        let hi = EvmInstSet::new();
        let bits = crate::ValueId::from_u32(0);
        let value = crate::ValueId::from_u32(1);
        let mut state = TestState::new([
            (bits, EvalValue::Imm(Immediate::I8(1))),
            (value, EvalValue::Imm(Immediate::I8(-8))),
        ]);

        assert_eq!(
            Shr::new(&hi, bits, value).interpret(&mut state),
            super::single_result(EvalValue::Imm(Immediate::I8(124)))
        );
        assert_eq!(
            Sar::new(&hi, bits, value).interpret(&mut state),
            super::single_result(EvalValue::Imm(Immediate::I8(-4)))
        );
    }

    #[test]
    fn shift_right_is_width_aware_for_subword_operands() {
        let hi = EvmInstSet::new();
        let bits = crate::ValueId::from_u32(0);
        let value = crate::ValueId::from_u32(1);
        let mut state = TestState::new([
            (bits, EvalValue::Imm(Immediate::I32(8))),
            (value, EvalValue::Imm(Immediate::I32(-1))),
        ]);

        assert_eq!(
            Shr::new(&hi, bits, value).interpret(&mut state),
            super::single_result(EvalValue::Imm(Immediate::I32(0x00ff_ffff)))
        );
        assert_eq!(
            Sar::new(&hi, bits, value).interpret(&mut state),
            super::single_result(EvalValue::Imm(Immediate::I32(-1)))
        );

        let mut overshift_state = TestState::new([
            (bits, EvalValue::Imm(Immediate::I32(40))),
            (value, EvalValue::Imm(Immediate::I32(-1))),
        ]);
        assert_eq!(
            Shr::new(&hi, bits, value).interpret(&mut overshift_state),
            super::single_result(EvalValue::Imm(Immediate::I32(0)))
        );
        assert_eq!(
            Sar::new(&hi, bits, value).interpret(&mut overshift_state),
            super::single_result(EvalValue::Imm(Immediate::I32(-1)))
        );
    }

    #[test]
    fn uaddo_returns_sum_and_overflow_flag() {
        let hi = EvmInstSet::new();
        let lhs = crate::ValueId::from_u32(0);
        let rhs = crate::ValueId::from_u32(1);
        let mut state = TestState::new([
            (lhs, EvalValue::Imm(Immediate::I8(-1))),
            (rhs, EvalValue::Imm(Immediate::I8(1))),
        ]);

        assert_eq!(
            Uaddo::new(&hi, lhs, rhs).interpret(&mut state),
            crate::interpret::EvalResults::from_vec(vec![
                EvalValue::Imm(Immediate::I8(0)),
                EvalValue::Imm(Immediate::I1(true))
            ])
        );
    }

    #[test]
    fn signed_overflow_ops_return_wrapped_values_and_flags() {
        let hi = EvmInstSet::new();
        let lhs = crate::ValueId::from_u32(0);
        let rhs = crate::ValueId::from_u32(1);
        let mut state = TestState::new([
            (lhs, EvalValue::Imm(Immediate::I8(-128))),
            (rhs, EvalValue::Imm(Immediate::I8(-1))),
        ]);

        assert_eq!(
            Saddo::new(&hi, lhs, rhs).interpret(&mut state),
            crate::interpret::EvalResults::from_vec(vec![
                EvalValue::Imm(Immediate::I8(127)),
                EvalValue::Imm(Immediate::I1(true))
            ])
        );
        assert_eq!(
            Ssubo::new(&hi, lhs, rhs).interpret(&mut state),
            crate::interpret::EvalResults::from_vec(vec![
                EvalValue::Imm(Immediate::I8(-127)),
                EvalValue::Imm(Immediate::I1(false))
            ])
        );
        assert_eq!(
            Snego::new(&hi, lhs).interpret(&mut state),
            crate::interpret::EvalResults::from_vec(vec![
                EvalValue::Imm(Immediate::I8(-128)),
                EvalValue::Imm(Immediate::I1(true))
            ])
        );
    }

    #[test]
    fn unsigned_and_signed_mul_sub_ops_cover_overflow_cases() {
        let hi = EvmInstSet::new();
        let lhs = crate::ValueId::from_u32(0);
        let rhs = crate::ValueId::from_u32(1);
        let mut state = TestState::new([
            (lhs, EvalValue::Imm(Immediate::I8(-1))),
            (rhs, EvalValue::Imm(Immediate::I8(2))),
        ]);

        assert_eq!(
            Usubo::new(&hi, rhs, lhs).interpret(&mut state),
            crate::interpret::EvalResults::from_vec(vec![
                EvalValue::Imm(Immediate::I8(3)),
                EvalValue::Imm(Immediate::I1(true))
            ])
        );
        assert_eq!(
            Umulo::new(&hi, lhs, rhs).interpret(&mut state),
            crate::interpret::EvalResults::from_vec(vec![
                EvalValue::Imm(Immediate::I8(-2)),
                EvalValue::Imm(Immediate::I1(true))
            ])
        );
        assert_eq!(
            Smulo::new(&hi, lhs, rhs).interpret(&mut state),
            crate::interpret::EvalResults::from_vec(vec![
                EvalValue::Imm(Immediate::I8(-2)),
                EvalValue::Imm(Immediate::I1(false))
            ])
        );
    }

    #[test]
    fn saturating_ops_clamp_at_bounds() {
        let hi = EvmInstSet::new();
        let lhs = crate::ValueId::from_u32(0);
        let rhs = crate::ValueId::from_u32(1);

        let mut state = TestState::new([
            (lhs, EvalValue::Imm(Immediate::I8(120))),
            (rhs, EvalValue::Imm(Immediate::I8(20))),
        ]);
        assert_eq!(
            Saddsat::new(&hi, lhs, rhs).interpret(&mut state),
            super::single_result(EvalValue::Imm(Immediate::I8(127)))
        );

        state.values.insert(lhs, EvalValue::Imm(Immediate::I8(3)));
        state.values.insert(rhs, EvalValue::Imm(Immediate::I8(5)));
        assert_eq!(
            Usubsat::new(&hi, lhs, rhs).interpret(&mut state),
            super::single_result(EvalValue::Imm(Immediate::I8(0)))
        );

        state
            .values
            .insert(lhs, EvalValue::Imm(Immediate::I8(-120)));
        state.values.insert(rhs, EvalValue::Imm(Immediate::I8(20)));
        assert_eq!(
            Ssubsat::new(&hi, lhs, rhs).interpret(&mut state),
            super::single_result(EvalValue::Imm(Immediate::I8(-128)))
        );

        state.values.insert(lhs, EvalValue::Imm(Immediate::I8(-56)));
        state.values.insert(rhs, EvalValue::Imm(Immediate::I8(3)));
        assert_eq!(
            Umulsat::new(&hi, lhs, rhs).interpret(&mut state),
            super::single_result(EvalValue::Imm(Immediate::I8(-1)))
        );

        state.values.insert(lhs, EvalValue::Imm(Immediate::I8(100)));
        state.values.insert(rhs, EvalValue::Imm(Immediate::I8(2)));
        assert_eq!(
            Smulsat::new(&hi, lhs, rhs).interpret(&mut state),
            super::single_result(EvalValue::Imm(Immediate::I8(127)))
        );

        state.values.insert(lhs, EvalValue::Imm(Immediate::I8(-6)));
        state.values.insert(rhs, EvalValue::Imm(Immediate::I8(10)));
        assert_eq!(
            Uaddsat::new(&hi, lhs, rhs).interpret(&mut state),
            super::single_result(EvalValue::Imm(Immediate::I8(-1)))
        );
    }
}
