use super::{Action, EvalValue, Interpret, State, single_result};
use crate::{Immediate, inst::cmp::*};

fn compare_f32(lhs: EvalValue, rhs: EvalValue, op: impl FnOnce(f32, f32) -> bool) -> EvalValue {
    match (lhs, rhs) {
        (EvalValue::Imm(Immediate::F32(lhs)), EvalValue::Imm(Immediate::F32(rhs))) => {
            EvalValue::Imm(Immediate::I1(op(f32::from_bits(lhs), f32::from_bits(rhs))))
        }
        _ => EvalValue::Undef,
    }
}

macro_rules! impl_compare_f32 {
    ($inst:ty, $op:expr) => {
        impl Interpret for $inst {
            fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
                state.set_action(Action::Continue);
                single_result(compare_f32(
                    state.lookup_val(*self.lhs()),
                    state.lookup_val(*self.rhs()),
                    $op,
                ))
            }
        }
    };
}

impl_compare_f32!(Feq, |lhs, rhs| lhs == rhs);
impl_compare_f32!(Flt, |lhs, rhs| lhs < rhs);
impl_compare_f32!(Fle, |lhs, rhs| lhs <= rhs);

impl Interpret for Lt {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        let lhs = state.lookup_val(*self.lhs());
        let rhs = state.lookup_val(*self.rhs());
        state.set_action(Action::Continue);

        single_result(EvalValue::zip_with_imm(lhs, rhs, |lhs, rhs| lhs.lt(rhs)))
    }
}

impl Interpret for Gt {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        let lhs = state.lookup_val(*self.lhs());
        let rhs = state.lookup_val(*self.rhs());
        state.set_action(Action::Continue);

        single_result(EvalValue::zip_with_imm(lhs, rhs, |lhs, rhs| lhs.gt(rhs)))
    }
}

impl Interpret for Slt {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        let lhs = state.lookup_val(*self.lhs());
        let rhs = state.lookup_val(*self.rhs());
        state.set_action(Action::Continue);

        single_result(EvalValue::zip_with_imm(lhs, rhs, |lhs, rhs| lhs.slt(rhs)))
    }
}

impl Interpret for Sgt {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        let lhs = state.lookup_val(*self.lhs());
        let rhs = state.lookup_val(*self.rhs());
        state.set_action(Action::Continue);

        single_result(EvalValue::zip_with_imm(lhs, rhs, |lhs, rhs| lhs.sgt(rhs)))
    }
}

impl Interpret for Le {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        let lhs = state.lookup_val(*self.lhs());
        let rhs = state.lookup_val(*self.rhs());
        state.set_action(Action::Continue);

        single_result(EvalValue::zip_with_imm(lhs, rhs, |lhs, rhs| lhs.le(rhs)))
    }
}

impl Interpret for Ge {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        let lhs = state.lookup_val(*self.lhs());
        let rhs = state.lookup_val(*self.rhs());
        state.set_action(Action::Continue);

        single_result(EvalValue::zip_with_imm(lhs, rhs, |lhs, rhs| lhs.ge(rhs)))
    }
}

impl Interpret for Sle {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        let lhs = state.lookup_val(*self.lhs());
        let rhs = state.lookup_val(*self.rhs());
        state.set_action(Action::Continue);

        single_result(EvalValue::zip_with_imm(lhs, rhs, |lhs, rhs| lhs.sle(rhs)))
    }
}

impl Interpret for Sge {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        let lhs = state.lookup_val(*self.lhs());
        let rhs = state.lookup_val(*self.rhs());
        state.set_action(Action::Continue);

        single_result(EvalValue::zip_with_imm(lhs, rhs, |lhs, rhs| lhs.sge(rhs)))
    }
}

impl Interpret for Eq {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        let lhs = state.lookup_val(*self.lhs());
        let rhs = state.lookup_val(*self.rhs());
        state.set_action(Action::Continue);

        single_result(EvalValue::zip_with_imm(lhs, rhs, |lhs, rhs| {
            lhs.imm_eq(rhs)
        }))
    }
}

impl Interpret for Ne {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        let lhs = state.lookup_val(*self.lhs());
        let rhs = state.lookup_val(*self.rhs());
        state.set_action(Action::Continue);

        single_result(EvalValue::zip_with_imm(lhs, rhs, |lhs, rhs| {
            lhs.imm_ne(rhs)
        }))
    }
}

impl Interpret for IsZero {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        let val = state.lookup_val(*self.lhs());
        state.set_action(Action::Continue);

        single_result(val.with_imm(|value| Immediate::from(value.is_zero())))
    }
}
