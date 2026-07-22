use super::{Action, Interpret, State, single_result};
use crate::inst::cast::*;

impl Interpret for Sext {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        state.set_action(Action::Continue);

        let value = state.lookup_val(*self.from());
        let ty = self.ty();

        single_result(value.with_imm(|value| value.sext(*ty)))
    }
}

impl Interpret for Zext {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        state.set_action(Action::Continue);

        let value = state.lookup_val(*self.from());
        let ty = self.ty();

        single_result(value.with_imm(|value| value.zext(*ty)))
    }
}

impl Interpret for Trunc {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        let value = state.lookup_val(*self.from());
        let ty = self.ty();
        state.set_action(Action::Continue);

        single_result(value.with_imm(|value| value.trunc(*ty)))
    }
}

impl Interpret for Bitcast {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        state.set_action(Action::Continue);
        single_result(
            state
                .lookup_val(*self.from())
                .with_imm(|value| value.bitcast(*self.ty())),
        )
    }
}

impl Interpret for IntToPtr {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        state.set_action(Action::Continue);
        let value = state.lookup_val(*self.from());
        let from_ty = state.dfg().value_ty(*self.from());

        single_result(value.with_imm(|value| {
            let ptr_repr = state.dfg().ctx.type_layout.pointer_repl();
            if from_ty > ptr_repr {
                value.trunc(ptr_repr)
            } else {
                value.zext(ptr_repr)
            }
        }))
    }
}

impl Interpret for PtrToInt {
    fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
        state.set_action(Action::Continue);
        let value = state.lookup_val(*self.from());
        let ty = self.ty();

        single_result(value.with_imm(|value| {
            let ptr_repr = state.dfg().ctx.type_layout.pointer_repl();
            if *ty > ptr_repr {
                value.zext(*ty)
            } else {
                value.trunc(*ty)
            }
        }))
    }
}

macro_rules! impl_numeric_conversion {
    ($inst:ty, $convert:expr) => {
        impl Interpret for $inst {
            fn interpret(&self, state: &mut dyn State) -> super::EvalResults {
                state.set_action(Action::Continue);
                let value = state.lookup_val(*self.from());
                single_result(match value {
                    super::EvalValue::Imm(imm) => ($convert)(imm),
                    other => other,
                })
            }
        }
    };
}

impl_numeric_conversion!(I32ToF32, |imm| match imm {
    crate::Immediate::I32(value) =>
        super::EvalValue::Imm(crate::Immediate::F32((value as f32).to_bits(),)),
    other => super::EvalValue::Imm(other),
});

impl_numeric_conversion!(U32ToF32, |imm| match imm {
    crate::Immediate::I32(value) =>
        super::EvalValue::Imm(crate::Immediate::F32(((value as u32) as f32).to_bits(),)),
    other => super::EvalValue::Imm(other),
});

impl_numeric_conversion!(F32ToI32, |imm| match imm {
    crate::Immediate::F32(bits) =>
        super::EvalValue::Imm(crate::Immediate::I32(f32::from_bits(bits) as i32,)),
    other => super::EvalValue::Imm(other),
});

impl_numeric_conversion!(F32ToU32, |imm| match imm {
    crate::Immediate::F32(bits) =>
        super::EvalValue::Imm(crate::Immediate::I32((f32::from_bits(bits) as u32) as i32,)),
    other => super::EvalValue::Imm(other),
});
