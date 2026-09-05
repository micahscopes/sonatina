//! Derived identities for the existing two-result integer IR operations.
//! Targets own physical emission; this decoder does not grant target support.

use sonatina_ir::{Inst, InstDowncast, InstSetBase, ValueId};

#[derive(Clone, Copy)]
pub(crate) enum OverflowArithmetic {
    Add,
    Sub,
    Mul,
}

pub(crate) fn overflow_operands(
    inst_set: &dyn InstSetBase,
    instruction: &dyn Inst,
) -> Option<(OverflowArithmetic, bool, ValueId, ValueId)> {
    use sonatina_ir::inst::arith::{Saddo, Smulo, Ssubo, Uaddo, Umulo, Usubo};
    macro_rules! recognize {
        ($kind:ty, $op:ident, $signed:expr) => {
            if let Some(inst) = <&$kind as InstDowncast>::downcast(inst_set, instruction) {
                return Some((OverflowArithmetic::$op, $signed, *inst.lhs(), *inst.rhs()));
            }
        };
    }
    recognize!(Uaddo, Add, false);
    recognize!(Usubo, Sub, false);
    recognize!(Umulo, Mul, false);
    recognize!(Saddo, Add, true);
    recognize!(Ssubo, Sub, true);
    recognize!(Smulo, Mul, true);
    None
}
