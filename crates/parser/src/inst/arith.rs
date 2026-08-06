use ir::inst::arith::*;

super::impl_inst_build! {Neg, (arg: ValueId)}
super::impl_inst_build! {Fneg, (arg: ValueId)}
super::impl_inst_build! {Fadd, (lhs: ValueId, rhs: ValueId)}
super::impl_inst_build! {Fsub, (lhs: ValueId, rhs: ValueId)}
super::impl_inst_build! {Fmul, (lhs: ValueId, rhs: ValueId)}
super::impl_inst_build! {Fdiv, (lhs: ValueId, rhs: ValueId)}
super::impl_inst_build! {Fsqrt, (arg: ValueId)}
super::impl_inst_build! {Fabs, (arg: ValueId)}
super::impl_inst_build! {Fmin, (lhs: ValueId, rhs: ValueId)}
super::impl_inst_build! {Fmax, (lhs: ValueId, rhs: ValueId)}
super::impl_inst_build! {Fclamp, (arg: ValueId, lo: ValueId, hi: ValueId)}
super::impl_inst_build! {Ffloor, (arg: ValueId)}
super::impl_inst_build! {Fceil, (arg: ValueId)}
super::impl_inst_build! {Ftrunc, (arg: ValueId)}
super::impl_inst_build! {Fround, (arg: ValueId)}
super::impl_inst_build! {Add, (lhs: ValueId, rhs: ValueId)}
super::impl_inst_build! {Uaddo, (lhs: ValueId, rhs: ValueId)}
super::impl_inst_build! {Uaddsat, (lhs: ValueId, rhs: ValueId)}
super::impl_inst_build! {Saddo, (lhs: ValueId, rhs: ValueId)}
super::impl_inst_build! {Saddsat, (lhs: ValueId, rhs: ValueId)}
super::impl_inst_build! {Mul, (lhs: ValueId, rhs: ValueId)}
super::impl_inst_build! {Sub, (lhs: ValueId, rhs: ValueId)}
super::impl_inst_build! {Usubo, (lhs: ValueId, rhs: ValueId)}
super::impl_inst_build! {Usubsat, (lhs: ValueId, rhs: ValueId)}
super::impl_inst_build! {Ssubo, (lhs: ValueId, rhs: ValueId)}
super::impl_inst_build! {Ssubsat, (lhs: ValueId, rhs: ValueId)}
super::impl_inst_build! {Umulo, (lhs: ValueId, rhs: ValueId)}
super::impl_inst_build! {Umulsat, (lhs: ValueId, rhs: ValueId)}
super::impl_inst_build! {Smulo, (lhs: ValueId, rhs: ValueId)}
super::impl_inst_build! {Smulsat, (lhs: ValueId, rhs: ValueId)}
super::impl_inst_build! {Snego, (arg: ValueId)}
super::impl_inst_build! {Sdiv, (lhs: ValueId, rhs: ValueId)}
super::impl_inst_build! {Udiv, (lhs: ValueId, rhs: ValueId)}
super::impl_inst_build! {Umod, (lhs: ValueId, rhs: ValueId)}
super::impl_inst_build! {Smod, (lhs: ValueId, rhs: ValueId)}
super::impl_inst_build! {Shl, (lhs: ValueId, rhs: ValueId)}
super::impl_inst_build! {Shr, (lhs: ValueId, rhs: ValueId)}
super::impl_inst_build! {Sar, (lhs: ValueId, rhs: ValueId)}
