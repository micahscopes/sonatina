use ir::inst::cast::*;

super::impl_inst_build! {Sext, (from: ValueId, ty: Type)}
super::impl_inst_build! {Zext, (from: ValueId, ty: Type)}
super::impl_inst_build! {Trunc, (from: ValueId, ty: Type)}
super::impl_inst_build! {Bitcast, (from: ValueId, ty: Type)}
super::impl_inst_build! {IntToPtr, (from: ValueId, ty: Type)}
super::impl_inst_build! {PtrToInt, (from: ValueId, ty: Type)}
super::impl_inst_build! {I32ToF32, (from: ValueId)}
super::impl_inst_build! {U32ToF32, (from: ValueId)}
super::impl_inst_build! {F32ToI32, (from: ValueId)}
super::impl_inst_build! {F32ToU32, (from: ValueId)}
