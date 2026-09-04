//! Byte-addressed 32-bit arena layout shared by portable targets.
//!
//! This preserves the existing Wasm arena representation. It is not the layout
//! of typed shader locals or GPU resource interfaces, which have separate
//! alignment and representation requirements.

use super::{Endian, TypeLayout, TypeLayoutError};
use crate::{Type, module::ModuleCtx, types::CompoundType};

pub(super) struct Arena32TypeLayout {}

impl TypeLayout for Arena32TypeLayout {
    fn size_of(&self, ty: Type, ctx: &ModuleCtx) -> Result<usize, TypeLayoutError> {
        let size = match ty {
            Type::Unit => 0,
            Type::I1 => 1,
            Type::I8 => 1,
            Type::I16 => 2,
            Type::I32 => 4,
            Type::F32 => 4,
            Type::I64 => 8,
            Type::I128 => 16,
            Type::I256 => 32,
            Type::EnumTag(_) => return Err(TypeLayoutError::UnrepresentableType(ty)),
            Type::Compound(cmpd) => {
                let cmpd = ctx.with_ty_store(|s| s.resolve_compound(cmpd).clone());
                match cmpd {
                    CompoundType::Array { elem, len } => {
                        let elem_size = self.size_of(elem, ctx)?;
                        elem_size * len
                    }
                    CompoundType::Struct(s) => {
                        let mut total = 0;
                        for &field in &s.fields {
                            let align = self.align_of(field, ctx)?;
                            total = (total + align - 1) & !(align - 1);
                            total += self.size_of(field, ctx)?;
                        }
                        let struct_align = s
                            .fields
                            .iter()
                            .map(|f| self.align_of(*f, ctx).unwrap_or(1))
                            .max()
                            .unwrap_or(1);
                        (total + struct_align - 1) & !(struct_align - 1)
                    }
                    // References in this byte-addressed representation use 32-bit offsets.
                    CompoundType::Ptr(_) | CompoundType::ObjRef(_) | CompoundType::ConstRef(_) => 4,
                    _ => return Err(TypeLayoutError::UnsupportedType(ty)),
                }
            }
        };
        Ok(size)
    }

    fn align_of(&self, ty: Type, ctx: &ModuleCtx) -> Result<usize, TypeLayoutError> {
        let align = match ty {
            Type::Unit => 1,
            Type::I1 | Type::I8 => 1,
            Type::I16 => 2,
            Type::I32 => 4,
            Type::F32 => 4,
            Type::I64 => 8,
            Type::I128 => 16,
            Type::I256 => 32,
            Type::EnumTag(_) => return Err(TypeLayoutError::UnrepresentableType(ty)),
            Type::Compound(cmpd) => {
                let cmpd = ctx.with_ty_store(|s| s.resolve_compound(cmpd).clone());
                match cmpd {
                    CompoundType::Array { elem, .. } => self.align_of(elem, ctx)?,
                    CompoundType::Struct(s) => s
                        .fields
                        .iter()
                        .map(|f| self.align_of(*f, ctx).unwrap_or(1))
                        .max()
                        .unwrap_or(1),
                    CompoundType::Ptr(_) | CompoundType::ObjRef(_) | CompoundType::ConstRef(_) => 4,
                    _ => return Err(TypeLayoutError::UnsupportedType(ty)),
                }
            }
        };
        Ok(align)
    }

    fn pointer_repl(&self) -> Type {
        Type::I32
    }

    fn endian(&self) -> Endian {
        Endian::Le
    }
}
