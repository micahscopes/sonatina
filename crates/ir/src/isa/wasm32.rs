//! The `wasm32` ISA: a portable 32-bit-pointer target whose Sonatina IR is
//! translated to WebAssembly by the `wasm` codegen backend (WAFFLE).
//!
//! It mirrors [`Native`](super::native::Native): a little-endian target using
//! the full generic [`NativeInstSet`] vocabulary, so the portable Fe -> wasm
//! lowering can emit the same checked/saturating arithmetic, comparisons,
//! control flow, and data ops it would for a native target. The one deliberate
//! difference is `pointer_repl = I32`: wasm linear-memory addresses are 32-bit,
//! matching Fe's `WASM_LAYOUT` (`pointer_size_bytes = 4`).

use std::sync::LazyLock;

use sonatina_triple::{Architecture, TargetTriple};

use super::{Endian, Isa, TypeLayout, TypeLayoutError};
use crate::{
    AddressSpaceDesc, AddressSpaceId, AddressSpaceInfo, AddressSpaceKind, Type,
    inst::native::inst_set::NativeInstSet, module::ModuleCtx, types::CompoundType,
};

pub mod space {
    use crate::AddressSpaceId;

    pub const MEMORY: AddressSpaceId = AddressSpaceId::new(0);
}

static WASM32_ADDRESS_SPACES: [AddressSpaceDesc; 1] = [AddressSpaceDesc {
    id: space::MEMORY,
    name: "memory",
    kind: AddressSpaceKind::Linear,
    immutable: false,
}];

struct Wasm32AddressSpaces;

impl AddressSpaceInfo for Wasm32AddressSpaces {
    fn default_space(&self) -> AddressSpaceId {
        space::MEMORY
    }

    fn desc(&self, id: AddressSpaceId) -> AddressSpaceDesc {
        WASM32_ADDRESS_SPACES[id.as_u32() as usize]
    }

    fn all_spaces(&self) -> &'static [AddressSpaceDesc] {
        &WASM32_ADDRESS_SPACES
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Wasm32 {
    triple: TargetTriple,
}

impl Wasm32 {
    pub fn new(triple: TargetTriple) -> Self {
        assert!(matches!(triple.architecture, Architecture::Wasm32));
        Self { triple }
    }
}

impl Isa for Wasm32 {
    type InstSet = NativeInstSet;

    fn triple(&self) -> TargetTriple {
        self.triple
    }

    fn type_layout(&self) -> &'static dyn TypeLayout {
        const TL: Wasm32TypeLayout = Wasm32TypeLayout {};
        &TL
    }

    fn address_spaces(&self) -> &'static dyn AddressSpaceInfo {
        static SPACES: Wasm32AddressSpaces = Wasm32AddressSpaces;
        &SPACES
    }

    fn inst_set(&self) -> &'static Self::InstSet {
        static IS: LazyLock<NativeInstSet> = LazyLock::new(NativeInstSet::new);
        &IS
    }
}

struct Wasm32TypeLayout {}

impl TypeLayout for Wasm32TypeLayout {
    fn size_of(&self, ty: Type, ctx: &ModuleCtx) -> Result<usize, TypeLayoutError> {
        let size = match ty {
            Type::Unit => 0,
            Type::I1 => 1,
            Type::I8 => 1,
            Type::I16 => 2,
            Type::I32 => 4,
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
                    // Pointers/refs are 32-bit on wasm linear memory.
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

#[cfg(test)]
mod tests {
    use sonatina_triple::{Architecture, OperatingSystem, TargetTriple, Vendor};

    use super::{Wasm32, space};
    use crate::{AddressSpaceKind, InstSetBase, Type, isa::Isa};

    fn wasm32_triple() -> TargetTriple {
        TargetTriple::new(
            Architecture::Wasm32,
            Vendor::Unknown,
            OperatingSystem::Native,
        )
    }

    #[test]
    fn wasm32_pointer_is_32_bit_and_little_endian() {
        let isa = Wasm32::new(wasm32_triple());
        assert_eq!(isa.type_layout().pointer_repl(), Type::I32);
        assert_eq!(isa.address_spaces().default_space(), space::MEMORY);
        assert_eq!(
            isa.address_spaces().desc(space::MEMORY).kind,
            AddressSpaceKind::Linear
        );
    }

    #[test]
    fn wasm32_has_portable_inst_vocabulary() {
        let isa = Wasm32::new(wasm32_triple());
        let is = isa.inst_set();
        // Portable arithmetic + control flow the wasm lowering relies on.
        assert!(is.has_add().is_some());
        assert!(is.has_sub().is_some());
        assert!(is.has_mul().is_some());
        assert!(is.has_lt().is_some());
        assert!(is.has_phi().is_some());
        assert!(is.has_call().is_some());
        assert!(is.has_return().is_some());
    }
}
