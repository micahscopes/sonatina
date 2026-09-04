//! Shader target identity and conservative arena representation.
//!
//! The instruction vocabulary is shared with portable CPU targets, not their
//! execution environment. All byte-arena accesses remain in one conservative
//! space until typed resource effects can distinguish domains soundly.
//! Typed Naga locals and resource interfaces do not use the arena byte layout.

use std::sync::LazyLock;

use sonatina_triple::{Architecture, TargetTriple};

use super::{Isa, TypeLayout, arena32::Arena32TypeLayout};
use crate::{
    AddressSpaceDesc, AddressSpaceId, AddressSpaceInfo, AddressSpaceKind,
    inst::native::inst_set::NativeInstSet,
};

pub mod space {
    use crate::AddressSpaceId;

    pub const MEMORY: AddressSpaceId = AddressSpaceId::new(0);
}

static SHADER_ADDRESS_SPACES: [AddressSpaceDesc; 1] = [AddressSpaceDesc {
    id: space::MEMORY,
    name: "private_arena",
    kind: AddressSpaceKind::Linear,
    immutable: false,
}];

struct ShaderAddressSpaces;

impl AddressSpaceInfo for ShaderAddressSpaces {
    fn default_space(&self) -> AddressSpaceId {
        space::MEMORY
    }

    fn desc(&self, id: AddressSpaceId) -> AddressSpaceDesc {
        SHADER_ADDRESS_SPACES[id.as_u32() as usize]
    }

    fn all_spaces(&self) -> &'static [AddressSpaceDesc] {
        &SHADER_ADDRESS_SPACES
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Shader {
    triple: TargetTriple,
}

impl Shader {
    pub fn new(triple: TargetTriple) -> Self {
        assert!(matches!(triple.architecture, Architecture::Shader));
        Self { triple }
    }
}

impl Isa for Shader {
    type InstSet = NativeInstSet;

    fn triple(&self) -> TargetTriple {
        self.triple
    }

    fn type_layout(&self) -> &'static dyn TypeLayout {
        const TL: Arena32TypeLayout = Arena32TypeLayout {};
        &TL
    }

    fn address_spaces(&self) -> &'static dyn AddressSpaceInfo {
        static SPACES: ShaderAddressSpaces = ShaderAddressSpaces;
        &SPACES
    }

    fn inst_set(&self) -> &'static Self::InstSet {
        static IS: LazyLock<NativeInstSet> = LazyLock::new(NativeInstSet::new);
        &IS
    }
}
