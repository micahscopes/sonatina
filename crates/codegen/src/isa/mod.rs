pub mod evm;

#[cfg(feature = "cranelift")]
pub mod cranelift;

#[cfg(feature = "wasm")]
pub mod wasm;

pub mod spirv;

/// Compute the element size (stride, in bytes) for an aggregate object type.
///
/// Feature-neutral layout helper shared by the cranelift and wasm ISA
/// translators; lives here so `--features wasm` builds without `cranelift`.
pub(crate) fn compute_element_size(
    obj_ty: sonatina_ir::Type,
    ctx: &sonatina_ir::module::ModuleCtx,
) -> usize {
    if let Some(cmpd) = obj_ty.resolve_compound(ctx) {
        match cmpd {
            sonatina_ir::types::CompoundType::Array { elem, .. } => {
                return ctx.size_of_unchecked(elem);
            }
            sonatina_ir::types::CompoundType::ObjRef(inner)
            | sonatina_ir::types::CompoundType::ConstRef(inner) => {
                return compute_element_size(inner, ctx);
            }
            _ => {}
        }
    }
    // Fallback: 32 bytes (i256 size)
    32
}
