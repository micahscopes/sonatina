//! Contextual helper ABI preparation, before any helper body is emitted.
//!
//! These are derived views tied to the current module and Naga type arena,
//! not serialized certificates or a second representation of control flow.

use std::collections::HashMap;

use sonatina_ir::{Module, module::FuncRef};

use super::{
    NagaArgumentSource, NagaFunctionMap, NagaFunctionVariant, NagaLiveArguments,
    NagaLogicalResultAbis, NagaMemoryAbi, NagaPackedArguments, NagaResourceCapabilities,
    NagaResourceVariants, NagaResultAbi, WordKind, helper_naga_argument_abi,
    helper_naga_result_abi, pack_wide_naga_helper_arguments,
};

pub(super) struct PlannedHelperAbi {
    pub variant: NagaFunctionVariant,
    pub arguments: Vec<NagaArgumentSource>,
    pub packed_arguments: Option<NagaPackedArguments>,
    pub result: NagaResultAbi,
    pub memory: NagaMemoryAbi,
}

/// Preserve call-postorder and resource-variant order. All type interning and
/// physical ABI adaptation complete before instruction emission starts. Body
/// legality and final Naga validation remain separate required gates.
#[allow(clippy::too_many_arguments)]
pub(super) fn plan_helper_abis(
    module: &Module,
    call_order: &[FuncRef],
    entry: FuncRef,
    word: WordKind,
    word_type: naga::Handle<naga::Type>,
    f32_type: naga::Handle<naga::Type>,
    bool_type: naga::Handle<naga::Type>,
    resource_capabilities: &NagaResourceCapabilities,
    logical_results: &NagaLogicalResultAbis,
    resource_variants: &NagaResourceVariants,
    memory_abis: &HashMap<FuncRef, NagaMemoryAbi>,
    live_arguments: &NagaLiveArguments,
    functions: &NagaFunctionMap,
    types: &mut naga::UniqueArena<naga::Type>,
) -> Result<Vec<PlannedHelperAbi>, String> {
    let mut plans = Vec::new();
    for function in call_order
        .iter()
        .copied()
        .filter(|&function| function != entry)
    {
        let memory = memory_abis.get(&function).copied().ok_or_else(|| {
            format!("spirv: helper {function:?} has no derived private-memory ABI. Fail closed.")
        })?;
        let signature = module
            .ctx
            .get_sig(function)
            .ok_or_else(|| format!("spirv: helper {function:?} has no signature"))?;
        let result = helper_naga_result_abi(
            module,
            function,
            word,
            word_type,
            f32_type,
            bool_type,
            resource_capabilities,
            logical_results,
            functions,
            types,
        )?;
        let variants = resource_variants.variants(function);
        if variants.is_empty() {
            return Err(format!(
                "spirv: helper `{}` has no entry-rooted resource variant. Fail closed.",
                signature.name(),
            ));
        }
        for (index, bindings) in variants.iter().enumerate() {
            let variant = NagaFunctionVariant {
                function,
                ordinal: u32::try_from(index).map_err(|_| {
                    format!(
                        "spirv: helper `{}` has more resource variants than fit in u32. Fail closed.",
                        signature.name(),
                    )
                })?,
            };
            let mut arguments = helper_naga_argument_abi(
                module,
                function,
                &signature,
                word,
                word_type,
                f32_type,
                bool_type,
                resource_capabilities,
                bindings,
                live_arguments,
                functions,
            )?;
            let packed_arguments = pack_wide_naga_helper_arguments(
                module,
                &signature,
                word,
                word_type,
                f32_type,
                bool_type,
                memory,
                &mut arguments,
                functions,
                types,
            )?;
            plans.push(PlannedHelperAbi {
                variant,
                arguments,
                packed_arguments,
                result: result.clone(),
                memory,
            });
        }
    }
    Ok(plans)
}
