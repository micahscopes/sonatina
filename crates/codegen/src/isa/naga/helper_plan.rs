//! Contextual helper ABI preparation, before any helper body is emitted.
//!
//! These are derived views tied to the current module and Naga type arena,
//! not serialized certificates or a second representation of control flow.

use std::{collections::HashMap, sync::Arc};

use sonatina_ir::{Module, module::FuncRef};

use super::{
    NagaArgumentSource, NagaFunctionMap, NagaFunctionVariant, NagaLiveArguments,
    NagaLogicalResultAbis, NagaMemoryAbi, NagaMemoryAbiTypes, NagaPackedArguments,
    NagaResourceCapabilities, NagaResourceVariantBindings, NagaResultAbi, WordKind,
    helper_naga_argument_abi, helper_naga_result_abi, pack_wide_naga_helper_arguments,
};

pub(super) struct PlannedHelperAbi {
    pub variant: NagaFunctionVariant,
    pub arguments: Vec<NagaArgumentSource>,
    pub packed_arguments: Option<NagaPackedArguments>,
    pub result: NagaResultAbi,
    pub memory: NagaMemoryAbi,
    pub body: Arc<HelperBodyPlan>,
    pub parameters: PhysicalHelperParameters,
}

/// The complete declared Naga parameters, including hidden memory transport.
/// Explicit parameters precede the heap/bump/trap suffix used by the emitter.
pub(super) struct PhysicalHelperParameters {
    pub arguments: Vec<naga::FunctionArgument>,
    pub explicit_count: u32,
}

/// Instruction/control-flow eligibility only, not a complete callable ABI.
/// Resource identity, types, argument packing, and transitive memory transport
/// still require the contextual planner. Recompute after changing the module.
pub struct HelperBodyPlan {
    pub(super) structured: crate::structurize::StructuredCfg,
    instruction_count: usize,
    accesses_resource: bool,
}

impl HelperBodyPlan {
    pub fn instruction_count(&self) -> usize {
        self.instruction_count
    }

    pub fn accesses_resource(&self) -> bool {
        self.accesses_resource
    }
}

/// Query the same body closure consumed by helper emission. The backend does
/// not accept this value as an external certificate; compilation rederives it.
pub fn analyze_helper_body(
    module: &Module,
    function: FuncRef,
) -> Result<HelperBodyPlan, super::SpirvError> {
    derive_helper_body(module, function).map_err(super::SpirvError::Translation)
}

fn derive_helper_body(module: &Module, function_ref: FuncRef) -> Result<HelperBodyPlan, String> {
    use sonatina_ir::{InstDowncast, inst::data};

    let signature = module
        .ctx
        .get_sig(function_ref)
        .ok_or_else(|| format!("spirv: helper {function_ref:?} has no signature"))?;
    module.func_store.try_view(function_ref, |function| {
        let inst_set = function.inst_set();
        let mut instruction_count = 0;
        let mut accesses_resource = false;
        for block in function.layout.iter_block() {
            for instruction in function.layout.iter_inst(block) {
                instruction_count += 1;
                let inst = function.dfg.inst(instruction);
                if !super::spirv_instruction_is_lowered(inst_set, inst) {
                    return Err(format!(
                        "spirv: instruction `{}` is unsupported in helper `{}`. Fail closed.",
                        inst.as_text(), signature.name(),
                    ));
                }
                if <&data::MemAllocDynamic as InstDowncast>::downcast(inst_set, inst).is_some()
                    || <&data::MemCheckpoint as InstDowncast>::downcast(inst_set, inst).is_some()
                    || <&data::MemRewind as InstDowncast>::downcast(inst_set, inst).is_some()
                    || <&data::Memcopy as InstDowncast>::downcast(inst_set, inst).is_some()
                    || <&data::ObjAlloc as InstDowncast>::downcast(inst_set, inst).is_some()
                {
                    return Err(format!(
                        "spirv: helper `{}` changes arena lifetime or object lifetime across a call. Fail closed.",
                        signature.name(),
                    ));
                }
                accesses_resource |= <&data::ObjLoad as InstDowncast>::downcast(inst_set, inst).is_some()
                    || <&data::ObjStore as InstDowncast>::downcast(inst_set, inst).is_some();
            }
        }
        let structured = crate::structurize::structurize_function(function)
            .map_err(|error| super::structurize_error_with_block_ir(error, function_ref, function))?;
        Ok(HelperBodyPlan { structured, instruction_count, accesses_resource })
    }).ok_or_else(|| format!("spirv: helper {function_ref:?} has no body. Fail closed."))?
}

/// Preserve call-postorder and resource-variant order. All type interning and
/// physical ABI adaptation complete before instruction emission starts. Body
/// legality and final Naga validation remain separate required gates.
#[allow(clippy::too_many_arguments)]
pub(super) fn plan_helper_abis(
    module: &Module,
    call_order: &[FuncRef],
    roots: &[FuncRef],
    word: WordKind,
    word_type: naga::Handle<naga::Type>,
    f32_type: naga::Handle<naga::Type>,
    bool_type: naga::Handle<naga::Type>,
    resource_capabilities: &NagaResourceCapabilities,
    logical_results: &NagaLogicalResultAbis,
    resource_variants: &NagaResourceVariantBindings,
    memory_abis: &HashMap<FuncRef, NagaMemoryAbi>,
    memory_types: NagaMemoryAbiTypes,
    live_arguments: &NagaLiveArguments,
    functions: &NagaFunctionMap,
    types: &mut naga::UniqueArena<naga::Type>,
) -> Result<Vec<PlannedHelperAbi>, String> {
    let mut plans = Vec::new();
    for function in call_order
        .iter()
        .copied()
        .filter(|function| !roots.contains(function))
    {
        let memory = memory_abis.get(&function).copied().ok_or_else(|| {
            format!("spirv: helper {function:?} has no derived private-memory ABI. Fail closed.")
        })?;
        let signature = module
            .ctx
            .get_sig(function)
            .ok_or_else(|| format!("spirv: helper {function:?} has no signature"))?;
        let body = Arc::new(derive_helper_body(module, function)?);
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
        let variants = resource_variants
            .get(&function)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
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
            let parameters = plan_physical_parameters(
                module,
                &signature,
                word,
                word_type,
                f32_type,
                bool_type,
                &arguments,
                packed_arguments.as_ref(),
                memory,
                memory_types,
                functions,
            )?;
            plans.push(PlannedHelperAbi {
                variant,
                arguments,
                packed_arguments,
                result: result.clone(),
                memory,
                body: Arc::clone(&body),
                parameters,
            });
        }
    }
    Ok(plans)
}

#[allow(clippy::too_many_arguments)]
fn plan_physical_parameters(
    module: &Module,
    signature: &sonatina_ir::Signature,
    word: WordKind,
    word_type: naga::Handle<naga::Type>,
    f32_type: naga::Handle<naga::Type>,
    bool_type: naga::Handle<naga::Type>,
    argument_abi: &[NagaArgumentSource],
    packed_arguments: Option<&NagaPackedArguments>,
    memory_abi: NagaMemoryAbi,
    memory_types: NagaMemoryAbiTypes,
    naga_functions: &NagaFunctionMap,
) -> Result<PhysicalHelperParameters, String> {
    if argument_abi.len() != signature.args().len() {
        return Err(format!(
            "spirv: helper `{}` has {} logical arguments but {} ABI entries. Fail closed.",
            signature.name(),
            signature.args().len(),
            argument_abi.len(),
        ));
    }
    let mut arguments = Vec::new();
    if let Some(packed) = packed_arguments {
        if packed.physical_index != 0 {
            return Err(format!(
                "spirv: helper `{}` has noncanonical packed argument index {}. Fail closed.",
                signature.name(),
                packed.physical_index,
            ));
        }
        arguments.push(naga::FunctionArgument {
            name: Some("fe_arguments".to_string()),
            ty: packed.ty,
            binding: None,
        });
    }
    for (logical_index, (&ty, source)) in signature.args().iter().zip(argument_abi).enumerate() {
        let physical_index = match source {
            NagaArgumentSource::Physical(physical_index) => physical_index,
            NagaArgumentSource::Packed { .. }
            | NagaArgumentSource::ImplicitResource(_)
            | NagaArgumentSource::Dead => continue,
        };
        if *physical_index as usize != arguments.len() {
            return Err(format!(
                "spirv: helper `{}` has a noncanonical physical argument index {physical_index}. Fail closed.",
                signature.name(),
            ));
        }
        arguments.push(naga::FunctionArgument {
            name: Some(format!("a{logical_index}")),
            ty: super::helper_naga_type(
                &module.ctx,
                ty,
                word,
                word_type,
                f32_type,
                bool_type,
                &naga_functions.typed_local_types,
            )?,
            binding: None,
        });
    }
    let physical_argument_count = arguments.len() as u32;
    if memory_abi.heap {
        arguments.push(naga::FunctionArgument {
            name: Some("fe_heap".to_string()),
            ty: memory_types.heap.ok_or_else(|| {
                "spirv: helper private-arena ABI has no heap pointer type. Fail closed.".to_string()
            })?,
            binding: None,
        });
        arguments.push(naga::FunctionArgument {
            name: Some("fe_bump".to_string()),
            ty: memory_types.word.ok_or_else(|| {
                "spirv: helper private-arena ABI has no bump pointer type. Fail closed.".to_string()
            })?,
            binding: None,
        });
    }
    if memory_abi.trap {
        arguments.push(naga::FunctionArgument {
            name: Some("fe_trapped".to_string()),
            ty: memory_types.trap.ok_or_else(|| {
                "spirv: helper trap ABI has no trap pointer type. Fail closed.".to_string()
            })?,
            binding: None,
        });
    }
    Ok(PhysicalHelperParameters {
        arguments,
        explicit_count: physical_argument_count,
    })
}
