//! Contextual helper ABI preparation, before any helper body is emitted.
//!
//! These are derived views tied to the current module and Naga type arena,
//! not serialized certificates or a second representation of control flow.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helper_report_preserves_children_and_siblings_of_rejected_parent() {
        let parsed = sonatina_parser::parse_module(r#"
target = "wasm32-unknown-native"

func public %entry() -> i32 {
    block0:
        v0.i32 = call %ancestor;
        v1.i32 = call %sibling;
        return v1;
}
func private %ancestor() -> i32 {
    block0:
        v0.i256 = call %parent;
        return 31.i32;
}
func private %parent() -> i256 {
    block0:
        v0.i32 = call %leaf;
        return 0.i256;
}
func private %leaf() -> i32 {
    block0:
        return 17.i32;
}
func private %sibling() -> i32 {
    block0:
        return 23.i32;
}
"#).expect("partial ABI fixture parses");
        let module = &parsed.module;
        let find = |name| module.funcs().into_iter().find(|&function| {
            module.ctx.func_sig(function, |signature| signature.name() == name)
        }).expect("fixture function exists");
        let entry = find("entry");
        let parent = find("parent");
        let ancestor = find("ancestor");
        let leaf = find("leaf");
        let sibling = find("sibling");
        let order = super::super::reachable_call_postorder(module, entry).unwrap();
        assert!(order.iter().position(|&f| f == parent) < order.iter().position(|&f| f == sibling));
        let live = super::super::analyze_live_arguments(module);
        let context = EntryHelperContext::derive(module, &order, entry, &[], &live, false)
            .expect("logical context is independent of the unsupported physical return type");
        let mut types = naga::UniqueArena::new();
        let mut scalar = |kind, width| types.insert(naga::Type {
            name: None,
            inner: naga::TypeInner::Scalar(naga::Scalar { kind, width }),
        }, naga::Span::UNDEFINED);
        let word = scalar(naga::ScalarKind::Uint, 4);
        let float = scalar(naga::ScalarKind::Float, 4);
        let boolean = scalar(naga::ScalarKind::Bool, 1);
        let report = plan_helper_abis(
            module, &order, &[entry], WordKind::U32, word, float, boolean,
            &context.resources, &context.logical_results, &context.variants.bindings,
            &context.memory, NagaMemoryAbiTypes::default(), &live,
            &NagaFunctionMap::new(), &mut types,
        );
        assert_eq!(report.rejections.len(), 2);
        assert_eq!(report.rejections[0].0, parent);
        assert!(report.rejections[0].1.contains("unsupported"));
        assert_eq!(report.rejections[1].0, ancestor);
        assert!(report.rejections[1].1.contains("rejected or unplanned callee"));
        let planned = report.plans.iter().map(|plan| plan.variant.function).collect::<Vec<_>>();
        assert_eq!(planned, vec![leaf, sibling]);
        assert!(report.into_complete().is_err(), "partial reports cannot authorize emission");
    }
}

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

/// Per-function ABI outcomes in call-postorder. An unsupported parent must not
/// discard an independently representable child or prevent visiting siblings.
/// Successful plans also require all their direct callees to have succeeded.
/// Shared entry context and typed-local validation remain prerequisites.
pub(super) struct HelperAbiReport {
    pub plans: Vec<PlannedHelperAbi>,
    pub rejections: Vec<(FuncRef, String)>,
}

impl HelperAbiReport {
    /// Final emission requires every selected helper. Keep the existing first
    /// diagnostic ordering, and never emit only the successful subset.
    pub fn into_complete(self) -> Result<Vec<PlannedHelperAbi>, String> {
        match self.rejections.into_iter().next() {
            Some((_, error)) => Err(error),
            None => Ok(self.plans),
        }
    }
}

/// Entry-rooted facts shared by logical ABI planning and instruction emission.
/// Resource identity and transitive memory requirements are derived together,
/// before physical parameter types or helper bodies are constructed.
pub(super) struct EntryHelperContext {
    pub resources: NagaResourceCapabilities,
    pub logical_results: NagaLogicalResultAbis,
    pub variants: super::NagaResourceVariants,
    pub memory: HashMap<FuncRef, NagaMemoryAbi>,
}

impl EntryHelperContext {
    pub fn derive(
        module: &Module,
        call_order: &[FuncRef],
        entry: FuncRef,
        external_roots: &[(u32, naga::Handle<naga::GlobalVariable>)],
        live_arguments: &NagaLiveArguments,
        entry_owns_arena: bool,
    ) -> Result<Self, String> {
        let signature = module.ctx.get_sig(entry)
            .ok_or_else(|| format!("spirv: entry {entry:?} has no signature"))?;
        let resources = super::helper_resource_capabilities(module, &signature, external_roots)?;
        let logical_results = super::helper_naga_logical_result_abis(
            module, call_order, &[entry], &resources,
        ).into_complete()?;
        let variants = super::helper_resource_variants(
            module, call_order, entry, external_roots, &resources,
            &logical_results, live_arguments,
        )?;
        let memory = super::helper_private_memory_abis(module, call_order, entry)?;
        if memory.values().any(|abi| abi.heap) && !entry_owns_arena {
            return Err(
                "spirv: a reachable helper accesses the private arena, but the entry function owns no proven arena allocation. Fail closed."
                    .to_string(),
            );
        }
        Ok(Self { resources, logical_results, variants, memory })
    }
}


pub(super) struct PreparedEntryHelpers {
    pub context: EntryHelperContext,
    pub plans: Vec<PlannedHelperAbi>,
    pub functions: NagaFunctionMap,
    pub private_heap_type: Option<naga::Handle<naga::Type>>,
    pub needs_trap_channel: bool,
}

/// Prepare memory transport and physical helper ABIs together. No instruction
/// bodies or shader writers run here; emission consumes this derived result.
#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_entry_helpers(
    module: &Module,
    call_order: &[FuncRef],
    first_func: FuncRef,
    word: WordKind,
    word_type: naga::Handle<naga::Type>,
    f32_type: naga::Handle<naga::Type>,
    bool_type: naga::Handle<naga::Type>,
    context: EntryHelperContext,
    entry: &super::NagaEntryBodyPlan,
    private_heap_words: u32,
    live_arguments: &NagaLiveArguments,
    typed_local_types: HashMap<sonatina_ir::Type, super::NagaTypedLocalType>,
    types: &mut naga::UniqueArena<naga::Type>,
) -> Result<PreparedEntryHelpers, String> {
    let helper_memory = context.memory.values().any(|abi| abi.heap);
    let helper_trap = context.memory.values().any(|abi| abi.trap);
    let needs_trap_channel = entry.uses_arena || entry.may_trap || helper_trap;
    let private_heap_type = if entry.uses_arena {
        let heap_len = std::num::NonZeroU32::new(private_heap_words)
            .ok_or_else(|| "spirv: derived private heap must be nonzero".to_string())?;
        Some(types.insert(
            naga::Type {
                name: Some("FeHeap".into()),
                inner: naga::TypeInner::Array {
                    base: word_type,
                    size: naga::ArraySize::Constant(heap_len),
                    stride: 4,
                },
            },
            naga::Span::UNDEFINED,
        ))
    } else {
        None
    };
    let helper_memory_types = NagaMemoryAbiTypes {
        heap: if helper_memory {
            Some(types.insert(
                naga::Type {
                    name: None,
                    inner: naga::TypeInner::Pointer {
                        base: private_heap_type.expect("helper memory requires an entry heap"),
                        space: naga::AddressSpace::Function,
                    },
                },
                naga::Span::UNDEFINED,
            ))
        } else {
            None
        },
        word: if helper_memory {
            Some(types.insert(
                naga::Type {
                    name: None,
                    inner: naga::TypeInner::Pointer {
                        base: word_type,
                        space: naga::AddressSpace::Function,
                    },
                },
                naga::Span::UNDEFINED,
            ))
        } else {
            None
        },
        trap: if helper_trap {
            Some(types.insert(
                naga::Type {
                    name: None,
                    inner: naga::TypeInner::Pointer {
                        base: bool_type,
                        space: naga::AddressSpace::Function,
                    },
                },
                naga::Span::UNDEFINED,
            ))
        } else {
            None
        },
    };

    let naga_functions =
        NagaFunctionMap::with_typed_local_types(typed_local_types);
    let helper_plans = plan_helper_abis(
        module,
        call_order,
        &[first_func],
        word,
        word_type,
        f32_type,
        bool_type,
        &context.resources,
        &context.logical_results,
        &context.variants.bindings,
        &context.memory,
        helper_memory_types,
        live_arguments,
        &naga_functions,
        types,
    ).into_complete()?;

    Ok(PreparedEntryHelpers {
        context, plans: helper_plans, functions: naga_functions,
        private_heap_type, needs_trap_channel,
    })
}

/// Instruction/control-flow eligibility only, not a complete callable ABI.
/// Resource identity, types, argument packing, and transitive memory transport
/// still require the contextual planner. Recompute after changing the module.
pub struct HelperBodyPlan {
    pub(super) structured: crate::structurize::StructuredCfg,
    instruction_count: usize,
    accesses_resource: bool,
    callees: Vec<FuncRef>,
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
        let mut callees = Vec::new();
        for block in function.layout.iter_block() {
            for instruction in function.layout.iter_inst(block) {
                instruction_count += 1;
                let inst = function.dfg.inst(instruction);
                if let Some(call) = function.dfg.call_info(instruction) {
                    callees.push(call.callee());
                }
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
        Ok(HelperBodyPlan { structured, instruction_count, accesses_resource, callees })
    }).ok_or_else(|| format!("spirv: helper {function_ref:?} has no body. Fail closed."))?
}

/// Preserve call-postorder and resource-variant order, recording each helper's
/// outcome independently. Shared entry context must already be derived. Type
/// interning and physical ABI adaptation precede instruction emission; callers
/// must require a complete report before emitting the selected call graph.
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
) -> HelperAbiReport {
    let mut report = HelperAbiReport { plans: Vec::new(), rejections: Vec::new() };
    let mut callable = std::collections::HashSet::new();
    for function in call_order
        .iter()
        .copied()
        .filter(|function| !roots.contains(function))
    {
        // A rejected resource variant rejects the whole helper. Do not expose
        // a prefix of its variants as a callable function.
        let outcome = (|| -> Result<Vec<PlannedHelperAbi>, String> {
            let mut plans = Vec::new();
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
            Ok(plans)
        })();
        match outcome {
            Ok(plans) => {
                // Body plans are shared by every resource variant. A local
                // ABI is not callable while any selected callee is unresolved.
                let blocked = plans.first().into_iter().flat_map(|plan| &plan.body.callees)
                    .copied().find(|callee| !callable.contains(callee));
                if let Some(callee) = blocked {
                    report.rejections.push((function, format!(
                        "spirv: helper {function:?} requires rejected or unplanned callee {callee:?}. Fail closed.",
                    )));
                } else {
                    callable.insert(function);
                    report.plans.extend(plans);
                }
            }
            Err(error) => report.rejections.push((function, error)),
        }
    }
    report
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
