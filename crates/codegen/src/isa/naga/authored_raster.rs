//! Source-authored raster realization in the shared Naga backend.

use std::collections::HashMap;

use sonatina_ir::{Function, InstDowncast, Module, Type, Value};

use crate::optim::dead_arg::analyze_live_arguments;

use super::{
    Access, LayoutMode, Role, SpirvBinding, SpirvBindingMember, SpirvBuiltinArgument,
    SpirvBuiltinInput, SpirvBuiltinSource, SpirvExternalResource, SpirvLayout, SpirvRasterPipeline,
    SpirvScalarKind, SpirvShaderStage, WordKind, NagaFunctionInfo,
    NagaFunctionMap, NagaMemoryAbi, NagaMemoryAbiTypes,
    NagaResourceCapabilities, append_external_resources,
    emit_naga_regions, lower_naga_helper, reachable_call_postorder,
    resolve_naga_value, spirv_instruction_is_lowered, unsupported_signed_op_under_u32,
};

// Physical shader entry names are compiler-owned ABI, not user-authored
// behavior identity. Keeping them fixed and WGSL-safe prevents a backend
// writer from silently renaming otherwise valid source identifiers (Naga, for
// example, suffixes names ending in a digit). The source behavior names still
// select the Sonatina functions and remain available to higher-level manifests
// for provenance and scheduling.
const PHYSICAL_VERTEX_ENTRY: &str = "fe_vertex_main";
const PHYSICAL_FRAGMENT_ENTRY: &str = "fe_fragment_main";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raster_helper_preparation_preserves_independent_plans_without_emission() {
        // Exercise paired-root helper preparation, not the vertex/fragment
        // interface contract, which translate_entries validates separately.
        let parsed = sonatina_parser::parse_module(r#"
target = "wasm32-unknown-native"
func public %vertex() -> i32 {
    block0:
        v0.i32 = call %ancestor;
        return v0;
}
func public %fragment() -> i32 {
    block0:
        v0.i32 = call %sibling;
        return v0;
}
func private %ancestor() -> i32 {
    block0:
        v0.i256 = call %bad;
        return 0.i32;
}
func private %bad() -> i256 {
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
"#).expect("raster helper fixture parses");
        let module = &parsed.module;
        let find = |name| module.funcs().into_iter().find(|&function| {
            module.ctx.func_sig(function, |signature| signature.name() == name)
        }).unwrap();
        let roots = [find("vertex"), find("fragment")];
        let mut output = naga::Module::default();
        let word = scalar_type(&mut output, naga::ScalarKind::Uint, 4);
        let float = scalar_type(&mut output, naga::ScalarKind::Float, 4);
        let boolean = scalar_type(&mut output, naga::ScalarKind::Bool, 1);
        let prepared = prepare_scalar_helpers(
            module, &roots, &mut output.types, word, float, boolean,
        ).expect("preparation reports rejected helpers");
        let rejected = prepared.report.rejections.iter().map(|(f, _)| *f).collect::<Vec<_>>();
        assert_eq!(rejected, vec![find("bad"), find("ancestor")]);
        let planned = prepared.report.plans.iter().map(|plan| plan.variant.function).collect::<Vec<_>>();
        assert_eq!(planned, vec![find("leaf"), find("sibling")]);
        assert!(output.functions.is_empty());
        assert!(output.entry_points.is_empty());
        assert!(prepared.report.into_complete().is_err());
        assert!(append_scalar_helpers(module, &roots, &mut output, word, float, boolean).is_err());
        assert!(output.functions.is_empty(), "partial plans must not emit even a valid prefix");
    }

    #[test]
    fn raster_entry_preparation_preserves_state_and_checks_pairing() {
        let parsed = sonatina_parser::parse_module(r#"
target = "wasm32-unknown-native"
func public %vertex(v0.i32, v1.f32, v2.i32) -> (f32, f32, f32, f32, f32) {
    block0:
        return (v1, 0x00000000.f32, 0x00000000.f32, 0x3f800000.f32, v1);
}
func public %fragment(v0.f32, v1.f32, v2.i32) -> i32 {
    block0:
        return v2;
}
func public %mismatch(v0.f32, v1.i32, v2.f32) -> i32 {
    block0:
        return v1;
}
"#).expect("paired-entry fixture parses");
        let module = &parsed.module;
        let find = |name| module.funcs().into_iter().find(|&function| {
            module.ctx.func_sig(function, |signature| signature.name() == name)
        }).unwrap();
        let vertex = find("vertex");
        let fragment = find("fragment");
        let prepared = prepare_raster_entries(module, vertex, fragment, &[], &[])
            .expect("paired entry preparation");
        assert!(prepared.vertex.normalized.is_none());
        assert_eq!(prepared.vertex.return_arguments.len(), 5);
        assert!(!prepared.vertex.structured.regions.is_empty());
        assert_eq!(prepared.fragment.return_arguments.len(), 1);
        assert!(!prepared.fragment.structured.regions.is_empty());
        assert_eq!(prepared.context_count, 1);
        assert_eq!(prepared.varying_count, 1);
        assert_eq!(prepared.scalar_state, vec![(0, Type::F32), (1, Type::I32)]);
        assert_eq!(prepared.state_stages, vec![SpirvShaderStage::Vertex, SpirvShaderStage::Fragment]);
        assert_eq!(prepared.builtin_arguments[0].source, SpirvBuiltinSource::VertexIndex);
        assert!(prepared.emitted_external_resources.is_empty());
        assert!(prepare_raster_entries(module, vertex, vertex, &[], &[])
            .err().unwrap().contains("must be distinct"));
        assert!(prepare_raster_entries(module, vertex, find("mismatch"), &[], &[])
            .err().unwrap().contains("actor-state suffixes differ"));
        assert!(prepare_raster_entries(module, vertex, fragment, &[], &[SpirvBuiltinArgument {
            arg_index: 0, source: SpirvBuiltinSource::InstanceIndex,
        }]).err().unwrap().contains("builtin context"));
    }
}

struct RasterEntryPreparation {
    vertex: RasterStagePlan,
    fragment: RasterStagePlan,
    pipeline: SpirvRasterPipeline,
    builtin_arguments: Vec<SpirvBuiltinArgument>,
    context_count: usize,
    varying_count: usize,
    scalar_state: Vec<(usize, Type)>,
    state_stages: Vec<SpirvShaderStage>,
    emitted_external_resources: Vec<SpirvExternalResource>,
    external_resource_stages: Vec<Vec<SpirvShaderStage>>,
}

/// Own the normalized body together with the control-flow and return IDs derived
/// from it. Emission must not reconstruct these decisions against another body.
struct RasterStagePlan {
    normalized: Option<Function>,
    structured: crate::structurize::StructuredCfg,
    return_arguments: Vec<sonatina_ir::ValueId>,
}

fn prepare_raster_stage(
    module: &Module,
    root: sonatina_ir::module::FuncRef,
    stage: &str,
    expected_leaves: usize,
) -> Result<RasterStagePlan, String> {
    module.func_store.try_view(root, |source| {
        let normalized = normalize_stage_to_single_exit(source)?;
        let function = normalized.as_ref().unwrap_or(source);
        let structured = crate::structurize::structurize_function(function)?;
        let return_arguments = unique_source_return_arguments(function)?
            .ok_or_else(|| format!("spirv raster: {stage} normalization did not produce one exit"))?;
        if return_arguments.len() != expected_leaves {
            return Err(format!(
                "spirv raster: {stage} returned {} leaves; expected {expected_leaves}",
                return_arguments.len(),
            ));
        }
        Ok(RasterStagePlan { normalized, structured, return_arguments })
    }).ok_or_else(|| format!("spirv raster: {stage} body is unavailable"))?
}

fn prepare_raster_entries(
    module: &Module,
    vertex_ref: sonatina_ir::module::FuncRef,
    fragment_ref: sonatina_ir::module::FuncRef,
    external_resources: &[SpirvExternalResource],
    builtin_arguments: &[SpirvBuiltinArgument],
) -> Result<RasterEntryPreparation, String> {
    if vertex_ref == fragment_ref {
        return Err("spirv raster: vertex and fragment entries must be distinct".to_string());
    }
    let functions = module.funcs();
    for (stage, entry) in [("vertex", vertex_ref), ("fragment", fragment_ref)] {
        if !functions.contains(&entry) {
            return Err(format!("spirv raster: selected {stage} entry {entry:?} is not defined in this module"));
        }
    }
    let vertex_sig = module.ctx.get_sig(vertex_ref)
        .ok_or_else(|| "spirv raster: vertex entry has no signature".to_string())?;
    let fragment_sig = module.ctx.get_sig(fragment_ref)
        .ok_or_else(|| "spirv raster: fragment entry has no signature".to_string())?;
    // Names are diagnostic and emitted-symbol metadata only. Stage identity was
    // selected and checked above using the module's function handles.
    let pipeline = SpirvRasterPipeline {
        vertex_entry: vertex_sig.name().to_owned(),
        fragment_entry: fragment_sig.name().to_owned(),
    };

    let implicit_vertex_index = [SpirvBuiltinArgument {
        arg_index: 0,
        source: SpirvBuiltinSource::VertexIndex,
    }];
    let builtin_arguments = if builtin_arguments.is_empty() {
        implicit_vertex_index.as_slice()
    } else {
        builtin_arguments
    };
    match builtin_arguments {
        [SpirvBuiltinArgument {
            arg_index: 0,
            source: SpirvBuiltinSource::VertexIndex,
        }] => {}
        [SpirvBuiltinArgument {
            arg_index: 0,
            source: SpirvBuiltinSource::VertexIndex,
        }, SpirvBuiltinArgument {
            arg_index: 1,
            source: SpirvBuiltinSource::InstanceIndex,
        }] => {}
        _ => {
            return Err(
                "spirv raster: builtin context must be vertex-index arg 0, optionally followed by instance-index arg 1"
                    .to_string(),
            );
        }
    }
    let context_count = builtin_arguments.len();
    if vertex_sig.args().len() < context_count
        || vertex_sig.args()[..context_count]
            .iter()
            .any(|ty| *ty != Type::I32)
    {
        return Err(format!(
            "spirv raster: vertex builtin context requires {context_count} u32/i32 carriers",
        ));
    }
    if vertex_sig.ret_tys().len() < 5 || vertex_sig.ret_tys().iter().any(|ty| *ty != Type::F32) {
        return Err(format!(
            "spirv raster: vertex `{}` must return four f32 clip-position leaves followed by at least one f32 varying; got {:?}",
            pipeline.vertex_entry, vertex_sig.ret_tys(),
        ));
    }
    let varying_count = vertex_sig.ret_tys().len() - 4;
    if fragment_sig.args().len() < varying_count
        || fragment_sig.args()[..varying_count].iter().any(|ty| *ty != Type::F32)
    {
        return Err(format!(
            "spirv raster: fragment `{}` does not consume the vertex function's {varying_count} flattened f32 varyings",
            pipeline.fragment_entry,
        ));
    }
    if fragment_sig.single_ret_ty() != Some(Type::I32) {
        return Err(format!(
            "spirv raster: fragment `{}` must return one packed u32/i32 color",
            pipeline.fragment_entry,
        ));
    }
    let vertex_state = &vertex_sig.args()[context_count..];
    let fragment_state = &fragment_sig.args()[varying_count..];
    if vertex_state != fragment_state {
        return Err(format!(
            "spirv raster: vertex and fragment actor-state suffixes differ: {:?} versus {:?}",
            vertex_state, fragment_state,
        ));
    }
    // `arg_index` names the vertex-entry argument. Both authored entries carry
    // the same actor-state suffix, so the corresponding fragment argument is
    // derived from that suffix position instead of guessed from scalar types.
    let mut resource_state_positions = Vec::with_capacity(external_resources.len());
    for resource in external_resources {
        let Some(state_position) = resource
            .arg_index
            .checked_sub(context_count as u32)
            .map(|index| index as usize)
        else {
            return Err(format!(
                "spirv raster: external resource {} cannot replace the vertex builtin prefix",
                resource.name
            ));
        };
        if state_position >= vertex_state.len() {
            return Err(format!(
                "spirv raster: external resource {} names absent vertex arg {}",
                resource.name, resource.arg_index,
            ));
        }
        if resource_state_positions.contains(&state_position) {
            return Err(format!(
                "spirv raster: multiple external resources replace actor-state slot {state_position}",
            ));
        }
        resource_state_positions.push(state_position);
    }
    let declared_scalar_state = vertex_state
        .iter()
        .copied()
        .enumerate()
        .filter(|(index, _)| !resource_state_positions.contains(index))
        .collect::<Vec<_>>();
    if declared_scalar_state
        .iter()
        .any(|(_, ty)| !matches!(ty, Type::I32 | Type::F32))
    {
        return Err(
            "spirv raster: non-resource actor state admits only i32/u32 and f32 leaves"
            .to_string(),
        );
    }

    // Vertex and fragment share one physical bind group but have distinct
    // entry signatures. Retain the union of resources that can reach either
    // stage, mapped through the common actor-state suffix. Every declared
    // resource position remains excluded from scalar state above.
    let live_arguments = analyze_live_arguments(module);
    let vertex_live = live_arguments.get(&vertex_ref).map(Vec::as_slice);
    let fragment_live = live_arguments.get(&fragment_ref).map(Vec::as_slice);
    let is_live = |mask: Option<&[bool]>, argument: usize| {
        mask.and_then(|mask| mask.get(argument))
            .copied()
            .unwrap_or(true)
    };
    // The scalar record is the complete actor-state transport, not merely a
    // shader-local optimization detail. Browser controls and resident Fe
    // transitions preserve that record across frames, including fields that
    // a particular raster pair does not read. Keep its semantic shape stable;
    // only external GPU resources are eligible for physical liveness pruning.
    let scalar_state = declared_scalar_state;
    let state_stages = [
        (
            SpirvShaderStage::Vertex,
            scalar_state
                .iter()
                .any(|(state_position, _)| is_live(vertex_live, context_count + *state_position)),
        ),
        (
            SpirvShaderStage::Fragment,
            scalar_state
                .iter()
                .any(|(state_position, _)| is_live(fragment_live, varying_count + *state_position)),
        ),
    ]
    .into_iter()
    .filter_map(|(stage, used)| used.then_some(stage))
    .collect::<Vec<_>>();
    let emitted_external_resources = external_resources
        .iter()
        .zip(resource_state_positions.iter().copied())
        .filter_map(|(resource, state_position)| {
            let vertex_used = is_live(vertex_live, resource.arg_index as usize);
            let fragment_arg = varying_count + state_position;
            let fragment_used = is_live(fragment_live, fragment_arg);
            (vertex_used || fragment_used).then(|| {
                let stages = [
                    (SpirvShaderStage::Vertex, vertex_used),
                    (SpirvShaderStage::Fragment, fragment_used),
                ]
                .into_iter()
                .filter_map(|(stage, used)| used.then_some(stage))
                .collect::<Vec<_>>();
                (resource.clone(), stages)
            })
        })
        .enumerate()
        .map(|(binding, (mut resource, stages))| {
            resource.group = 0;
            resource.binding = binding as u32;
            (resource, stages)
        })
        .collect::<Vec<_>>();
    let (emitted_external_resources, external_resource_stages): (Vec<_>, Vec<_>) =
        emitted_external_resources.into_iter().unzip();

    // Access restrictions belong to the emitted stage interface. An actor may
    // own compute-only writable/atomic storage that neither raster root reaches.
    // Still validate every declared argument position above, and reject writable
    // resources used by either stage rather than coercing their access to Read.
    if let Some(resource) = emitted_external_resources
        .iter()
        .find(|resource| resource.access != Access::Read)
    {
        return Err(format!(
            "spirv raster: external resource {} must be read-only",
            resource.name,
        ));
    }

    // Fe retains only the scalar, memory-free portion of the helper graph.
    // Object indexing/projection/load remains in the paired stage roots and
    // resolves through their shared external globals. Allocation, pointers,
    // private memory and traps still have no stage-paired helper channel.
    for (stage, func_ref) in [("vertex", vertex_ref), ("fragment", fragment_ref)] {
        module.func_store.try_view(func_ref, |function| -> Result<(), String> {
            let inst_set = function.inst_set();
            for block in function.layout.iter_block() {
                for inst in function.layout.iter_inst(block) {
                    let data = function.dfg.inst(inst);
                    if !spirv_instruction_is_lowered(inst_set, data) {
                        return Err(format!(
                            "spirv raster: {stage} instruction `{}` is unsupported",
                            data.as_text(),
                        ));
                    }
                    let has_untransported_effect =
                        <&sonatina_ir::inst::data::ObjAlloc as InstDowncast>::downcast(inst_set, data).is_some()
                        || <&sonatina_ir::inst::data::MemAllocDynamic as InstDowncast>::downcast(inst_set, data).is_some()
                        || <&sonatina_ir::inst::data::Mload as InstDowncast>::downcast(inst_set, data).is_some()
                        || <&sonatina_ir::inst::data::Mstore as InstDowncast>::downcast(inst_set, data).is_some()
                        || <&sonatina_ir::inst::control_flow::Unreachable as InstDowncast>::downcast(inst_set, data).is_some();
                    if has_untransported_effect {
                        return Err(format!(
                            "spirv raster: {stage} body uses allocation/private-memory/trap operations that have no raster-stage channel; fail closed",
                        ));
                    }
                    let has_external_object_effect =
                        <&sonatina_ir::inst::data::ObjLoad as InstDowncast>::downcast(inst_set, data).is_some()
                        || <&sonatina_ir::inst::data::ObjAtomicAdd as InstDowncast>::downcast(inst_set, data).is_some()
                        || <&sonatina_ir::inst::data::ObjAtomicUMin as InstDowncast>::downcast(inst_set, data).is_some()
                        || <&sonatina_ir::inst::data::ObjAtomicLoad as InstDowncast>::downcast(inst_set, data).is_some()
                        || <&sonatina_ir::inst::data::ObjAtomicStore as InstDowncast>::downcast(inst_set, data).is_some()
                        || <&sonatina_ir::inst::data::ObjStore as InstDowncast>::downcast(inst_set, data).is_some()
                        || <&sonatina_ir::inst::data::ObjIndex as InstDowncast>::downcast(inst_set, data).is_some()
                        || <&sonatina_ir::inst::data::ObjProj as InstDowncast>::downcast(inst_set, data).is_some();
                    if has_external_object_effect && emitted_external_resources.is_empty() {
                        return Err(format!(
                            "spirv raster: {stage} body uses object operations without an external resource root; fail closed",
                        ));
                    }
                    if let Some(op) = unsupported_signed_op_under_u32(inst_set, data) {
                        return Err(format!(
                            "spirv raster: signedness-sensitive `{op}` is unsupported under the u32 browser carrier",
                        ));
                    }
                    let divisor = <&sonatina_ir::inst::arith::Udiv as InstDowncast>::downcast(inst_set, data)
                        .map(|op| *op.rhs())
                        .or_else(|| <&sonatina_ir::inst::arith::Umod as InstDowncast>::downcast(inst_set, data).map(|op| *op.rhs()));
                    if let Some(divisor) = divisor {
                        let nonzero = function.dfg.value_imm(divisor).is_some_and(|value| {
                            match value {
                                sonatina_ir::Immediate::I8(value) => value != 0,
                                sonatina_ir::Immediate::I32(value) => value != 0,
                                sonatina_ir::Immediate::I64(value) => value != 0,
                                _ => false,
                            }
                        });
                        if !nonzero {
                            return Err(format!(
                                "spirv raster: {stage} integer division/remainder requires a statically nonzero divisor because raster stages have no trap channel",
                            ));
                        }
                    }
                }
            }
            Ok(())
        }).ok_or_else(|| format!("spirv raster: {stage} body is unavailable"))??;
    }
    let vertex = prepare_raster_stage(module, vertex_ref, "vertex", 4 + varying_count)?;
    let fragment = prepare_raster_stage(module, fragment_ref, "fragment", 1)?;
    Ok(RasterEntryPreparation {
        vertex, fragment,
        pipeline, context_count, varying_count, scalar_state, state_stages,
        emitted_external_resources, external_resource_stages,
        builtin_arguments: builtin_arguments.to_vec(),
    })
}

pub(super) fn translate_entries(
    module: &Module,
    vertex_ref: sonatina_ir::module::FuncRef,
    fragment_ref: sonatina_ir::module::FuncRef,
    external_resources: &[SpirvExternalResource],
    builtin_arguments: &[SpirvBuiltinArgument],
) -> Result<(naga::Module, SpirvLayout), String> {
    let RasterEntryPreparation {
        vertex: vertex_plan, fragment: fragment_plan,
        pipeline, context_count, varying_count, scalar_state, state_stages,
        emitted_external_resources, external_resource_stages,
        builtin_arguments,
    } = prepare_raster_entries(module, vertex_ref, fragment_ref, external_resources, builtin_arguments)?;
    let pipeline = &pipeline;
    let builtin_arguments = builtin_arguments.as_slice();

    let mut naga_mod = naga::Module::default();
    let u32_type = scalar_type(&mut naga_mod, naga::ScalarKind::Uint, 4);
    let f32_type = scalar_type(&mut naga_mod, naga::ScalarKind::Float, 4);
    let bool_type = scalar_type(&mut naga_mod, naga::ScalarKind::Bool, 1);
    let vec4f = naga_mod.types.insert(
        naga::Type {
            name: None,
            inner: naga::TypeInner::Vector {
                size: naga::VectorSize::Quad,
                scalar: naga::Scalar { kind: naga::ScalarKind::Float, width: 4 },
            },
        },
        naga::Span::UNDEFINED,
    );

    let (external_roots, mut bindings) = append_external_resources(
        &mut naga_mod,
        &emitted_external_resources,
        &external_resource_stages,
        WordKind::U32,
        u32_type,
        f32_type,
    )?;

    let (state_var, state_span, layout_members) = append_state_binding(
        &mut naga_mod,
        &scalar_state,
        u32_type,
        f32_type,
        varying_count,
        emitted_external_resources.len() as u32,
    );
    let output_type = append_vertex_output_type(&mut naga_mod, f32_type, vec4f, varying_count);
    let (mut naga_functions, mut root_call_sites) = append_scalar_helpers(
        module,
        &[vertex_ref, fragment_ref],
        &mut naga_mod,
        u32_type,
        f32_type,
        bool_type,
    )?;
    naga_functions.set_atomic_resources(&emitted_external_resources, &external_roots);
    naga_functions.replace_call_sites(
        root_call_sites.remove(&vertex_ref).ok_or_else(|| {
            "spirv raster: vertex root has no derived helper call-site map".to_string()
        })?,
    );
    let vertex = lower_vertex(
        module, vertex_ref, &vertex_plan, pipeline, state_var, &scalar_state, &external_roots,
        builtin_arguments, u32_type, f32_type, bool_type, vec4f, output_type, &naga_functions,
    )?;
    naga_functions.replace_call_sites(
        root_call_sites.remove(&fragment_ref).ok_or_else(|| {
            "spirv raster: fragment root has no derived helper call-site map".to_string()
        })?,
    );
    let fragment = lower_fragment(
        module, fragment_ref, &fragment_plan, pipeline, state_var, &scalar_state, &external_roots, varying_count,
        context_count, u32_type, f32_type, bool_type, vec4f, &naga_functions,
    )?;

    naga_mod.entry_points.push(entry_point(
        PHYSICAL_VERTEX_ENTRY.to_string(), naga::ShaderStage::Vertex, vertex,
    ));
    naga_mod.entry_points.push(entry_point(
        PHYSICAL_FRAGMENT_ENTRY.to_string(), naga::ShaderStage::Fragment, fragment,
    ));

    if state_var.is_some() {
        bindings.push(SpirvBinding {
            group: 0,
            binding: emitted_external_resources.len() as u32,
            name: "state".to_string(),
            access: Access::Read,
            role: Role::Input,
            stages: state_stages,
            stride: state_span,
            span: state_span,
            members: layout_members,
            resource_element: None,
            resource_length: None,
            resource_arg_index: None,
        });
    }

    Ok((naga_mod, SpirvLayout {
        entry_point: PHYSICAL_FRAGMENT_ENTRY.to_string(),
        mode: LayoutMode::Render,
        workgroup_size: [0, 0, 0],
        word: WordKind::U32,
        bindings,
        builtin_inputs: builtin_arguments
            .iter()
            .map(|argument| SpirvBuiltinInput {
                arg_index: argument.arg_index,
                source: argument.source,
                scalar: SpirvScalarKind::U32,
            })
            .collect(),
        result: None,
        trap: None,
        vertex_entry: Some(PHYSICAL_VERTEX_ENTRY.to_string()),
        fragment_entry: Some(PHYSICAL_FRAGMENT_ENTRY.to_string()),
        color_target_format: Some("rgba8unorm".to_string()),
    }))
}

/// Materialize the ordinary scalar helper closure shared by both raster stages.
///
/// WebGPU functions cannot receive storage-resource pointers, so resource
/// identity remains a property of the paired entry roots. This first authored
/// raster helper ABI is intentionally smaller than the general compute ABI:
/// booleans, u32/f32 scalars, flattened multi-results, and no memory effects.
/// The compiler rejects every wider shape rather than silently inlining it in
/// the backend or smuggling state through a generated global.
struct PreparedRasterHelpers {
    functions: NagaFunctionMap,
    resources: NagaResourceCapabilities,
    logical_results: super::NagaLogicalResultAbis,
    report: super::helper_plan::HelperAbiReport,
}

pub(super) fn analyze_helpers(
    module: &Module,
    vertex: sonatina_ir::module::FuncRef,
    fragment: sonatina_ir::module::FuncRef,
    resources: &[SpirvExternalResource],
    builtins: &[SpirvBuiltinArgument],
) -> Result<super::ShaderHelperAnalysis, String> {
    let _entries = prepare_raster_entries(module, vertex, fragment, resources, builtins)?;
    let mut types = naga::Module::default();
    let word = scalar_type(&mut types, naga::ScalarKind::Uint, 4);
    let float = scalar_type(&mut types, naga::ScalarKind::Float, 4);
    let boolean = scalar_type(&mut types, naga::ScalarKind::Bool, 1);
    Ok(prepare_scalar_helpers(
        module, &[vertex, fragment], &mut types.types, word, float, boolean,
    )?.report.into_analysis())
}

fn prepare_scalar_helpers(
    module: &Module,
    roots: &[sonatina_ir::module::FuncRef],
    types: &mut naga::UniqueArena<naga::Type>,
    u32_type: naga::Handle<naga::Type>,
    f32_type: naga::Handle<naga::Type>,
    bool_type: naga::Handle<naga::Type>,
) -> Result<PreparedRasterHelpers, String> {
    use sonatina_ir::InstDowncast;

    let root_set = roots.iter().copied().collect::<std::collections::HashSet<_>>();
    let mut call_order = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for &root in roots {
        for function in reachable_call_postorder(module, root)? {
            if seen.insert(function) {
                call_order.push(function);
            }
        }
    }

    let scalar = |ty: Type| matches!(ty, Type::I1 | Type::I32 | Type::F32);
    let mut resource_bindings = super::NagaResourceVariantBindings::new();
    let mut memory_abis = HashMap::new();
    let mut live_arguments = super::NagaLiveArguments::default();
    let mut rejections = Vec::new();
    for &function_ref in call_order.iter().filter(|function| !root_set.contains(function)) {
        let outcome = (|| -> Result<(), String> {
            let signature = module
                .ctx
                .get_sig(function_ref)
                .ok_or_else(|| format!("spirv raster: helper {function_ref:?} has no signature"))?;
            if !signature.args().iter().copied().all(scalar)
                || !signature.ret_tys().iter().copied().all(scalar)
            {
                return Err(format!(
                    "spirv raster: helper `{}` crosses the scalar helper ABI with {:?} -> {:?}; fail closed",
                    signature.name(),
                    signature.args(),
                    signature.ret_tys(),
                ));
            }
            module
                .func_store
                .try_view(function_ref, |function| -> Result<(), String> {
                    for block in function.layout.iter_block() {
                        for instruction in function.layout.iter_inst(block) {
                            let data = function.dfg.inst(instruction);
                            if data.declared_effect_hint().has_memory_effect()
                                && function.dfg.call_info(instruction).is_none()
                            {
                                return Err(format!(
                                    "spirv raster: helper `{}` has memory effect `{}` outside the scalar helper ABI; fail closed",
                                    signature.name(),
                                    data.as_text(),
                                ));
                            }
                            if function
                                .dfg
                                .inst_results(instruction)
                                .iter()
                                .copied()
                                .any(|value| !scalar(function.dfg.value_ty(value)))
                                || data
                                    .collect_values()
                                    .into_iter()
                                    .any(|value| !scalar(function.dfg.value_ty(value)))
                            {
                                return Err(format!(
                                    "spirv raster: helper `{}` contains a non-scalar value outside the scalar helper ABI; fail closed",
                                    signature.name(),
                                ));
                            }
                            if <&sonatina_ir::inst::control_flow::CallIndirect as InstDowncast>::downcast(
                                function.inst_set(),
                                data,
                            )
                            .is_some()
                            {
                                return Err(format!(
                                    "spirv raster: helper `{}` contains an indirect call; fail closed",
                                    signature.name(),
                                ));
                            }
                        }
                    }
                    Ok(())
                })
                .ok_or_else(|| {
                    format!("spirv raster: helper `{}` has no body", signature.name())
                })??;
            // This validated context has no resource or memory transport and
            // retains every scalar argument. It supplies facts to the common ABI
            // planner rather than constructing a separate physical convention.
            resource_bindings.insert(function_ref, vec![vec![None; signature.args().len()]]);
            memory_abis.insert(function_ref, NagaMemoryAbi::default());
            live_arguments.insert(function_ref, vec![true; signature.args().len()]);
            Ok(())
        })();
        if let Err(error) = outcome {
            rejections.push((function_ref, error));
        }
    }

    let resource_capabilities = NagaResourceCapabilities::new();
    let logical_results = super::helper_naga_logical_result_abis(
        module, &call_order, roots, &resource_capabilities,
    ).into_complete()?;
    let naga_functions = NagaFunctionMap::new();
    let plans = super::helper_plan::plan_helper_abis(
        module,
        &call_order,
        roots,
        WordKind::U32,
        u32_type,
        f32_type,
        bool_type,
        &resource_capabilities,
        &logical_results,
        &resource_bindings,
        &memory_abis,
        NagaMemoryAbiTypes::default(),
        &live_arguments,
        &naga_functions,
        &rejections,
        types,
    );
    Ok(PreparedRasterHelpers {
        functions: naga_functions, resources: resource_capabilities,
        logical_results, report: plans,
    })
}

fn append_scalar_helpers(
    module: &Module,
    roots: &[sonatina_ir::module::FuncRef],
    naga_module: &mut naga::Module,
    u32_type: naga::Handle<naga::Type>,
    f32_type: naga::Handle<naga::Type>,
    bool_type: naga::Handle<naga::Type>,
) -> Result<
    (
        NagaFunctionMap,
        HashMap<
            sonatina_ir::module::FuncRef,
            HashMap<sonatina_ir::InstId, NagaFunctionInfo>,
        >,
    ),
    String,
> {
    fn call_sites(
        module: &Module,
        functions: impl IntoIterator<Item = sonatina_ir::module::FuncRef>,
        lowered: &HashMap<sonatina_ir::module::FuncRef, NagaFunctionInfo>,
    ) -> Result<HashMap<sonatina_ir::InstId, NagaFunctionInfo>, String> {
        let mut sites = HashMap::new();
        for function_ref in functions {
            module
                .func_store
                .try_view(function_ref, |function| -> Result<(), String> {
                    for block in function.layout.iter_block() {
                        for instruction in function.layout.iter_inst(block) {
                            let Some(call) = function.dfg.call_info(instruction) else {
                                continue;
                            };
                            let info = lowered.get(&call.callee()).ok_or_else(|| {
                                format!(
                                    "spirv raster: call to helper {:?} has no lowered scalar ABI; fail closed",
                                    call.callee(),
                                )
                            })?;
                            sites.insert(instruction, info.clone());
                        }
                    }
                    Ok(())
                })
                .ok_or_else(|| {
                    format!("spirv raster: function {function_ref:?} has no body")
                })??;
        }
        Ok(sites)
    }

    let PreparedRasterHelpers {
        functions: mut naga_functions, resources: resource_capabilities,
        logical_results, report,
    } = prepare_scalar_helpers(module, roots, &mut naga_module.types, u32_type, f32_type, bool_type)?;
    let plans = report.into_complete()?;
    let mut lowered = HashMap::<sonatina_ir::module::FuncRef, NagaFunctionInfo>::new();
    for super::helper_plan::PlannedHelperAbi {
        variant,
        arguments: argument_abi,
        packed_arguments,
        result: result_abi,
        memory: memory_abi,
        body: body_plan,
        parameters,
    } in plans
    {
        let function_ref = variant.function;
        naga_functions.replace_call_sites(call_sites(module, [function_ref], &lowered)?);
        let helper = lower_naga_helper(
            module,
            function_ref,
            &body_plan,
            WordKind::U32,
            u32_type,
            f32_type,
            bool_type,
            &argument_abi,
            packed_arguments.as_ref(),
            &result_abi,
            memory_abi,
            parameters,
            0,
            &resource_capabilities,
            &logical_results,
            &naga_functions,
        )?;
        let handle = naga_module.functions.append(helper, naga::Span::UNDEFINED);
        lowered.insert(
            function_ref,
            NagaFunctionInfo {
                handle,
                argument_abi,
                packed_arguments,
                result_abi,
                memory_abi,
            },
        );
    }
    // Instruction IDs are local to one Sonatina function. Keep each authored
    // stage root's call sites separate: merging the vertex and fragment maps
    // would allow an equal numeric ID in one stage to overwrite the other's
    // (possibly differently shaped) helper ABI.
    let root_call_sites = roots
        .iter()
        .copied()
        .map(|root| Ok((root, call_sites(module, [root], &lowered)?)))
        .collect::<Result<HashMap<_, _>, String>>()?;
    Ok((naga_functions, root_call_sites))
}

fn scalar_type(
    module: &mut naga::Module,
    kind: naga::ScalarKind,
    width: u8,
) -> naga::Handle<naga::Type> {
    module.types.insert(
        naga::Type { name: None, inner: naga::TypeInner::Scalar(naga::Scalar { kind, width }) },
        naga::Span::UNDEFINED,
    )
}

fn append_state_binding(
    module: &mut naga::Module,
    state: &[(usize, Type)],
    u32_type: naga::Handle<naga::Type>,
    f32_type: naga::Handle<naga::Type>,
    fragment_prefix: usize,
    binding: u32,
) -> (Option<naga::Handle<naga::GlobalVariable>>, u32, Vec<SpirvBindingMember>) {
    if state.is_empty() {
        return (None, 0, Vec::new());
    }
    let mut layout = Vec::with_capacity(state.len());
    let members = state.iter().enumerate().map(|(index, (state_index, ty))| {
        let (naga_ty, scalar) = match ty {
            Type::I32 => (u32_type, SpirvScalarKind::I32),
            Type::F32 => (f32_type, SpirvScalarKind::F32),
            _ => unreachable!("state admission checked by translate"),
        };
        let offset = index as u32 * 4;
        layout.push(SpirvBindingMember {
            arg_index: (fragment_prefix + *state_index) as u32,
            offset,
            width: 4,
            scalar,
        });
        naga::StructMember {
            name: Some(format!("p{}", fragment_prefix + *state_index)),
            ty: naga_ty,
            binding: None,
            offset,
        }
    }).collect();
    let span = state.len() as u32 * 4;
    let ty = module.types.insert(
        naga::Type {
            name: Some("RasterState".into()),
            inner: naga::TypeInner::Struct { members, span },
        },
        naga::Span::UNDEFINED,
    );
    let var = module.global_variables.append(
        naga::GlobalVariable {
            name: Some("state".into()),
            space: naga::AddressSpace::Storage { access: naga::StorageAccess::LOAD },
            binding: Some(naga::ResourceBinding { group: 0, binding }),
            ty,
            init: None,
            memory_decorations: naga::ir::MemoryDecorations::empty(),
        },
        naga::Span::UNDEFINED,
    );
    (Some(var), span, layout)
}

fn location_binding(location: u32) -> naga::Binding {
    naga::Binding::Location {
        location,
        interpolation: Some(naga::Interpolation::Perspective),
        sampling: None,
        blend_src: None,
        per_primitive: false,
    }
}

fn append_vertex_output_type(
    module: &mut naga::Module,
    f32_type: naga::Handle<naga::Type>,
    vec4f: naga::Handle<naga::Type>,
    varying_count: usize,
) -> naga::Handle<naga::Type> {
    let mut members = vec![naga::StructMember {
        name: Some("position".into()),
        ty: vec4f,
        binding: Some(naga::Binding::BuiltIn(naga::BuiltIn::Position { invariant: false })),
        offset: 0,
    }];
    members.extend((0..varying_count).map(|index| naga::StructMember {
        name: Some(format!("v{index}")),
        ty: f32_type,
        binding: Some(location_binding(index as u32)),
        offset: 16 + index as u32 * 4,
    }));
    module.types.insert(
        naga::Type {
            name: Some("RasterVertexOutput".into()),
            inner: naga::TypeInner::Struct {
                members,
                span: 16 + varying_count as u32 * 4,
            },
        },
        naga::Span::UNDEFINED,
    )
}

fn load_state(
    function: &sonatina_ir::Function,
    first_arg: usize,
    state_fields: &[(usize, Type)],
    state_var: Option<naga::Handle<naga::GlobalVariable>>,
    naga_func: &mut naga::Function,
    values: &mut HashMap<sonatina_ir::ValueId, naga::Handle<naga::Expression>>,
) {
    let Some(state_var) = state_var else { return };
    let state = naga_func.expressions.append(
        naga::Expression::GlobalVariable(state_var), naga::Span::UNDEFINED,
    );
    for (member, (state_index, _)) in state_fields.iter().enumerate() {
        let arg = function.arg_values[first_arg + *state_index];
        let pointer = naga_func.expressions.append(
            naga::Expression::AccessIndex { base: state, index: member as u32 },
            naga::Span::UNDEFINED,
        );
        let loaded = naga_func.expressions.append(
            naga::Expression::Load { pointer }, naga::Span::UNDEFINED,
        );
        naga_func.body.push(
            naga::Statement::Emit(naga::Range::new_from_bounds(pointer, loaded)),
            naga::Span::UNDEFINED,
        );
        values.insert(arg, loaded);
    }
}

/// Return the logical leaves of a stage when its CFG has one return terminator.
fn unique_source_return_arguments(
    function: &sonatina_ir::Function,
) -> Result<Option<Vec<sonatina_ir::ValueId>>, String> {
    use sonatina_ir::InstDowncast;

    let mut args = None;
    for block in function.layout.iter_block() {
        for inst in function.layout.iter_inst(block) {
            if let Some(ret) = <&sonatina_ir::inst::control_flow::Return as InstDowncast>::downcast(
                function.inst_set(),
                function.dfg.inst(inst),
            ) {
                if args.is_some() {
                    return Ok(None);
                }
                args = Some(ret.args().as_slice().to_vec());
            }
        }
    }
    args.map(Some)
        .ok_or_else(|| "spirv raster: stage has no return terminator".to_string())
}

/// Give authored raster stages one ordinary SSA exit before structurization.
///
/// The generic structured-control-flow emitter supports returns nested inside
/// arbitrary regions by threading mutable return state through every enclosing
/// construct. That is useful for general helpers, but a large inlined shader
/// with many source-level early returns can otherwise duplicate substantial
/// continuations while the structured tree is formed. Raster stages have a
/// fixed result signature, so a compiler-private clone can express the same
/// semantics more directly: every old return jumps to one exit block and one
/// phi per result selects its value. Source IR is never mutated.
fn normalize_stage_to_single_exit(function: &Function) -> Result<Option<Function>, String> {
    use sonatina_ir::inst::control_flow::{Phi, Return};

    let mut returns = Vec::new();
    for block in function.layout.iter_block() {
        for inst in function.layout.iter_inst(block) {
            if let Some(ret) = <&Return as InstDowncast>::downcast(
                function.inst_set(),
                function.dfg.inst(inst),
            ) {
                returns.push((block, inst, ret.args().as_slice().to_vec()));
            }
        }
    }
    match returns.len() {
        0 => return Err("spirv raster: stage has no return terminator".to_string()),
        1 => return Ok(None),
        _ => {}
    }

    let result_types = returns[0]
        .2
        .iter()
        .map(|value| function.dfg.value_ty(*value))
        .collect::<Vec<_>>();
    for (_, _, arguments) in &returns[1..] {
        if arguments.len() != result_types.len() {
            return Err(format!(
                "spirv raster: stage return arity differs: expected {}, got {}",
                result_types.len(),
                arguments.len(),
            ));
        }
        for (index, (&argument, &expected)) in
            arguments.iter().zip(&result_types).enumerate()
        {
            let actual = function.dfg.value_ty(argument);
            if actual != expected {
                return Err(format!(
                    "spirv raster: stage return type differs at leaf {index}: expected {expected:?}, got {actual:?}",
                ));
            }
        }
    }

    let mut normalized = function.clone();
    let exit = normalized.dfg.make_block();
    normalized.layout.append_block(exit);
    for (_, inst, _) in &returns {
        let jump = normalized.dfg.make_jump(exit);
        normalized.dfg.replace_inst(*inst, Box::new(jump));
    }

    let mut exit_values = smallvec::SmallVec::<[sonatina_ir::ValueId; 2]>::new();
    for (result_index, result_type) in result_types.into_iter().enumerate() {
        let arguments = returns
            .iter()
            .map(|(block, _, values)| (values[result_index], *block))
            .collect();
        let phi = Phi::new(normalized.inst_set(), arguments);
        let phi_inst = normalized.dfg.make_inst(phi);
        normalized.layout.append_inst(phi_inst, exit);
        let phi_value = normalized.dfg.make_value(Value::Inst {
            inst: phi_inst,
            result_idx: 0,
            ty: result_type,
        });
        normalized.dfg.attach_result(phi_inst, phi_value);
        exit_values.push(phi_value);
    }
    let ret = Return::new(normalized.inst_set(), exit_values.into());
    let ret_inst = normalized.dfg.make_inst(ret);
    normalized.layout.append_inst(ret_inst, exit);
    normalized.rebuild_users();

    Ok(Some(normalized))
}

fn resolve_source_return_values(
    function: &sonatina_ir::Function,
    arguments: &[sonatina_ir::ValueId],
    values: &mut HashMap<sonatina_ir::ValueId, naga::Handle<naga::Expression>>,
    phis: &HashMap<sonatina_ir::ValueId, naga::Handle<naga::LocalVariable>>,
    naga_func: &mut naga::Function,
) -> Result<Vec<naga::Handle<naga::Expression>>, String> {
    arguments
        .iter()
        .copied()
        .map(|value| {
            resolve_naga_value(value, function, WordKind::U32, values, phis, naga_func)
                .ok_or_else(|| format!("spirv raster: returned value {value:?} is unresolved"))
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn lower_vertex(
    module: &Module,
    func_ref: sonatina_ir::module::FuncRef,
    plan: &RasterStagePlan,
    pipeline: &SpirvRasterPipeline,
    state_var: Option<naga::Handle<naga::GlobalVariable>>,
    state_fields: &[(usize, Type)],
    external_roots: &[(u32, naga::Handle<naga::GlobalVariable>)],
    builtin_arguments: &[SpirvBuiltinArgument],
    u32_type: naga::Handle<naga::Type>,
    f32_type: naga::Handle<naga::Type>,
    bool_type: naga::Handle<naga::Type>,
    vec4f: naga::Handle<naga::Type>,
    output_type: naga::Handle<naga::Type>,
    naga_functions: &NagaFunctionMap,
) -> Result<naga::Function, String> {
    let arguments = builtin_arguments
        .iter()
        .map(|argument| {
            let (name, builtin) = match argument.source {
                SpirvBuiltinSource::VertexIndex => {
                    ("vertex_index", naga::BuiltIn::VertexIndex)
                }
                SpirvBuiltinSource::InstanceIndex => {
                    ("instance_index", naga::BuiltIn::InstanceIndex)
                }
                _ => unreachable!("authored-raster builtin context was validated above"),
            };
            naga::FunctionArgument {
                name: Some(name.into()),
                ty: u32_type,
                binding: Some(naga::Binding::BuiltIn(builtin)),
            }
        })
        .collect();
    let mut output = naga::Function {
        name: Some(pipeline.vertex_entry.clone()),
        arguments,
        result: Some(naga::FunctionResult { ty: output_type, binding: None }),
        local_variables: naga::Arena::new(),
        expressions: naga::Arena::new(),
        named_expressions: Default::default(),
        body: naga::Block::new(),
        diagnostic_filter_leaf: None,
    };
    let error = module.func_store.try_view(func_ref, |source_function| -> Result<(), String> {
        let function = plan.normalized.as_ref().unwrap_or(source_function);
        let mut values = HashMap::new();
        let mut phis = HashMap::new();
        for (physical_index, argument) in builtin_arguments.iter().enumerate() {
            let value = output.expressions.append(
                naga::Expression::FunctionArgument(physical_index as u32),
                naga::Span::UNDEFINED,
            );
            values.insert(function.arg_values[argument.arg_index as usize], value);
        }
        for &(arg_index, global) in external_roots {
            let arg = function.arg_values[arg_index as usize];
            let root = output.expressions.append(
                naga::Expression::GlobalVariable(global),
                naga::Span::UNDEFINED,
            );
            values.insert(arg, root);
        }
        load_state(
            function,
            builtin_arguments.len(),
            state_fields,
            state_var,
            &mut output,
            &mut values,
        );
        let mut result = None;
        emit_naga_regions(
            function, function.inst_set(), WordKind::U32, &plan.structured.regions,
            u32_type, f32_type, bool_type, &mut output, &mut values, &mut phis,
            &mut result,
            None,
            naga_functions,
            None,
        )?;
        let leaves = resolve_source_return_values(
            function,
            &plan.return_arguments,
            &mut values,
            &phis,
            &mut output,
        )?;
        let position = output.expressions.append(
            naga::Expression::Compose { ty: vec4f, components: leaves[..4].to_vec() },
            naga::Span::UNDEFINED,
        );
        output.body.push(
            naga::Statement::Emit(naga::Range::new_from_bounds(position, position)),
            naga::Span::UNDEFINED,
        );
        let mut components = vec![position];
        components.extend_from_slice(&leaves[4..]);
        let result = output.expressions.append(
            naga::Expression::Compose { ty: output_type, components },
            naga::Span::UNDEFINED,
        );
        output.body.push(
            naga::Statement::Emit(naga::Range::new_from_bounds(result, result)),
            naga::Span::UNDEFINED,
        );
        output.body.push(naga::Statement::Return { value: Some(result) }, naga::Span::UNDEFINED);
        Ok(())
    }).ok_or_else(|| "spirv raster: vertex body is unavailable".to_string())?;
    error?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn lower_fragment(
    module: &Module,
    func_ref: sonatina_ir::module::FuncRef,
    plan: &RasterStagePlan,
    pipeline: &SpirvRasterPipeline,
    state_var: Option<naga::Handle<naga::GlobalVariable>>,
    state_fields: &[(usize, Type)],
    external_roots: &[(u32, naga::Handle<naga::GlobalVariable>)],
    varying_count: usize,
    vertex_context_count: usize,
    u32_type: naga::Handle<naga::Type>,
    f32_type: naga::Handle<naga::Type>,
    bool_type: naga::Handle<naga::Type>,
    vec4f: naga::Handle<naga::Type>,
    naga_functions: &NagaFunctionMap,
) -> Result<naga::Function, String> {
    let mut output = naga::Function {
        name: Some(pipeline.fragment_entry.clone()),
        arguments: (0..varying_count).map(|index| naga::FunctionArgument {
            name: Some(format!("v{index}")),
            ty: f32_type,
            binding: Some(location_binding(index as u32)),
        }).collect(),
        result: Some(naga::FunctionResult { ty: vec4f, binding: Some(location_binding(0)) }),
        local_variables: naga::Arena::new(),
        expressions: naga::Arena::new(),
        named_expressions: Default::default(),
        body: naga::Block::new(),
        diagnostic_filter_leaf: None,
    };
    let error = module.func_store.try_view(func_ref, |source_function| -> Result<(), String> {
        let function = plan.normalized.as_ref().unwrap_or(source_function);
        let mut values = HashMap::new();
        let mut phis = HashMap::new();
        for (index, arg) in function.arg_values.iter().copied().take(varying_count).enumerate() {
            let value = output.expressions.append(
                naga::Expression::FunctionArgument(index as u32), naga::Span::UNDEFINED,
            );
            values.insert(arg, value);
        }
        for &(vertex_arg_index, global) in external_roots {
            let state_index = vertex_arg_index as usize - vertex_context_count;
            let fragment_arg_index = varying_count + state_index;
            let arg = function.arg_values[fragment_arg_index];
            let root = output.expressions.append(
                naga::Expression::GlobalVariable(global),
                naga::Span::UNDEFINED,
            );
            values.insert(arg, root);
        }
        load_state(function, varying_count, state_fields, state_var, &mut output, &mut values);
        let mut result = None;
        emit_naga_regions(
            function, function.inst_set(), WordKind::U32, &plan.structured.regions,
            u32_type, f32_type, bool_type, &mut output, &mut values, &mut phis,
            &mut result,
            None,
            naga_functions,
            None,
        )?;
        let packed = resolve_source_return_values(
            function,
            &plan.return_arguments,
            &mut values,
            &phis,
            &mut output,
        )?[0];
        let color = output.expressions.append(
            naga::Expression::Math {
                fun: naga::MathFunction::Unpack4x8unorm,
                arg: packed,
                arg1: None,
                arg2: None,
                arg3: None,
            },
            naga::Span::UNDEFINED,
        );
        output.body.push(
            naga::Statement::Emit(naga::Range::new_from_bounds(color, color)),
            naga::Span::UNDEFINED,
        );
        output.body.push(naga::Statement::Return { value: Some(color) }, naga::Span::UNDEFINED);
        Ok(())
    }).ok_or_else(|| "spirv raster: fragment body is unavailable".to_string())?;
    error?;
    Ok(output)
}

fn entry_point(name: String, stage: naga::ShaderStage, function: naga::Function) -> naga::EntryPoint {
    naga::EntryPoint {
        name,
        stage,
        early_depth_test: None,
        workgroup_size: [0, 0, 0],
        workgroup_size_overrides: None,
        function,
        mesh_info: None,
        task_payload: None,
        incoming_ray_payload: None,
    }
}
