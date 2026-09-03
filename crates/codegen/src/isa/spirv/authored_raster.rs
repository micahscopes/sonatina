use std::collections::HashMap;

use sonatina_ir::{InstDowncast, Module, Type};

use crate::optim::dead_arg::analyze_live_arguments;

use super::{
    Access, LayoutMode, Role, SpirvBinding, SpirvBindingMember, SpirvBuiltinArgument,
    SpirvBuiltinInput, SpirvBuiltinSource, SpirvExternalResource, SpirvLayout, SpirvRasterPipeline,
    SpirvScalarKind, SpirvShaderStage, WordKind, append_external_resources, emit_naga_regions,
    resolve_naga_value, spirv_instruction_is_lowered, unsupported_signed_op_under_u32,
};

/// Lower the narrow first authored-raster ABI. Keeping it separate from the
/// established fullscreen path makes paired multi-value lowering opt-in and
/// leaves existing shader byte streams alone.
pub(super) fn translate(
    module: &Module,
    pipeline: &SpirvRasterPipeline,
    external_resources: &[SpirvExternalResource],
    builtin_arguments: &[SpirvBuiltinArgument],
) -> Result<(naga::Module, SpirvLayout), String> {
    if pipeline.vertex_entry == pipeline.fragment_entry {
        return Err("spirv raster: vertex and fragment entries must be distinct".to_string());
    }

    let find_entry = |name: &str| {
        module.funcs().iter().copied().find(|func| {
            module.ctx.get_sig(*func).is_some_and(|sig| sig.name() == name)
        })
    };
    let vertex_ref = find_entry(&pipeline.vertex_entry).ok_or_else(|| {
        format!("spirv raster: vertex entry `{}` is absent", pipeline.vertex_entry)
    })?;
    let fragment_ref = find_entry(&pipeline.fragment_entry).ok_or_else(|| {
        format!("spirv raster: fragment entry `{}` is absent", pipeline.fragment_entry)
    })?;
    let vertex_sig = module.ctx.get_sig(vertex_ref)
        .ok_or_else(|| "spirv raster: vertex entry has no signature".to_string())?;
    let fragment_sig = module.ctx.get_sig(fragment_ref)
        .ok_or_else(|| "spirv raster: fragment entry has no signature".to_string())?;

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
        if resource.access != Access::Read {
            return Err(format!(
                "spirv raster: external resource {} must be read-only",
                resource.name,
            ));
        }
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
    let scalar_state = declared_scalar_state
        .into_iter()
        .filter(|(state_position, _)| {
            is_live(vertex_live, context_count + *state_position)
                || is_live(fragment_live, varying_count + *state_position)
        })
        .collect::<Vec<_>>();
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

    // Calls have already been inlined by Fe. Object indexing/projection/load
    // resolves through the shared external globals. Allocation, private memory
    // and traps still have no stage-paired channel.
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

    let vertex = lower_vertex(
        module, vertex_ref, pipeline, state_var, &scalar_state, &external_roots, varying_count,
        builtin_arguments, u32_type, f32_type, bool_type, vec4f, output_type,
    )?;
    let fragment = lower_fragment(
        module, fragment_ref, pipeline, state_var, &scalar_state, &external_roots, varying_count,
        context_count, u32_type, f32_type, bool_type, vec4f,
    )?;

    naga_mod.entry_points.push(entry_point(
        pipeline.vertex_entry.clone(), naga::ShaderStage::Vertex, vertex,
    ));
    naga_mod.entry_points.push(entry_point(
        pipeline.fragment_entry.clone(), naga::ShaderStage::Fragment, fragment,
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
        entry_point: pipeline.fragment_entry.clone(),
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
        vertex_entry: Some(pipeline.vertex_entry.clone()),
        fragment_entry: Some(pipeline.fragment_entry.clone()),
        color_target_format: Some("rgba8unorm".to_string()),
    }))
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

fn source_return_values(
    function: &sonatina_ir::Function,
    values: &HashMap<sonatina_ir::ValueId, naga::Handle<naga::Expression>>,
    phis: &HashMap<sonatina_ir::ValueId, naga::Handle<naga::LocalVariable>>,
    naga_func: &mut naga::Function,
) -> Result<Vec<naga::Handle<naga::Expression>>, String> {
    let mut args = None;
    for block in function.layout.iter_block() {
        for inst in function.layout.iter_inst(block) {
            if let Some(ret) = <&sonatina_ir::inst::control_flow::Return as InstDowncast>::downcast(
                function.inst_set(), function.dfg.inst(inst),
            ) {
                if args.is_some() {
                    return Err("spirv raster: multiple return terminators need multi-value return transport; fail closed".to_string());
                }
                args = Some(ret.args().as_slice().to_vec());
            }
        }
    }
    args.ok_or_else(|| "spirv raster: stage has no return terminator".to_string())?
        .into_iter()
        .map(|value| resolve_naga_value(value, function, WordKind::U32, values, phis, naga_func)
            .ok_or_else(|| format!("spirv raster: returned value {value:?} is unresolved")))
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn lower_vertex(
    module: &Module,
    func_ref: sonatina_ir::module::FuncRef,
    pipeline: &SpirvRasterPipeline,
    state_var: Option<naga::Handle<naga::GlobalVariable>>,
    state_fields: &[(usize, Type)],
    external_roots: &[(u32, naga::Handle<naga::GlobalVariable>)],
    varying_count: usize,
    builtin_arguments: &[SpirvBuiltinArgument],
    u32_type: naga::Handle<naga::Type>,
    f32_type: naga::Handle<naga::Type>,
    bool_type: naga::Handle<naga::Type>,
    vec4f: naga::Handle<naga::Type>,
    output_type: naga::Handle<naga::Type>,
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
    let error = module.func_store.try_view(func_ref, |function| -> Result<(), String> {
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
        let scfg = crate::structurize::structurize_function(function)?;
        let mut ignored = None;
        let naga_functions = super::NagaFunctionMap::new();
        emit_naga_regions(
            function, function.inst_set(), WordKind::U32, &scfg.regions,
            u32_type, f32_type, bool_type, &mut output, &mut values, &mut phis,
            &mut ignored, None, &naga_functions, None,
        )?;
        let leaves = source_return_values(function, &values, &phis, &mut output)?;
        if leaves.len() != 4 + varying_count {
            return Err(format!(
                "spirv raster: vertex returned {} leaves; expected {}",
                leaves.len(), 4 + varying_count,
            ));
        }
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
    let error = module.func_store.try_view(func_ref, |function| -> Result<(), String> {
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
        let scfg = crate::structurize::structurize_function(function)?;
        let mut result = None;
        let naga_functions = super::NagaFunctionMap::new();
        emit_naga_regions(
            function, function.inst_set(), WordKind::U32, &scfg.regions,
            u32_type, f32_type, bool_type, &mut output, &mut values, &mut phis,
            &mut result, None, &naga_functions, None,
        )?;
        let packed = result.ok_or_else(|| "spirv raster: fragment produced no result".to_string())?;
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
