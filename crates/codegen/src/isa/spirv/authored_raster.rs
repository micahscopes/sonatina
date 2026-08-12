use std::collections::HashMap;

use sonatina_ir::{InstDowncast, Module, Type};

use super::{
    Access, LayoutMode, Role, SpirvBinding, SpirvBindingMember, SpirvBuiltinInput,
    SpirvBuiltinSource, SpirvExternalResource, SpirvLayout, SpirvRasterPipeline,
    SpirvScalarKind, WordKind, emit_naga_regions, resolve_naga_value,
    spirv_instruction_is_lowered, unsupported_signed_op_under_u32,
};

/// Lower the narrow first authored-raster ABI. Keeping it separate from the
/// established fullscreen path makes paired multi-value lowering opt-in and
/// leaves existing shader byte streams alone.
pub(super) fn translate(
    module: &Module,
    pipeline: &SpirvRasterPipeline,
    external_resources: &[SpirvExternalResource],
) -> Result<(naga::Module, SpirvLayout), String> {
    if !external_resources.is_empty() {
        return Err("spirv raster: authored resources are not wired into both stages yet; fail closed".to_string());
    }
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

    if vertex_sig.args().first() != Some(&Type::I32) {
        return Err("spirv raster: vertex arg 0 must be the u32/i32 vertex-index carrier".to_string());
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
    let vertex_state = &vertex_sig.args()[1..];
    let fragment_state = &fragment_sig.args()[varying_count..];
    if vertex_state != fragment_state {
        return Err(format!(
            "spirv raster: vertex and fragment actor-state suffixes differ: {:?} versus {:?}",
            vertex_state, fragment_state,
        ));
    }
    if vertex_state.iter().any(|ty| !matches!(ty, Type::I32 | Type::F32)) {
        return Err("spirv raster: actor state admits only i32/u32 and f32 leaves".to_string());
    }

    // Calls have already been inlined by Fe. Object/memory/trap operations do
    // not yet have a stage-paired transport, so refuse them explicitly.
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
                        || <&sonatina_ir::inst::data::ObjLoad as InstDowncast>::downcast(inst_set, data).is_some()
                        || <&sonatina_ir::inst::data::ObjStore as InstDowncast>::downcast(inst_set, data).is_some()
                        || <&sonatina_ir::inst::data::ObjIndex as InstDowncast>::downcast(inst_set, data).is_some()
                        || <&sonatina_ir::inst::data::ObjProj as InstDowncast>::downcast(inst_set, data).is_some()
                        || <&sonatina_ir::inst::data::MemAllocDynamic as InstDowncast>::downcast(inst_set, data).is_some()
                        || <&sonatina_ir::inst::data::Mload as InstDowncast>::downcast(inst_set, data).is_some()
                        || <&sonatina_ir::inst::data::Mstore as InstDowncast>::downcast(inst_set, data).is_some()
                        || <&sonatina_ir::inst::control_flow::Unreachable as InstDowncast>::downcast(inst_set, data).is_some();
                    if has_untransported_effect {
                        return Err(format!(
                            "spirv raster: {stage} body uses object/memory/trap operations that have no raster-stage channel; fail closed",
                        ));
                    }
                    if let Some(op) = unsupported_signed_op_under_u32(inst_set, data) {
                        return Err(format!(
                            "spirv raster: signedness-sensitive `{op}` is unsupported under the u32 browser carrier",
                        ));
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

    let (state_var, state_span, layout_members) = append_state_binding(
        &mut naga_mod, vertex_state, u32_type, f32_type, varying_count,
    );
    let output_type = append_vertex_output_type(&mut naga_mod, f32_type, vec4f, varying_count);

    let vertex = lower_vertex(
        module, vertex_ref, pipeline, state_var, varying_count,
        u32_type, f32_type, bool_type, vec4f, output_type,
    )?;
    let fragment = lower_fragment(
        module, fragment_ref, pipeline, state_var, varying_count,
        u32_type, f32_type, bool_type, vec4f,
    )?;

    naga_mod.entry_points.push(entry_point(
        pipeline.vertex_entry.clone(), naga::ShaderStage::Vertex, vertex,
    ));
    naga_mod.entry_points.push(entry_point(
        pipeline.fragment_entry.clone(), naga::ShaderStage::Fragment, fragment,
    ));

    let bindings = state_var.map(|_| vec![SpirvBinding {
        group: 0,
        binding: 0,
        name: "state".to_string(),
        access: Access::Read,
        role: Role::Input,
        stride: state_span,
        span: state_span,
        members: layout_members,
        resource_element: None,
        resource_length: None,
        resource_arg_index: None,
    }]).unwrap_or_default();

    Ok((naga_mod, SpirvLayout {
        entry_point: pipeline.fragment_entry.clone(),
        mode: LayoutMode::Render,
        workgroup_size: [0, 0, 0],
        word: WordKind::U32,
        bindings,
        builtin_inputs: vec![SpirvBuiltinInput {
            arg_index: 0,
            source: SpirvBuiltinSource::VertexIndex,
            scalar: SpirvScalarKind::U32,
        }],
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
    state: &[Type],
    u32_type: naga::Handle<naga::Type>,
    f32_type: naga::Handle<naga::Type>,
    fragment_prefix: usize,
) -> (Option<naga::Handle<naga::GlobalVariable>>, u32, Vec<SpirvBindingMember>) {
    if state.is_empty() {
        return (None, 0, Vec::new());
    }
    let mut layout = Vec::with_capacity(state.len());
    let members = state.iter().enumerate().map(|(index, ty)| {
        let (naga_ty, scalar) = match ty {
            Type::I32 => (u32_type, SpirvScalarKind::I32),
            Type::F32 => (f32_type, SpirvScalarKind::F32),
            _ => unreachable!("state admission checked by translate"),
        };
        let offset = index as u32 * 4;
        layout.push(SpirvBindingMember {
            arg_index: (fragment_prefix + index) as u32,
            offset,
            width: 4,
            scalar,
        });
        naga::StructMember {
            name: Some(format!("p{}", fragment_prefix + index)),
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
            binding: Some(naga::ResourceBinding { group: 0, binding: 0 }),
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
        interpolation: None,
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
    state_var: Option<naga::Handle<naga::GlobalVariable>>,
    naga_func: &mut naga::Function,
    values: &mut HashMap<sonatina_ir::ValueId, naga::Handle<naga::Expression>>,
) {
    let Some(state_var) = state_var else { return };
    let state = naga_func.expressions.append(
        naga::Expression::GlobalVariable(state_var), naga::Span::UNDEFINED,
    );
    for (member, arg) in function.arg_values.iter().copied().skip(first_arg).enumerate() {
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
    varying_count: usize,
    u32_type: naga::Handle<naga::Type>,
    f32_type: naga::Handle<naga::Type>,
    bool_type: naga::Handle<naga::Type>,
    vec4f: naga::Handle<naga::Type>,
    output_type: naga::Handle<naga::Type>,
) -> Result<naga::Function, String> {
    let mut output = naga::Function {
        name: Some(pipeline.vertex_entry.clone()),
        arguments: vec![naga::FunctionArgument {
            name: Some("vertex_index".into()),
            ty: u32_type,
            binding: Some(naga::Binding::BuiltIn(naga::BuiltIn::VertexIndex)),
        }],
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
        let index = output.expressions.append(
            naga::Expression::FunctionArgument(0), naga::Span::UNDEFINED,
        );
        values.insert(function.arg_values[0], index);
        load_state(function, 1, state_var, &mut output, &mut values);
        let scfg = crate::structurize::structurize_function(function)?;
        let mut ignored = None;
        emit_naga_regions(
            function, function.inst_set(), WordKind::U32, &scfg.regions,
            u32_type, f32_type, bool_type, &mut output, &mut values, &mut phis,
            &mut ignored, None,
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
    varying_count: usize,
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
        load_state(function, varying_count, state_var, &mut output, &mut values);
        let scfg = crate::structurize::structurize_function(function)?;
        let mut result = None;
        emit_naga_regions(
            function, function.inst_set(), WordKind::U32, &scfg.regions,
            u32_type, f32_type, bool_type, &mut output, &mut values, &mut phis,
            &mut result, None,
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
