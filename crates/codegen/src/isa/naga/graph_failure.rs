//! Explicit compute graph epochs. Invocation poison results retain their ABI;
//! a separate atomic status prevents later dispatches from consuming them.

use super::{GraphFailureBinding, MemCtx, emit_expr, lit_u32};

fn violates_admission(block: &naga::Block, entry: bool) -> bool {
    block.iter().any(|statement| match statement {
        naga::Statement::ControlBarrier(_) | naga::Statement::MemoryBarrier(_)
        | naga::Statement::WorkGroupUniformLoad { .. } => true,
        naga::Statement::Return { .. } => entry,
        naga::Statement::Block(body) => violates_admission(body, entry),
        naga::Statement::If { accept, reject, .. } => violates_admission(accept, entry) || violates_admission(reject, entry),
        naga::Statement::Loop { body, continuing, .. } => violates_admission(body, entry) || violates_admission(continuing, entry),
        naga::Statement::Switch { cases, .. } => cases.iter().any(|case| violates_admission(&case.body, entry)),
        _ => false,
    })
}

pub(super) fn wrap_compute_entry(
    module: &mut naga::Module,
    function: &mut naga::Function,
    mem: Option<MemCtx>,
    binding: GraphFailureBinding,
) -> Result<(), String> {
    // A failure in another invocation can make entry admission nonuniform.
    // Do not admit workgroup cooperation under that condition. A future
    // workgroup-uniform admission protocol needs its own capability/gates.
    if violates_admission(&function.body, true) || module.functions.iter().any(|(_, f)| violates_admission(&f.body, false)) {
        return Err("graph failure entry guards require barrier-free code and a single entry epilogue".into());
    }
    if module.global_variables.iter().any(|(_, global)| {
        global.binding.as_ref().is_some_and(|b| b.group == binding.group && b.binding == binding.binding)
    }) {
        return Err("graph failure binding collides with an emitted shader binding".into());
    }
    let ty = module.types.insert(naga::Type {
        name: None, inner: naga::TypeInner::Atomic(naga::Scalar::U32),
    }, naga::Span::UNDEFINED);
    let global = module.global_variables.append(naga::GlobalVariable {
        name: Some("graph_failure".into()),
        space: naga::AddressSpace::Storage { access: naga::StorageAccess::LOAD | naga::StorageAccess::STORE },
        binding: Some(naga::ResourceBinding { group: binding.group, binding: binding.binding }),
        ty, init: None, memory_decorations: naga::ir::MemoryDecorations::empty(),
    }, naga::Span::UNDEFINED);
    let pointer = function.expressions.append(naga::Expression::GlobalVariable(global), naga::Span::UNDEFINED);
    let mut body = std::mem::take(&mut function.body);
    if let Some(mem) = mem {
        let failed = emit_expr(function, &mut body, naga::Expression::Load { pointer: mem.trapped });
        let value = emit_expr(function, &mut body, naga::Expression::As {
            expr: failed, kind: naga::ScalarKind::Uint, convert: Some(4),
        });
        // OR, not Store: successful invocations and later dispatches must not
        // erase a failure. Reset belongs to the owner of the next graph epoch.
        body.push(naga::Statement::Atomic {
            pointer, fun: naga::AtomicFunction::InclusiveOr, value, result: None,
        }, naga::Span::UNDEFINED);
    }
    let mut admission = naga::Block::new();
    let status = emit_expr(function, &mut admission, naga::Expression::Load { pointer });
    let zero = lit_u32(function, 0);
    let ready = emit_expr(function, &mut admission, naga::Expression::Binary {
        op: naga::BinaryOperator::Equal, left: status, right: zero,
    });
    admission.push(naga::Statement::If {
        condition: ready, accept: body, reject: naga::Block::new(),
    }, naga::Span::UNDEFINED);
    function.body = admission;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::naga::{Access, NagaBackend, ShaderCompileRequest, ShaderEncoding, ShaderEnvironment,
        ShaderPipeline, ShaderTargetContract, SpirvExternalResource, SpirvResourceElement, SpirvScalarKind};

    #[test]
    fn graph_failure_admission_rejects_barriers_and_bypassed_epilogues() {
        for statement in [
            naga::Statement::ControlBarrier(naga::Barrier::WORK_GROUP),
            naga::Statement::Return { value: None },
        ] {
            let mut module = naga::Module::default();
            let mut function = naga::Function::default();
            let mut nested = naga::Block::new();
            nested.push(statement, naga::Span::UNDEFINED);
            function.body.push(naga::Statement::Block(nested), naga::Span::UNDEFINED);
            assert!(wrap_compute_entry(&mut module, &mut function, None,
                GraphFailureBinding { group: 0, binding: 0 }).is_err());
            assert!(module.global_variables.is_empty(), "rejection must precede mutation");
        }
    }

    #[test]
    fn graph_failure_contract_compiles_and_rejects_binding_collision() {
        let source = r#"
target = "shader-unknown-unknown"
func public %entry(v0.objref<[i32; 4]>) {
    block0:
        v1.objref<i32> = obj.index v0 0.i32;
        v2.i32 = obj.load v1;
        v3.i1 = eq v2 0.i32;
        br v3 block1 block2;
    block1:
        unreachable;
    block2:
        v4.objref<i32> = obj.index v0 3.i32;
        obj.store v4 33.i32;
        return;
}
"#;
        let module = sonatina_parser::parse_module(source).unwrap().module;
        let entry = module.funcs()[0];
        let target = ShaderTargetContract::new(ShaderEnvironment::WebGpu,
            [ShaderEncoding::Wgsl, ShaderEncoding::Spirv]).unwrap();
        let resources = [SpirvExternalResource {
            arg_index: 0, group: 0, binding: 0, name: "data".into(), access: Access::ReadWrite,
            element: SpirvResourceElement::Scalar(SpirvScalarKind::U32), stride: 4, length: 4,
        }];
        let mut request = ShaderCompileRequest::new(&target, ShaderPipeline::Compute {
            entry, workgroup_size: [1, 1, 1], dispatch_grid: [1, 1, 1],
        });
        request.resources = &resources;
        let plain = NagaBackend::compile_request(&module, &request).unwrap();
        assert_eq!(plain.layout.graph_failure, None);
        let binding = GraphFailureBinding { group: 0, binding: 2 };
        request.graph_failure = Some(binding);
        let artifact = NagaBackend::compile_request(&module, &request).unwrap();
        assert_eq!(artifact.layout.graph_failure, Some(binding));
        let wgsl = artifact.wgsl.unwrap();
        assert!(wgsl.contains("atomicOr"));
        assert!(wgsl.contains("atomicLoad"));
        if let Some(path) = std::env::var_os("SONATINA_GRAPH_FAILURE_WGSL") {
            std::fs::write(path, &wgsl).unwrap();
        }
        request.graph_failure = Some(GraphFailureBinding { group: 0, binding: 0 });
        assert!(NagaBackend::compile_request(&module, &request).err().unwrap().iter()
            .any(|error| error.to_string().contains("collides")));
        request.graph_failure = None;
        let unchanged = NagaBackend::compile_request(&module, &request).unwrap();
        assert_eq!(unchanged.wgsl, plain.wgsl);
        assert_eq!(unchanged.words, plain.words);
    }

    #[test]
    fn graph_failure_mixed_invocations_keep_private_trap_lanes() {
        let source = r#"
target = "shader-unknown-unknown"
func public %entry(v0.objref<[i32; 4]>, v1.i32) {
    block0:
        v2.objref<i32> = obj.index v0 v1;
        v3.i32 = obj.load v2;
        v4.i1 = eq v3 0.i32;
        br v4 block1 block2;
    block1:
        unreachable;
    block2:
        v5.i32 = add v1 2.i32;
        v6.objref<i32> = obj.index v0 v5;
        obj.store v6 33.i32;
        return;
}
"#;
        let module = sonatina_parser::parse_module(source).unwrap().module;
        let target = ShaderTargetContract::new(ShaderEnvironment::WebGpu,
            [ShaderEncoding::Wgsl, ShaderEncoding::Spirv]).unwrap();
        let resources = [SpirvExternalResource {
            arg_index: 0, group: 0, binding: 0, name: "data".into(), access: Access::ReadWrite,
            element: SpirvResourceElement::Scalar(SpirvScalarKind::U32), stride: 4, length: 4,
        }];
        let builtins = [crate::isa::naga::SpirvBuiltinArgument {
            arg_index: 1, source: crate::isa::naga::SpirvBuiltinSource::GlobalInvocationIdX,
        }];
        let mut request = ShaderCompileRequest::new(&target, ShaderPipeline::Compute {
            entry: module.funcs()[0], workgroup_size: [2, 1, 1], dispatch_grid: [1, 1, 1],
        });
        request.resources = &resources;
        request.builtin_arguments = &builtins;
        request.graph_failure = Some(GraphFailureBinding { group: 0, binding: 2 });
        let artifact = NagaBackend::compile_request(&module, &request).unwrap();
        assert_eq!(artifact.layout.trap.unwrap().width, 8);
        assert_eq!(artifact.layout.bindings.iter().find(|b| b.binding == 2).unwrap().span, 4);
        if let Some(path) = std::env::var_os("SONATINA_GRAPH_FAILURE_MULTI_WGSL") {
            std::fs::write(path, artifact.wgsl.as_ref().unwrap()).unwrap();
        }
    }
}
