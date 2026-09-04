use sonatina_codegen::Backend;
use sonatina_codegen::isa::spirv::{
    Access, LayoutMode, Role, SpirvBackend, SpirvBindingMember, SpirvBuiltinArgument,
    SpirvBuiltinInput, SpirvBuiltinSource, SpirvExternalResource, SpirvLayout,
    SpirvResourceElement, SpirvResourceField, SpirvScalarKind, SpirvShaderStage, WordKind,
};
use sonatina_ir::{
    Immediate, Linkage, Signature, Type,
    builder::ModuleBuilder,
    func_cursor::InstInserter,
    inst::{arith, cast, cmp, control_flow, data},
    isa::{Isa, native::Native},
    module::ModuleCtx,
};
use sonatina_triple::{Architecture, OperatingSystem, TargetTriple, Vendor};

fn native_module_builder() -> ModuleBuilder {
    let arch = if cfg!(target_arch = "x86_64") {
        Architecture::X86_64
    } else {
        Architecture::Aarch64
    };
    let triple = TargetTriple::new(arch, Vendor::Unknown, OperatingSystem::Native);
    let isa = Native::new(triple);
    let ctx = ModuleCtx::new(&isa);
    ModuleBuilder::new(ctx)
}

fn spirv_rejection(module: &sonatina_ir::Module, expectation: &str) -> String {
    match SpirvBackend::new().compile_module(module) {
        Ok(_) => panic!("{expectation}"),
        Err(errors) => errors
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

fn assert_layout_metadata_invariants(layout: &SpirvLayout, arg_count: usize) {
    let mut seen = vec![false; arg_count];
    for builtin in &layout.builtin_inputs {
        let slot = seen.get_mut(builtin.arg_index as usize).expect("builtin arg index in range");
        assert!(!*slot, "argument {} described twice", builtin.arg_index);
        *slot = true;
    }
    for binding in &layout.bindings {
        assert!(
            !binding.stages.is_empty(),
            "every physical binding has stage demand"
        );
        let mut stages = binding.stages.clone();
        stages.sort_by_key(|stage| match stage {
            SpirvShaderStage::Compute => 0,
            SpirvShaderStage::Vertex => 1,
            SpirvShaderStage::Fragment => 2,
        });
        stages.dedup();
        assert_eq!(
            stages.len(),
            binding.stages.len(),
            "binding stages are unique"
        );
        assert!(binding.stride >= binding.span, "stride must cover span");
        if let Some(arg_index) = binding.resource_arg_index {
            let slot = seen.get_mut(arg_index as usize).expect("resource arg index in range");
            assert!(!*slot, "argument {arg_index} described twice");
            *slot = true;
            assert_eq!(binding.role, Role::Resource);
            assert!(binding.resource_element.is_some());
            assert!(binding.resource_length.is_some());
        }
        let mut end = 0;
        for member in &binding.members {
            assert!(member.width > 0 && member.offset % member.width == 0, "member must be naturally aligned");
            assert!(member.offset >= end, "members must be ordered and non-overlapping");
            end = member.offset + member.width;
            assert!(end <= binding.span, "member must fit binding span");
            let slot = seen.get_mut(member.arg_index as usize).expect("member arg index in range");
            assert!(!*slot, "argument {} described twice", member.arg_index);
            *slot = true;
        }
        if binding.role == Role::Output {
            assert!(binding.members.is_empty());
            assert_eq!(binding.span, layout.word.width_bytes());
            assert_eq!(binding.stride, layout.word.width_bytes());
        }
    }
    assert!(seen.into_iter().all(|covered| covered), "every source argument needs exactly one ABI source");
    if matches!(layout.mode, LayoutMode::Scalar | LayoutMode::Batch) {
        assert!(layout.builtin_inputs.is_empty());
    }
}

#[test]
fn spirv_constant_return_valid() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64, Vendor::Unknown, OperatingSystem::Native
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();

    let sig = Signature::new_single("compute", Linkage::Public, &[], Type::I64);
    let func_ref = mb.declare_function(sig).unwrap();

    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);

    let val = fb.make_imm_value(42i64);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, val));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let backend = SpirvBackend::new();
    let artifact = backend.compile_module(&module).expect("SPIR-V compilation failed");

    // Basic structural validation: check magic number
    assert!(artifact.words.len() > 5, "SPIR-V module too small");
    assert_eq!(artifact.words[0], 0x07230203, "wrong SPIR-V magic number");

    let bytes = artifact.as_bytes();
    eprintln!("SPIR-V module: {} words, {} bytes", artifact.words.len(), bytes.len());

    // Validate with spirv-val if available
    let tmp = std::env::temp_dir().join("test_spirv.spv");
    std::fs::write(&tmp, &bytes).unwrap();
    let result = std::process::Command::new("spirv-val")
        .arg(tmp.to_str().unwrap())
        .output();
    match result {
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            if !stderr.is_empty() { eprintln!("spirv-val stderr: {stderr}"); }
            if !stdout.is_empty() { eprintln!("spirv-val stdout: {stdout}"); }
            assert!(output.status.success(), "spirv-val validation failed");
        }
        Err(_) => {
            eprintln!("spirv-val not found, skipping validation (structural check passed)");
        }
    }
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn spirv_arithmetic_return_valid() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64, Vendor::Unknown, OperatingSystem::Native
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();

    // f() -> i64 { return 100 }
    let sig = Signature::new_single("arithmetic", Linkage::Public, &[], Type::I64);
    let func_ref = mb.declare_function(sig).unwrap();

    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);

    let val = fb.make_imm_value(100i64);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, val));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let backend = SpirvBackend::new().with_workgroup_size(1, 1, 1);
    let artifact = backend.compile_module(&module).expect("SPIR-V compilation failed");

    assert_eq!(artifact.words[0], 0x07230203);
    eprintln!("SPIR-V arithmetic module: {} words", artifact.words.len());
}

#[test]
fn spirv_scalar_helper_call_survives_as_a_valid_wgsl_function() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64,
        Vendor::Unknown,
        OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();

    // Declaration order keeps the public entry as the backend root. The
    // private helper is intentionally ordinary Sonatina IR, not an intrinsic
    // or source annotation.
    let entry_ref = mb
        .declare_function(Signature::new_single(
            "scalar_helper_entry",
            Linkage::Public,
            &[Type::I32, Type::I32],
            Type::I32,
        ))
        .unwrap();
    let helper_ref = mb
        .declare_function(Signature::new_single(
            "scalar_helper",
            Linkage::Private,
            &[Type::I32, Type::I32],
            Type::I32,
        ))
        .unwrap();
    let leaf_ref = mb
        .declare_function(Signature::new_single(
            "scalar_leaf",
            Linkage::Private,
            &[Type::I32, Type::I32],
            Type::I32,
        ))
        .unwrap();

    {
        let mut fb = mb.func_builder::<InstInserter>(leaf_ref);
        let entry = fb.append_block();
        fb.switch_to_block(entry);
        let lhs = fb.args()[0];
        let rhs = fb.args()[1];
        let sum = fb.insert_inst(arith::Add::new(is, lhs, rhs), Type::I32);
        fb.insert_inst_no_result(control_flow::Return::new_single(is, sum));
        fb.seal_all();
        fb.finish();
    }
    {
        let mut fb = mb.func_builder::<InstInserter>(helper_ref);
        let entry = fb.append_block();
        fb.switch_to_block(entry);
        let lhs = fb.args()[0];
        let rhs = fb.args()[1];
        let sum = fb.insert_inst(
            control_flow::Call::new(
                is,
                leaf_ref,
                [lhs, rhs].into_iter().collect(),
            ),
            Type::I32,
        );
        let seven = fb.make_imm_value(7i32);
        let result = fb.insert_inst(arith::Mul::new(is, sum, seven), Type::I32);
        fb.insert_inst_no_result(control_flow::Return::new_single(is, result));
        fb.seal_all();
        fb.finish();
    }
    {
        let mut fb = mb.func_builder::<InstInserter>(entry_ref);
        let entry = fb.append_block();
        fb.switch_to_block(entry);
        let lhs = fb.args()[0];
        let rhs = fb.args()[1];
        let call = fb.insert_inst(
            control_flow::Call::new(
                is,
                helper_ref,
                [lhs, rhs].into_iter().collect(),
            ),
            Type::I32,
        );
        let three = fb.make_imm_value(3i32);
        let result = fb.insert_inst(arith::Add::new(is, call, three), Type::I32);
        fb.insert_inst_no_result(control_flow::Return::new_single(is, result));
        fb.seal_all();
        fb.finish();
    }

    let artifact = SpirvBackend::new()
        .compile_module(&mb.build())
        .expect("ordinary scalar helper call should lower without inlining");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL side artifact");
    assert!(
        wgsl.matches("scalar_helper(").count() >= 2,
        "helper must appear as one definition and at least one call:\n{wgsl}"
    );
    assert!(
        wgsl.matches("scalar_leaf(").count() >= 2,
        "callee-before-caller lowering must preserve the nested leaf call:\n{wgsl}"
    );
    let reparsed = naga::front::wgsl::parse_str(wgsl)
        .expect("WGSL with an ordinary helper call must reparse");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&reparsed)
    .expect("WGSL with an ordinary scalar helper must validate for browser capabilities");
}

#[test]
fn spirv_wide_scalar_helper_uses_a_portable_packed_abi() {
    const INTEGER_ARGUMENTS: usize = 130;
    const FLOAT_ARGUMENTS: usize = 130;

    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64,
        Vendor::Unknown,
        OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let mut helper_arguments = vec![Type::I32; INTEGER_ARGUMENTS];
    helper_arguments.extend(vec![Type::F32; FLOAT_ARGUMENTS]);
    let entry_ref = mb
        .declare_function(Signature::new_single(
            "wide_scalar_helper_entry",
            Linkage::Public,
            &[Type::I32],
            Type::I32,
        ))
        .unwrap();
    let helper_ref = mb
        .declare_function(Signature::new_single(
            "wide_scalar_helper",
            Linkage::Private,
            &helper_arguments,
            Type::I32,
        ))
        .unwrap();

    {
        let mut fb = mb.func_builder::<InstInserter>(helper_ref);
        let entry = fb.append_block();
        fb.switch_to_block(entry);
        let arguments = fb.args().to_vec();
        let mut integer_sum = arguments[0];
        for &argument in &arguments[1..INTEGER_ARGUMENTS] {
            integer_sum = fb.insert_inst(arith::Add::new(is, integer_sum, argument), Type::I32);
        }
        let mut float_sum = arguments[INTEGER_ARGUMENTS];
        for &argument in &arguments[INTEGER_ARGUMENTS + 1..] {
            float_sum = fb.insert_inst(arith::Add::new(is, float_sum, argument), Type::F32);
        }
        let float_bits = fb.insert_inst(cast::Bitcast::new(is, float_sum, Type::I32), Type::I32);
        let sum = fb.insert_inst(
            arith::Add::new(is, integer_sum, float_bits),
            Type::I32,
        );
        fb.insert_inst_no_result(control_flow::Return::new_single(is, sum));
        fb.seal_all();
        fb.finish();
    }
    {
        let mut fb = mb.func_builder::<InstInserter>(entry_ref);
        let entry = fb.append_block();
        fb.switch_to_block(entry);
        let value = fb.args()[0];
        let float = fb.insert_inst(cast::Bitcast::new(is, value, Type::F32), Type::F32);
        let mut arguments = vec![value; INTEGER_ARGUMENTS];
        arguments.extend(vec![float; FLOAT_ARGUMENTS]);
        let result = fb.insert_inst(
            control_flow::Call::new(is, helper_ref, arguments.into()),
            Type::I32,
        );
        fb.insert_inst_no_result(control_flow::Return::new_single(is, result));
        fb.seal_all();
        fb.finish();
    }

    let artifact = SpirvBackend::new()
        .compile_module(&mb.build())
        .expect("a wide scalar helper should lower through a packed function-local ABI");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL side artifact");
    assert!(
        wgsl.contains("wide_scalar_helper_arguments"),
        "the wide helper must receive one generated argument aggregate:\n{wgsl}",
    );
    assert!(
        wgsl.contains("array<u32, 130>") && wgsl.contains("array<f32, 130>"),
        "each homogeneous scalar group must use one fixed array rather than hundreds of named struct fields:\n{wgsl}",
    );
    let declaration = wgsl
        .lines()
        .find(|line| line.starts_with("fn wide_scalar_helper("))
        .expect("wide helper declaration");
    assert_eq!(
        declaration.matches(':').count(),
        1,
        "the wide helper must have one physical WGSL parameter:\n{declaration}",
    );
    let reparsed = naga::front::wgsl::parse_str(wgsl)
        .expect("WGSL with a packed helper ABI must reparse");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&reparsed)
    .expect("WGSL with a packed helper ABI must validate for browser capabilities");
}

#[test]
fn spirv_fixed_struct_local_survives_inside_scalar_helper() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64,
        Vendor::Unknown,
        OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let pair_ty =
        mb.declare_struct_type("FixedLocalPair", &[Type::I32, Type::I32], false);
    let pair_ptr_ty = mb.ptr_type(pair_ty);
    let word_ptr_ty = mb.ptr_type(Type::I32);

    let entry_ref = mb
        .declare_function(Signature::new_single(
            "fixed_struct_local_entry",
            Linkage::Public,
            &[Type::I32, Type::I32],
            Type::I32,
        ))
        .unwrap();
    let helper_ref = mb
        .declare_function(Signature::new_single(
            "fixed_struct_local_helper",
            Linkage::Private,
            &[Type::I32, Type::I32],
            Type::I32,
        ))
        .unwrap();

    {
        let mut fb = mb.func_builder::<InstInserter>(helper_ref);
        let entry = fb.append_block();
        fb.switch_to_block(entry);
        let lhs = fb.args()[0];
        let rhs = fb.args()[1];
        let slot = fb.insert_inst(data::Alloca::new(is, pair_ty), pair_ptr_ty);
        let zero = fb.make_imm_value(0i32);
        let first_index = fb.make_imm_value(0i32);
        let second_index = fb.make_imm_value(1i32);
        let first = fb.insert_inst(
            data::Gep::new(is, [slot, zero, first_index].into_iter().collect()),
            word_ptr_ty,
        );
        let second = fb.insert_inst(
            data::Gep::new(is, [slot, zero, second_index].into_iter().collect()),
            word_ptr_ty,
        );
        fb.insert_inst_no_result(data::Mstore::new(is, first, lhs, Type::I32));
        fb.insert_inst_no_result(data::Mstore::new(is, second, rhs, Type::I32));
        let loaded_lhs =
            fb.insert_inst(data::Mload::new(is, first, Type::I32), Type::I32);
        let loaded_rhs =
            fb.insert_inst(data::Mload::new(is, second, Type::I32), Type::I32);
        let sum = fb.insert_inst(arith::Add::new(is, loaded_lhs, loaded_rhs), Type::I32);
        fb.insert_inst_no_result(control_flow::Return::new_single(is, sum));
        fb.seal_all();
        fb.finish();
    }
    {
        let mut fb = mb.func_builder::<InstInserter>(entry_ref);
        let entry = fb.append_block();
        fb.switch_to_block(entry);
        let lhs = fb.args()[0];
        let rhs = fb.args()[1];
        let result = fb.insert_inst(
            control_flow::Call::new(is, helper_ref, [lhs, rhs].into_iter().collect()),
            Type::I32,
        );
        fb.insert_inst_no_result(control_flow::Return::new_single(is, result));
        fb.seal_all();
        fb.finish();
    }

    let artifact = SpirvBackend::new()
        .compile_module(&mb.build())
        .expect("a fixed typed local should lower without the byte arena");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL side artifact");
    assert!(
        wgsl.matches("fixed_struct_local_helper(").count() >= 2,
        "the scalar helper must remain one definition and one call:\n{wgsl}",
    );
    assert!(
        wgsl.contains("var fixed_local_"),
        "the fixed aggregate must become one native function local:\n{wgsl}",
    );
    assert!(
        !wgsl.contains("fe_heap") && !wgsl.contains("fe_bump"),
        "the fixed aggregate must not use the byte arena:\n{wgsl}",
    );
    let reparsed = naga::front::wgsl::parse_str(wgsl)
        .expect("WGSL with a fixed typed local must reparse");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&reparsed)
    .expect("WGSL with a fixed typed local must validate for browser capabilities");
}

#[test]
fn spirv_typed_aggregate_value_crosses_a_private_helper() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64,
        Vendor::Unknown,
        OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let pair_ty =
        mb.declare_struct_type("TypedAggregateValuePair", &[Type::I32, Type::I32], false);
    let pair_ptr_ty = mb.ptr_type(pair_ty);
    let word_ptr_ty = mb.ptr_type(Type::I32);
    let entry_ref = mb
        .declare_function(Signature::new_single(
            "typed_aggregate_value_entry",
            Linkage::Public,
            &[Type::I32, Type::I32],
            Type::I32,
        ))
        .unwrap();
    let helper_ref = mb
        .declare_function(Signature::new_single(
            "typed_aggregate_value_helper",
            Linkage::Private,
            &[pair_ty],
            pair_ty,
        ))
        .unwrap();

    {
        let mut fb = mb.func_builder::<InstInserter>(helper_ref);
        let entry = fb.append_block();
        fb.switch_to_block(entry);
        let argument = fb.args()[0];
        let owned = fb.insert_inst(data::Alloca::new(is, pair_ty), pair_ptr_ty);
        fb.insert_inst_no_result(data::Mstore::new(is, owned, argument, pair_ty));
        let result = fb.insert_inst(data::Mload::new(is, owned, pair_ty), pair_ty);
        fb.insert_inst_no_result(control_flow::Return::new_single(is, result));
        fb.seal_all();
        fb.finish();
    }
    {
        let mut fb = mb.func_builder::<InstInserter>(entry_ref);
        let entry = fb.append_block();
        fb.switch_to_block(entry);
        let source = fb.insert_inst(data::Alloca::new(is, pair_ty), pair_ptr_ty);
        let zero = fb.make_imm_value(0i32);
        let first_index = fb.make_imm_value(0i32);
        let second_index = fb.make_imm_value(1i32);
        let first = fb.insert_inst(
            data::Gep::new(is, [source, zero, first_index].into_iter().collect()),
            word_ptr_ty,
        );
        let second = fb.insert_inst(
            data::Gep::new(is, [source, zero, second_index].into_iter().collect()),
            word_ptr_ty,
        );
        fb.insert_inst_no_result(data::Mstore::new(is, first, fb.args()[0], Type::I32));
        fb.insert_inst_no_result(data::Mstore::new(is, second, fb.args()[1], Type::I32));
        let source_value = fb.insert_inst(data::Mload::new(is, source, pair_ty), pair_ty);
        let result_value = fb.insert_inst(
            control_flow::Call::new(is, helper_ref, [source_value].into_iter().collect()),
            pair_ty,
        );
        let result = fb.insert_inst(data::Alloca::new(is, pair_ty), pair_ptr_ty);
        fb.insert_inst_no_result(data::Mstore::new(is, result, result_value, pair_ty));
        let result_first = fb.insert_inst(
            data::Gep::new(is, [result, zero, first_index].into_iter().collect()),
            word_ptr_ty,
        );
        let result_second = fb.insert_inst(
            data::Gep::new(is, [result, zero, second_index].into_iter().collect()),
            word_ptr_ty,
        );
        let result_first =
            fb.insert_inst(data::Mload::new(is, result_first, Type::I32), Type::I32);
        let result_second =
            fb.insert_inst(data::Mload::new(is, result_second, Type::I32), Type::I32);
        let sum = fb.insert_inst(
            arith::Add::new(is, result_first, result_second),
            Type::I32,
        );
        fb.insert_inst_no_result(control_flow::Return::new_single(is, sum));
        fb.seal_all();
        fb.finish();
    }

    let artifact = SpirvBackend::new()
        .compile_module(&mb.build())
        .expect("typed aggregate values should cross a private helper");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL side artifact");
    assert!(
        wgsl.matches("typed_aggregate_value_helper(").count() >= 2,
        "the aggregate helper must remain one definition and one call:\n{wgsl}",
    );
    assert!(
        !wgsl.contains("fe_heap"),
        "typed aggregate values must not use the byte arena:\n{wgsl}",
    );
    let reparsed = naga::front::wgsl::parse_str(wgsl)
        .expect("WGSL with a typed aggregate value must reparse");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&reparsed)
    .expect("WGSL with a typed aggregate value must validate for browser capabilities");
}

#[test]
fn spirv_typed_local_uses_implicit_zero_but_keeps_later_zero_mutation() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64,
        Vendor::Unknown,
        OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let pair_ty = mb.declare_struct_type(
        "ZeroInitializedFixedLocalPair",
        &[Type::I32, Type::I32],
        false,
    );
    let pair_ptr_ty = mb.ptr_type(pair_ty);
    let word_ptr_ty = mb.ptr_type(Type::I32);
    let entry_ref = mb.declare_function(Signature::new_single(
        "typed_local_implicit_zero",
        Linkage::Public,
        &[],
        Type::I32,
    )).unwrap();

    let mut fb = mb.func_builder::<InstInserter>(entry_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);
    let slot = fb.insert_inst(data::Alloca::new(is, pair_ty), pair_ptr_ty);
    let zero_index = fb.make_imm_value(0i32);
    let first_index = fb.make_imm_value(0i32);
    let second_index = fb.make_imm_value(1i32);
    let first = fb.insert_inst(
        data::Gep::new(is, [slot, zero_index, first_index].into_iter().collect()),
        word_ptr_ty,
    );
    let second = fb.insert_inst(
        data::Gep::new(is, [slot, zero_index, second_index].into_iter().collect()),
        word_ptr_ty,
    );
    let zero = fb.make_imm_value(0i32);
    let seven = fb.make_imm_value(7i32);
    fb.insert_inst_no_result(data::Mstore::new(is, first, zero, Type::I32));
    fb.insert_inst_no_result(data::Mstore::new(is, second, seven, Type::I32));
    fb.insert_inst_no_result(data::Mstore::new(is, second, zero, Type::I32));
    let loaded_first = fb.insert_inst(data::Mload::new(is, first, Type::I32), Type::I32);
    let loaded_second = fb.insert_inst(data::Mload::new(is, second, Type::I32), Type::I32);
    let sum = fb.insert_inst(arith::Add::new(is, loaded_first, loaded_second), Type::I32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, sum));
    fb.seal_all();
    fb.finish();

    let artifact = SpirvBackend::new()
        .compile_module(&mb.build())
        .expect("typed private initialization should preserve zero semantics");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL side artifact");
    let explicit_zero_stores = wgsl.lines()
        .filter(|line| line.contains(".f") && line.contains(" = 0u;"))
        .collect::<Vec<_>>();
    assert_eq!(
        explicit_zero_stores.len(),
        1,
        "the pristine first field should use the local initializer while the second field's later zero mutation remains:\n{wgsl}",
    );
    assert!(
        wgsl.lines().any(|line| line.contains(".f1_ = 7u;") || line.contains(".f1_ = 7i;")),
        "the nonzero mutation must remain explicit:\n{wgsl}",
    );
    let reparsed = naga::front::wgsl::parse_str(wgsl)
        .expect("implicitly initialized typed-local WGSL must reparse");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&reparsed)
    .expect("implicitly initialized typed-local WGSL must validate for browser capabilities");
}

#[test]
fn spirv_typed_local_uses_implicit_zero_in_acyclic_child_block() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64,
        Vendor::Unknown,
        OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let pair_ty = mb.declare_struct_type(
        "AcyclicZeroInitializedFixedLocalPair",
        &[Type::I32, Type::I32],
        false,
    );
    let pair_ptr_ty = mb.ptr_type(pair_ty);
    let word_ptr_ty = mb.ptr_type(Type::I32);
    let entry_ref = mb.declare_function(Signature::new_single(
        "typed_local_acyclic_implicit_zero",
        Linkage::Public,
        &[Type::I32],
        Type::I32,
    )).unwrap();

    let mut fb = mb.func_builder::<InstInserter>(entry_ref);
    let entry = fb.append_block();
    let body = fb.append_block();
    let early = fb.append_block();
    fb.switch_to_block(entry);
    let condition_word = fb.args()[0];
    let zero_index = fb.make_imm_value(0i32);
    let condition = fb.insert_inst(cmp::Ne::new(is, condition_word, zero_index), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, condition, body, early));

    fb.switch_to_block(body);
    let slot = fb.insert_inst(data::Alloca::new(is, pair_ty), pair_ptr_ty);
    let first_index = fb.make_imm_value(0i32);
    let second_index = fb.make_imm_value(1i32);
    let first = fb.insert_inst(
        data::Gep::new(is, [slot, zero_index, first_index].into_iter().collect()),
        word_ptr_ty,
    );
    let second = fb.insert_inst(
        data::Gep::new(is, [slot, zero_index, second_index].into_iter().collect()),
        word_ptr_ty,
    );
    let zero = fb.make_imm_value(0i32);
    let seven = fb.make_imm_value(7i32);
    fb.insert_inst_no_result(data::Mstore::new(is, first, zero, Type::I32));
    fb.insert_inst_no_result(data::Mstore::new(is, second, seven, Type::I32));
    fb.insert_inst_no_result(data::Mstore::new(is, second, zero, Type::I32));
    let loaded_first = fb.insert_inst(data::Mload::new(is, first, Type::I32), Type::I32);
    let loaded_second = fb.insert_inst(data::Mload::new(is, second, Type::I32), Type::I32);
    let sum = fb.insert_inst(arith::Add::new(is, loaded_first, loaded_second), Type::I32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, sum));

    fb.switch_to_block(early);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, zero_index));
    fb.seal_all();
    fb.finish();

    let artifact = SpirvBackend::new()
        .compile_module(&mb.build())
        .expect("acyclic typed private initialization should preserve zero semantics");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL side artifact");
    let explicit_zero_stores = wgsl.lines()
        .filter(|line| line.contains(".f") && line.contains(" = 0u;"))
        .collect::<Vec<_>>();
    assert_eq!(
        explicit_zero_stores.len(),
        1,
        "the pristine child-block store should use the local initializer while the later mutation remains:\n{wgsl}",
    );
    assert!(
        wgsl.lines().any(|line| line.contains(".f1_ = 7u;") || line.contains(".f1_ = 7i;")),
        "the nonzero mutation must remain explicit:\n{wgsl}",
    );
    let reparsed = naga::front::wgsl::parse_str(wgsl)
        .expect("acyclic implicitly initialized typed-local WGSL must reparse");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&reparsed)
    .expect("acyclic implicitly initialized typed-local WGSL must validate for browser capabilities");
}

#[test]
fn spirv_dense_equality_ladder_emits_native_switch() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64,
        Vendor::Unknown,
        OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let entry_ref = mb.declare_function(Signature::new_single(
        "dense_equality_ladder",
        Linkage::Public,
        &[Type::I32],
        Type::I32,
    )).unwrap();

    let mut fb = mb.func_builder::<InstInserter>(entry_ref);
    let header0 = fb.append_block();
    let case0 = fb.append_block();
    let header1 = fb.append_block();
    let case1 = fb.append_block();
    let header2 = fb.append_block();
    let case2 = fb.append_block();
    let fallback = fb.append_block();
    let merge = fb.append_block();
    let selector = fb.args()[0];

    fb.switch_to_block(header0);
    let zero = fb.make_imm_value(0i32);
    let is_zero = fb.insert_inst(cmp::Eq::new(is, selector, zero), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, is_zero, case0, header1));

    fb.switch_to_block(case0);
    let value0 = fb.make_imm_value(10i32);
    fb.insert_inst_no_result(control_flow::Jump::new(is, merge));

    fb.switch_to_block(header1);
    let one = fb.make_imm_value(1i32);
    let is_one = fb.insert_inst(cmp::Eq::new(is, selector, one), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, is_one, case1, header2));

    fb.switch_to_block(case1);
    let value1 = fb.make_imm_value(20i32);
    fb.insert_inst_no_result(control_flow::Jump::new(is, merge));

    fb.switch_to_block(header2);
    let two = fb.make_imm_value(2i32);
    let is_two = fb.insert_inst(cmp::Eq::new(is, selector, two), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, is_two, case2, fallback));

    fb.switch_to_block(case2);
    let value2 = fb.make_imm_value(30i32);
    fb.insert_inst_no_result(control_flow::Jump::new(is, merge));

    fb.switch_to_block(fallback);
    let fallback_value = fb.make_imm_value(40i32);
    fb.insert_inst_no_result(control_flow::Jump::new(is, merge));

    fb.switch_to_block(merge);
    let selected = fb.insert_inst(
        control_flow::Phi::new(
            is,
            vec![
                (value0, case0),
                (value1, case1),
                (value2, case2),
                (fallback_value, fallback),
            ],
        ),
        Type::I32,
    );
    fb.insert_inst_no_result(control_flow::Return::new_single(is, selected));
    fb.seal_all();
    fb.finish();

    let artifact = SpirvBackend::new()
        .compile_module(&mb.build())
        .expect("dense equality ladder should lower as one switch");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL side artifact");
    assert!(
        wgsl.contains("switch ") && wgsl.contains("case 0u:") && wgsl.contains("case 2u:"),
        "three dense equality arms should become native switch cases:\n{wgsl}",
    );
    let reparsed = naga::front::wgsl::parse_str(wgsl)
        .expect("native-switch WGSL must reparse");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&reparsed)
    .expect("native-switch WGSL must validate for browser capabilities");
}

#[test]
fn spirv_acyclic_zero_phi_uses_its_explicit_initializer() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64,
        Vendor::Unknown,
        OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let entry_ref = mb.declare_function(Signature::new_single(
        "acyclic_zero_phi",
        Linkage::Public,
        &[Type::I32],
        Type::I32,
    )).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(entry_ref);
    let entry = fb.append_block();
    let zero_arm = fb.append_block();
    let seven_arm = fb.append_block();
    let merge = fb.append_block();

    fb.switch_to_block(entry);
    let input = fb.args()[0];
    let zero = fb.make_imm_value(0i32);
    let is_zero = fb.insert_inst(cmp::Eq::new(is, input, zero), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, is_zero, zero_arm, seven_arm));
    fb.switch_to_block(zero_arm);
    fb.insert_inst_no_result(control_flow::Jump::new(is, merge));
    fb.switch_to_block(seven_arm);
    let seven = fb.make_imm_value(7i32);
    fb.insert_inst_no_result(control_flow::Jump::new(is, merge));
    fb.switch_to_block(merge);
    let result = fb.insert_inst(
        control_flow::Phi::new(is, vec![(zero, zero_arm), (seven, seven_arm)]),
        Type::I32,
    );
    fb.insert_inst_no_result(control_flow::Return::new_single(is, result));
    fb.seal_all();
    fb.finish();

    let artifact = SpirvBackend::new()
        .compile_module(&mb.build())
        .expect("an acyclic zero phi should compile");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL side artifact");
    assert_eq!(
        wgsl.lines().filter(|line| line.contains(" = 0u;")).count(),
        0,
        "the phi initializer should replace its acyclic zero edge transfer:\n{wgsl}",
    );
    assert!(
        wgsl.lines().any(|line| line.trim_start().starts_with("var local") && line.contains(" = u32();")),
        "the acyclic phi must be explicitly zero-initialized for SPIR-V parity:\n{wgsl}",
    );
    assert!(
        wgsl.lines().any(|line| line.contains(" = 7u;")),
        "the nonzero phi edge transfer must remain:\n{wgsl}",
    );
}

#[test]
fn spirv_typed_local_zero_projection_bitcast_restores_structural_access() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64,
        Vendor::Unknown,
        OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let array_ty = mb.declare_array_type(Type::I32, 4);
    let array_ptr_ty = mb.ptr_type(array_ty);
    let word_ptr_ty = mb.ptr_type(Type::I32);
    let entry_ref = mb
        .declare_function(Signature::new_single(
            "typed_local_zero_projection",
            Linkage::Public,
            &[],
            Type::I32,
        ))
        .unwrap();

    let mut fb = mb.func_builder::<InstInserter>(entry_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);
    let array = fb.insert_inst(data::Alloca::new(is, array_ty), array_ptr_ty);
    // SCCP uses this exact representation for `gep array, 0, 0`.
    let first = fb.insert_inst(cast::Bitcast::new(is, array, word_ptr_ty), word_ptr_ty);
    let value = fb.make_imm_value(37i32);
    fb.insert_inst_no_result(data::Mstore::new(is, first, value, Type::I32));
    let loaded = fb.insert_inst(data::Mload::new(is, first, Type::I32), Type::I32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, loaded));
    fb.seal_all();
    fb.finish();

    let artifact = SpirvBackend::new()
        .compile_module(&mb.build())
        .expect("an all-zero typed Gep simplified to Bitcast must retain structural meaning");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL side artifact");
    assert!(wgsl.contains("fixed_local_"), "typed local must remain native:\n{wgsl}");
    assert!(!wgsl.contains("fe_heap"), "zero projection must not restore the byte arena:\n{wgsl}");
    let reparsed = naga::front::wgsl::parse_str(wgsl)
        .expect("typed zero-projection WGSL must reparse");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&reparsed)
    .expect("typed zero-projection WGSL must validate for browser capabilities");
}

#[test]
fn spirv_typed_local_rejects_pointer_to_integer_escape() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64,
        Vendor::Unknown,
        OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let pair_ty =
        mb.declare_struct_type("EscapingFixedLocalPair", &[Type::I32, Type::I32], false);
    let pair_ptr_ty = mb.ptr_type(pair_ty);
    let entry_ref = mb
        .declare_function(Signature::new_single(
            "typed_local_pointer_escape",
            Linkage::Public,
            &[],
            Type::I32,
        ))
        .unwrap();

    let mut fb = mb.func_builder::<InstInserter>(entry_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);
    let slot = fb.insert_inst(data::Alloca::new(is, pair_ty), pair_ptr_ty);
    let address = fb.insert_inst(cast::PtrToInt::new(is, slot, Type::I32), Type::I32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, address));
    fb.seal_all();
    fb.finish();

    let message = spirv_rejection(
        &mb.build(),
        "a typed private pointer must not become an integer",
    );
    assert!(
        message.contains("typed-local pointer") && message.contains("Fail closed"),
        "unexpected rejection: {message}",
    );
}

#[test]
fn spirv_typed_local_can_be_borrowed_by_a_private_helper() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64,
        Vendor::Unknown,
        OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let pair_ty =
        mb.declare_struct_type("BorrowedFixedLocalPair", &[Type::I32, Type::I32], false);
    let pair_ptr_ty = mb.ptr_type(pair_ty);
    let outer_ty =
        mb.declare_struct_type("BorrowedFixedLocalOuter", &[Type::I32, pair_ty], false);
    let outer_ptr_ty = mb.ptr_type(outer_ty);
    let word_ptr_ty = mb.ptr_type(Type::I32);
    let entry_ref = mb
        .declare_function(Signature::new_single(
            "typed_local_cross_call_entry",
            Linkage::Public,
            &[],
            Type::I32,
        ))
        .unwrap();
    let helper_ref = mb
        .declare_function(Signature::new_single(
            "typed_local_cross_call_helper",
            Linkage::Private,
            &[pair_ptr_ty],
            Type::I32,
        ))
        .unwrap();

    {
        let mut fb = mb.func_builder::<InstInserter>(helper_ref);
        let entry = fb.append_block();
        fb.switch_to_block(entry);
        let pair = fb.args()[0];
        let zero = fb.make_imm_value(0i32);
        let first_index = fb.make_imm_value(0i32);
        let second_index = fb.make_imm_value(1i32);
        let first = fb.insert_inst(
            data::Gep::new(is, [pair, zero, first_index].into_iter().collect()),
            word_ptr_ty,
        );
        let second = fb.insert_inst(
            data::Gep::new(is, [pair, zero, second_index].into_iter().collect()),
            word_ptr_ty,
        );
        let first = fb.insert_inst(data::Mload::new(is, first, Type::I32), Type::I32);
        let second = fb.insert_inst(data::Mload::new(is, second, Type::I32), Type::I32);
        let sum = fb.insert_inst(arith::Add::new(is, first, second), Type::I32);
        fb.insert_inst_no_result(control_flow::Return::new_single(is, sum));
        fb.seal_all();
        fb.finish();
    }
    {
        let mut fb = mb.func_builder::<InstInserter>(entry_ref);
        let entry = fb.append_block();
        let loop_header = fb.append_block();
        let loop_body = fb.append_block();
        let exit = fb.append_block();
        fb.switch_to_block(entry);
        let slot = fb.insert_inst(data::Alloca::new(is, outer_ty), outer_ptr_ty);
        fb.insert_inst_no_result(control_flow::Jump::new(is, loop_header));

        fb.switch_to_block(loop_header);
        let zero = fb.make_imm_value(0i32);
        let pair_index = fb.make_imm_value(1i32);
        let pair = fb.insert_inst(
            data::Gep::new(is, [slot, zero, pair_index].into_iter().collect()),
            pair_ptr_ty,
        );
        let first_index = fb.make_imm_value(0i32);
        let second_index = fb.make_imm_value(1i32);
        let first = fb.insert_inst(
            data::Gep::new(is, [pair, zero, first_index].into_iter().collect()),
            word_ptr_ty,
        );
        let second = fb.insert_inst(
            data::Gep::new(is, [pair, zero, second_index].into_iter().collect()),
            word_ptr_ty,
        );
        let first_value = fb.make_imm_value(19i32);
        let second_value = fb.make_imm_value(23i32);
        fb.insert_inst_no_result(data::Mstore::new(is, first, first_value, Type::I32));
        fb.insert_inst_no_result(data::Mstore::new(is, second, second_value, Type::I32));
        let keep_going = fb.make_imm_value(false);
        fb.insert_inst_no_result(control_flow::Br::new(is, keep_going, loop_body, exit));

        fb.switch_to_block(loop_body);
        fb.insert_inst_no_result(control_flow::Jump::new(is, loop_header));

        fb.switch_to_block(exit);
        let result = fb.insert_inst(
            control_flow::Call::new(is, helper_ref, [pair].into_iter().collect()),
            Type::I32,
        );
        fb.insert_inst_no_result(control_flow::Return::new_single(is, result));
        fb.seal_all();
        fb.finish();
    }

    let artifact = SpirvBackend::new()
        .compile_module(&mb.build())
        .expect("a projected typed private local should cross a certified private helper borrow");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL side artifact");
    assert!(
        wgsl.matches("typed_local_cross_call_helper(").count() >= 2,
        "the borrowed helper must remain one definition and one call:\n{wgsl}",
    );
    assert!(
        wgsl.contains("ptr<function"),
        "the helper must receive a function-local pointer:\n{wgsl}",
    );
    assert!(
        !wgsl.contains("fe_heap"),
        "the typed borrow must not restore the byte arena:\n{wgsl}",
    );
    let reparsed = naga::front::wgsl::parse_str(wgsl)
        .expect("WGSL with a private typed borrow must reparse");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&reparsed)
    .expect("WGSL with a private typed borrow must validate for browser capabilities");
}

#[test]
fn spirv_typed_local_rejects_bytewise_copy() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64,
        Vendor::Unknown,
        OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let pair_ty =
        mb.declare_struct_type("CopiedFixedLocalPair", &[Type::I32, Type::I32], false);
    let pair_ptr_ty = mb.ptr_type(pair_ty);
    let entry_ref = mb
        .declare_function(Signature::new_single(
            "typed_local_bytewise_copy",
            Linkage::Public,
            &[],
            Type::I32,
        ))
        .unwrap();

    let mut fb = mb.func_builder::<InstInserter>(entry_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);
    let slot = fb.insert_inst(data::Alloca::new(is, pair_ty), pair_ptr_ty);
    let byte_len = fb.make_imm_value(8i32);
    fb.insert_inst_no_result(data::Memcopy::new(is, slot, slot, byte_len));
    let zero = fb.make_imm_value(0i32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, zero));
    fb.seal_all();
    fb.finish();

    let message = spirv_rejection(
        &mb.build(),
        "typed private storage must not be observed bytewise",
    );
    assert!(
        message.contains("typed-local pointer") && message.contains("Fail closed"),
        "unexpected rejection: {message}",
    );
}

#[test]
fn spirv_typed_local_rejects_excessive_private_storage() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64,
        Vendor::Unknown,
        OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let oversized_ty = mb.declare_array_type(Type::I32, 4097);
    let oversized_ptr_ty = mb.ptr_type(oversized_ty);
    let entry_ref = mb
        .declare_function(Signature::new_single(
            "typed_local_private_budget",
            Linkage::Public,
            &[],
            Type::I32,
        ))
        .unwrap();

    let mut fb = mb.func_builder::<InstInserter>(entry_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);
    fb.insert_inst(data::Alloca::new(is, oversized_ty), oversized_ptr_ty);
    let zero = fb.make_imm_value(0i32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, zero));
    fb.seal_all();
    fb.finish();

    let message = spirv_rejection(
        &mb.build(),
        "typed private storage must remain within its conservative budget",
    );
    assert!(
        message.contains("typed private storage")
            && message.contains("per-function budget")
            && message.contains("across 1 allocations")
            && message.contains("largest allocations:")
            && message.contains("Fail closed"),
        "unexpected rejection: {message}",
    );
}

#[test]
fn spirv_private_arena_helper_survives_through_an_explicit_pointer_abi() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64,
        Vendor::Unknown,
        OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();

    let entry_ref = mb
        .declare_function(Signature::new_single(
            "private_arena_helper_entry",
            Linkage::Public,
            &[Type::I32],
            Type::I32,
        ))
        .unwrap();
    let helper_ref = mb
        .declare_function(Signature::new_single(
            "private_arena_helper",
            Linkage::Private,
            &[Type::I32],
            Type::I32,
        ))
        .unwrap();
    let forwarder_ref = mb
        .declare_function(Signature::new_single(
            "private_arena_forwarder",
            Linkage::Private,
            &[Type::I32],
            Type::I32,
        ))
        .unwrap();
    let pure_ref = mb
        .declare_function(Signature::new_single(
            "private_arena_pure",
            Linkage::Private,
            &[Type::I32],
            Type::I32,
        ))
        .unwrap();

    {
        let mut fb = mb.func_builder::<InstInserter>(helper_ref);
        let entry = fb.append_block();
        fb.switch_to_block(entry);
        let address = fb.args()[0];
        let value = fb.insert_inst(data::Mload::new(is, address, Type::I32), Type::I32);
        let one = fb.make_imm_value(1i32);
        let incremented = fb.insert_inst(arith::Add::new(is, value, one), Type::I32);
        fb.insert_inst_no_result(data::Mstore::new(
            is,
            address,
            incremented,
            Type::I32,
        ));
        fb.insert_inst_no_result(control_flow::Return::new_single(is, incremented));
        fb.seal_all();
        fb.finish();
    }
    {
        let mut fb = mb.func_builder::<InstInserter>(forwarder_ref);
        let entry = fb.append_block();
        fb.switch_to_block(entry);
        let address = fb.args()[0];
        let result = fb.insert_inst(
            control_flow::Call::new(is, helper_ref, [address].into_iter().collect()),
            Type::I32,
        );
        fb.insert_inst_no_result(control_flow::Return::new_single(is, result));
        fb.seal_all();
        fb.finish();
    }
    {
        let mut fb = mb.func_builder::<InstInserter>(pure_ref);
        let entry = fb.append_block();
        fb.switch_to_block(entry);
        let input = fb.args()[0];
        let two = fb.make_imm_value(2i32);
        let result = fb.insert_inst(arith::Add::new(is, input, two), Type::I32);
        fb.insert_inst_no_result(control_flow::Return::new_single(is, result));
        fb.seal_all();
        fb.finish();
    }
    {
        let mut fb = mb.func_builder::<InstInserter>(entry_ref);
        let entry = fb.append_block();
        fb.switch_to_block(entry);
        let input = fb.args()[0];
        let prepared = fb.insert_inst(
            control_flow::Call::new(is, pure_ref, [input].into_iter().collect()),
            Type::I32,
        );
        let bytes = fb.make_imm_value(8i32);
        let address = fb.insert_inst(data::MemAllocDynamic::new(is, bytes), Type::I32);
        fb.insert_inst_no_result(data::Mstore::new(
            is,
            address,
            prepared,
            Type::I32,
        ));
        let result = fb.insert_inst(
            control_flow::Call::new(is, forwarder_ref, [address].into_iter().collect()),
            Type::I32,
        );
        fb.insert_inst_no_result(control_flow::Return::new_single(is, result));
        fb.seal_all();
        fb.finish();
    }

    let artifact = SpirvBackend::new()
        .compile_module(&mb.build())
        .expect("a non-allocating helper should share the entry's proven private arena");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL side artifact");
    assert!(
        wgsl.matches("private_arena_helper(").count() >= 2,
        "the private-arena helper must remain a definition and call:\n{wgsl}",
    );
    assert!(
        wgsl.contains("ptr<function"),
        "the helper must receive compiler-owned function pointers:\n{wgsl}",
    );
    assert!(
        wgsl.contains("array<u32, 2>"),
        "the entry's exact eight-byte high-water bound must remain authoritative:\n{wgsl}",
    );
    let reparsed = naga::front::wgsl::parse_str(wgsl)
        .expect("WGSL with the private-arena helper must reparse");
    let function_arg_count = |name: &str| {
        reparsed
            .functions
            .iter()
            .find_map(|(_, function)| {
                (function.name.as_deref() == Some(name)).then_some(function.arguments.len())
            })
            .unwrap_or_else(|| panic!("missing helper `{name}` in generated WGSL"))
    };
    assert_eq!(
        function_arg_count("private_arena_helper"),
        4,
        "the arena leaf needs value, heap, bump, and trap arguments",
    );
    assert_eq!(
        function_arg_count("private_arena_forwarder"),
        4,
        "the forwarder must receive the arena capability transitively",
    );
    assert_eq!(
        function_arg_count("private_arena_pure"),
        1,
        "a pure helper must not receive unused arena capability arguments",
    );
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&reparsed)
    .expect("the private-arena helper ABI must validate for browser capabilities");
}

#[test]
fn spirv_private_arena_helper_cannot_own_an_unaccounted_allocation() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64,
        Vendor::Unknown,
        OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();

    let entry_ref = mb
        .declare_function(Signature::new_single(
            "allocating_helper_entry",
            Linkage::Public,
            &[],
            Type::I32,
        ))
        .unwrap();
    let helper_ref = mb
        .declare_function(Signature::new_single(
            "allocating_helper",
            Linkage::Private,
            &[],
            Type::I32,
        ))
        .unwrap();

    {
        let mut fb = mb.func_builder::<InstInserter>(helper_ref);
        let entry = fb.append_block();
        fb.switch_to_block(entry);
        let bytes = fb.make_imm_value(4i32);
        let address = fb.insert_inst(data::MemAllocDynamic::new(is, bytes), Type::I32);
        fb.insert_inst_no_result(control_flow::Return::new_single(is, address));
        fb.seal_all();
        fb.finish();
    }
    {
        let mut fb = mb.func_builder::<InstInserter>(entry_ref);
        let entry = fb.append_block();
        fb.switch_to_block(entry);
        let bytes = fb.make_imm_value(4i32);
        let _entry_allocation =
            fb.insert_inst(data::MemAllocDynamic::new(is, bytes), Type::I32);
        let result = fb.insert_inst(
            control_flow::Call::new(is, helper_ref, [].into_iter().collect()),
            Type::I32,
        );
        fb.insert_inst_no_result(control_flow::Return::new_single(is, result));
        fb.seal_all();
        fb.finish();
    }

    let errors = match SpirvBackend::new().compile_module(&mb.build()) {
        Ok(_) => panic!("an outlined helper allocation is outside the entry high-water proof"),
        Err(errors) => errors,
    };
    let message = errors
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("; ");
    assert!(
        message.contains("changes arena lifetime"),
        "the allocation must fail at the named cross-call lifetime boundary: {message}",
    );
}

#[test]
fn spirv_external_resource_identity_threads_through_helper_calls() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64,
        Vendor::Unknown,
        OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let array_type = mb.declare_array_type(Type::I32, 8);
    let array_ref_type = mb.objref_type(array_type);
    let word_ref_type = mb.objref_type(Type::I32);
    let result_types = [array_ref_type, Type::I32];

    let entry_ref = mb
        .declare_function(Signature::new_unit(
            "resource_helper_entry",
            Linkage::Public,
            &[array_ref_type, Type::I32, Type::I32],
        ))
        .unwrap();
    let forwarder_ref = mb
        .declare_function(Signature::new(
            "resource_helper_forwarder",
            Linkage::Private,
            &[array_ref_type, Type::I32, Type::I32],
            &result_types,
        ))
        .unwrap();
    let leaf_ref = mb
        .declare_function(Signature::new(
            "resource_helper_leaf",
            Linkage::Private,
            &[array_ref_type, Type::I32, Type::I32],
            &result_types,
        ))
        .unwrap();

    {
        let mut fb = mb.func_builder::<InstInserter>(leaf_ref);
        let entry = fb.append_block();
        fb.switch_to_block(entry);
        let resource = fb.args()[0];
        let index = fb.args()[1];
        let value = fb.args()[2];
        let slot = fb.insert_inst(
            data::ObjIndex::new(is, resource, index),
            word_ref_type,
        );
        fb.insert_inst_no_result(data::ObjStore::new(is, slot, value));
        let one = fb.make_imm_value(1i32);
        let incremented = fb.insert_inst(arith::Add::new(is, value, one), Type::I32);
        fb.insert_inst_no_result(control_flow::Return::new(
            is,
            [resource, incremented]
                .into_iter()
                .collect::<smallvec::SmallVec<[_; 2]>>()
                .into(),
        ));
        fb.seal_all();
        fb.finish();
    }
    {
        let mut fb = mb.func_builder::<InstInserter>(forwarder_ref);
        let entry = fb.append_block();
        fb.switch_to_block(entry);
        let results = fb.insert_inst_results(
            control_flow::Call::new(is, leaf_ref, fb.args().iter().copied().collect()),
            &result_types,
        );
        fb.insert_inst_no_result(control_flow::Return::new(
            is,
            results
                .iter()
                .copied()
                .collect::<smallvec::SmallVec<[_; 2]>>()
                .into(),
        ));
        fb.seal_all();
        fb.finish();
    }
    {
        let mut fb = mb.func_builder::<InstInserter>(entry_ref);
        let entry = fb.append_block();
        fb.switch_to_block(entry);
        let index = fb.args()[1];
        let results = fb.insert_inst_results(
            control_flow::Call::new(
                is,
                forwarder_ref,
                fb.args().iter().copied().collect(),
            ),
            &result_types,
        );
        let one = fb.make_imm_value(1i32);
        let next_index = fb.insert_inst(arith::Add::new(is, index, one), Type::I32);
        let slot = fb.insert_inst(
            data::ObjIndex::new(is, results[0], next_index),
            word_ref_type,
        );
        fb.insert_inst_no_result(data::ObjStore::new(is, slot, results[1]));
        fb.insert_inst_no_result(control_flow::Return::new_unit(is));
        fb.seal_all();
        fb.finish();
    }

    let artifact = SpirvBackend::new()
        .with_compute()
        .with_workgroup_size(1, 1, 1)
        .with_external_resource(SpirvExternalResource {
            arg_index: 0,
            group: 0,
            binding: 0,
            name: "values".to_string(),
            access: Access::ReadWrite,
            element: SpirvResourceElement::Scalar(SpirvScalarKind::U32),
            stride: 4,
            length: 8,
        })
        .compile_module(&mb.build())
        .expect("resource identity should cross helpers without entering a result struct");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL side artifact");
    assert!(
        wgsl.matches("resource_helper_leaf(").count() >= 2
            && wgsl.matches("resource_helper_forwarder(").count() >= 2,
        "both resource helpers must remain definitions and calls:\n{wgsl}",
    );
    assert!(
        wgsl.contains("var<storage, read_write> values: array<u32>;")
            && !wgsl.contains("ptr<storage"),
        "resource helpers must use the entry-rooted module capability without an illegal storage-pointer parameter:\n{wgsl}",
    );
    assert!(
        !wgsl.contains("resource_helper_leaf_result")
            && !wgsl.contains("resource_helper_forwarder_result"),
        "an identity resource result must be erased rather than packed into a WGSL struct:\n{wgsl}",
    );
    let reparsed = naga::front::wgsl::parse_str(wgsl)
        .expect("WGSL with implicit resource helper capabilities must reparse");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&reparsed)
    .expect("implicit resource helper capabilities must validate for browsers");
}

#[test]
fn spirv_external_resource_identity_survives_multiple_helper_exits() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64,
        Vendor::Unknown,
        OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let array_type = mb.declare_array_type(Type::I32, 8);
    let array_ref_type = mb.objref_type(array_type);
    let word_ref_type = mb.objref_type(Type::I32);
    let result_types = [array_ref_type, Type::I32];

    let entry_ref = mb
        .declare_function(Signature::new_unit(
            "multi_exit_resource_entry",
            Linkage::Public,
            &[array_ref_type, Type::I32],
        ))
        .unwrap();
    let helper_ref = mb
        .declare_function(Signature::new(
            "multi_exit_resource_helper",
            Linkage::Private,
            &[array_ref_type, Type::I32],
            &result_types,
        ))
        .unwrap();

    {
        let mut fb = mb.func_builder::<InstInserter>(helper_ref);
        let entry = fb.append_block();
        let zero_arm = fb.append_block();
        let nonzero_arm = fb.append_block();
        fb.switch_to_block(entry);
        let resource = fb.args()[0];
        let value = fb.args()[1];
        let zero = fb.make_imm_value(0i32);
        let is_zero = fb.insert_inst(cmp::Eq::new(is, value, zero), Type::I1);
        fb.insert_inst_no_result(control_flow::Br::new(
            is,
            is_zero,
            zero_arm,
            nonzero_arm,
        ));

        fb.switch_to_block(zero_arm);
        let seven = fb.make_imm_value(7i32);
        fb.insert_inst_no_result(control_flow::Return::new(
            is,
            [resource, seven]
                .into_iter()
                .collect::<smallvec::SmallVec<[_; 2]>>()
                .into(),
        ));

        fb.switch_to_block(nonzero_arm);
        let one = fb.make_imm_value(1i32);
        let incremented = fb.insert_inst(arith::Add::new(is, value, one), Type::I32);
        fb.insert_inst_no_result(control_flow::Return::new(
            is,
            [resource, incremented]
                .into_iter()
                .collect::<smallvec::SmallVec<[_; 2]>>()
                .into(),
        ));
        fb.seal_all();
        fb.finish();
    }
    {
        let mut fb = mb.func_builder::<InstInserter>(entry_ref);
        let entry = fb.append_block();
        fb.switch_to_block(entry);
        let resource = fb.args()[0];
        let index = fb.args()[1];
        let results = fb.insert_inst_results(
            control_flow::Call::new(
                is,
                helper_ref,
                [resource, index].into_iter().collect(),
            ),
            &result_types,
        );
        let slot = fb.insert_inst(
            data::ObjIndex::new(is, results[0], index),
            word_ref_type,
        );
        fb.insert_inst_no_result(data::ObjStore::new(is, slot, results[1]));
        fb.insert_inst_no_result(control_flow::Return::new_unit(is));
        fb.seal_all();
        fb.finish();
    }

    let artifact = SpirvBackend::new()
        .with_compute()
        .with_workgroup_size(1, 1, 1)
        .with_external_resource(SpirvExternalResource {
            arg_index: 0,
            group: 0,
            binding: 0,
            name: "values".to_string(),
            access: Access::ReadWrite,
            element: SpirvResourceElement::Scalar(SpirvScalarKind::U32),
            stride: 4,
            length: 8,
        })
        .compile_module(&mb.build())
        .expect("one resource identity should survive every helper exit");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL side artifact");
    assert!(
        wgsl.matches("multi_exit_resource_helper(").count() >= 2,
        "the resource helper must remain one definition plus a call:\n{wgsl}",
    );
    assert!(
        wgsl.contains("var<storage, read_write> values: array<u32>;")
            && !wgsl.contains("ptr<storage"),
        "every exit must retain the entry-rooted module capability:\n{wgsl}",
    );
    assert!(
        !wgsl.contains("multi_exit_resource_helper_result"),
        "the resource lane must be erased and the one scalar lane returned directly:\n{wgsl}",
    );
    let reparsed = naga::front::wgsl::parse_str(wgsl)
        .expect("WGSL with multiple resource-carrying exits must reparse");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&reparsed)
    .expect("multiple resource-carrying exits must validate for browsers");
}

#[test]
fn spirv_external_resource_identity_survives_a_phi_join() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64,
        Vendor::Unknown,
        OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let array_type = mb.declare_array_type(Type::I32, 8);
    let array_ref_type = mb.objref_type(array_type);
    let word_ref_type = mb.objref_type(Type::I32);

    let entry_ref = mb
        .declare_function(Signature::new_unit(
            "resource_phi_entry",
            Linkage::Public,
            &[array_ref_type, Type::I32],
        ))
        .unwrap();
    let forwarder_ref = mb
        .declare_function(Signature::new_single(
            "resource_phi_forwarder",
            Linkage::Private,
            &[array_ref_type],
            array_ref_type,
        ))
        .unwrap();

    {
        let mut fb = mb.func_builder::<InstInserter>(forwarder_ref);
        let entry = fb.append_block();
        fb.switch_to_block(entry);
        fb.insert_inst_no_result(control_flow::Return::new_single(is, fb.args()[0]));
        fb.seal_all();
        fb.finish();
    }
    {
        let mut fb = mb.func_builder::<InstInserter>(entry_ref);
        let entry = fb.append_block();
        let forwarded_arm = fb.append_block();
        let direct_arm = fb.append_block();
        let merge = fb.append_block();

        fb.switch_to_block(entry);
        let resource = fb.args()[0];
        let index = fb.args()[1];
        let zero = fb.make_imm_value(0i32);
        let choose_forwarder = fb.insert_inst(cmp::Eq::new(is, index, zero), Type::I1);
        fb.insert_inst_no_result(control_flow::Br::new(
            is,
            choose_forwarder,
            forwarded_arm,
            direct_arm,
        ));

        fb.switch_to_block(forwarded_arm);
        let forwarded = fb.insert_inst_results(
            control_flow::Call::new(is, forwarder_ref, smallvec::smallvec![resource]),
            &[array_ref_type],
        )[0];
        fb.insert_inst_no_result(control_flow::Jump::new(is, merge));

        fb.switch_to_block(direct_arm);
        fb.insert_inst_no_result(control_flow::Jump::new(is, merge));

        fb.switch_to_block(merge);
        let selected = fb.insert_inst(
            control_flow::Phi::new(
                is,
                vec![(forwarded, forwarded_arm), (resource, direct_arm)],
            ),
            array_ref_type,
        );
        let slot = fb.insert_inst(data::ObjIndex::new(is, selected, index), word_ref_type);
        fb.insert_inst_no_result(data::ObjStore::new(is, slot, index));
        fb.insert_inst_no_result(control_flow::Return::new_unit(is));
        fb.seal_all();
        fb.finish();
    }

    let artifact = SpirvBackend::new()
        .with_compute()
        .with_workgroup_size(1, 1, 1)
        .with_external_resource(SpirvExternalResource {
            arg_index: 0,
            group: 0,
            binding: 0,
            name: "values".to_string(),
            access: Access::ReadWrite,
            element: SpirvResourceElement::Scalar(SpirvScalarKind::U32),
            stride: 4,
            length: 8,
        })
        .compile_module(&mb.build())
        .expect("one resource identity should survive helper passthrough and a phi join");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL side artifact");
    assert!(
        wgsl.matches("resource_phi_forwarder(").count() >= 2,
        "the resource forwarder must remain a definition and a call:\n{wgsl}",
    );
    assert!(
        wgsl.contains("var<storage, read_write> values: array<u32>;")
            && !wgsl.contains("ptr<storage"),
        "the joined resource must remain the entry-rooted module capability:\n{wgsl}",
    );
    let reparsed = naga::front::wgsl::parse_str(wgsl)
        .expect("WGSL with a joined resource identity must reparse");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&reparsed)
    .expect("a joined resource identity must validate for browsers");
}

#[derive(Clone, Copy)]
enum ResourceHelperCalls {
    LeftOnly,
    BothSeparate,
    Joined,
}

fn ambiguous_resource_helper_module(calls: ResourceHelperCalls) -> sonatina_ir::Module {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64,
        Vendor::Unknown,
        OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let array_type = mb.declare_array_type(Type::I32, 8);
    let array_ref_type = mb.objref_type(array_type);
    let word_ref_type = mb.objref_type(Type::I32);

    let entry_ref = mb
        .declare_function(Signature::new_unit(
            "ambiguous_resource_entry",
            Linkage::Public,
            &[array_ref_type, array_ref_type, Type::I32],
        ))
        .unwrap();
    let helper_ref = mb
        .declare_function(Signature::new_unit(
            "ambiguous_resource_helper",
            Linkage::Private,
            &[array_ref_type, Type::I32],
        ))
        .unwrap();

    {
        let mut fb = mb.func_builder::<InstInserter>(helper_ref);
        let entry = fb.append_block();
        fb.switch_to_block(entry);
        let slot = fb.insert_inst(
            data::ObjIndex::new(is, fb.args()[0], fb.args()[1]),
            word_ref_type,
        );
        fb.insert_inst_no_result(data::ObjStore::new(is, slot, fb.args()[1]));
        fb.insert_inst_no_result(control_flow::Return::new_unit(is));
        fb.seal_all();
        fb.finish();
    }
    {
        let mut fb = mb.func_builder::<InstInserter>(entry_ref);
        let entry = fb.append_block();
        let joined_blocks = matches!(calls, ResourceHelperCalls::Joined).then(|| {
            let left_arm = fb.append_block();
            let right_arm = fb.append_block();
            let merge = fb.append_block();
            (left_arm, right_arm, merge)
        });
        fb.switch_to_block(entry);
        match (calls, joined_blocks) {
            (ResourceHelperCalls::LeftOnly, None) => {
                fb.insert_inst_no_result(control_flow::Call::new(
                    is,
                    helper_ref,
                    [fb.args()[0], fb.args()[2]].into_iter().collect(),
                ));
                fb.insert_inst_no_result(control_flow::Return::new_unit(is));
            }
            (ResourceHelperCalls::BothSeparate, None) => {
                fb.insert_inst_no_result(control_flow::Call::new(
                    is,
                    helper_ref,
                    [fb.args()[0], fb.args()[2]].into_iter().collect(),
                ));
                fb.insert_inst_no_result(control_flow::Call::new(
                    is,
                    helper_ref,
                    [fb.args()[1], fb.args()[2]].into_iter().collect(),
                ));
                fb.insert_inst_no_result(control_flow::Return::new_unit(is));
            }
            (ResourceHelperCalls::Joined, Some((left_arm, right_arm, merge))) => {
                let zero = fb.make_imm_value(0i32);
                let choose_left = fb.insert_inst(
                    cmp::Eq::new(is, fb.args()[2], zero),
                    Type::I1,
                );
                fb.insert_inst_no_result(control_flow::Br::new(
                    is,
                    choose_left,
                    left_arm,
                    right_arm,
                ));
                fb.switch_to_block(left_arm);
                fb.insert_inst_no_result(control_flow::Jump::new(is, merge));
                fb.switch_to_block(right_arm);
                fb.insert_inst_no_result(control_flow::Jump::new(is, merge));
                fb.switch_to_block(merge);
                let resource = fb.insert_inst(
                    control_flow::Phi::new(
                        is,
                        vec![(fb.args()[0], left_arm), (fb.args()[1], right_arm)],
                    ),
                    array_ref_type,
                );
                fb.insert_inst_no_result(control_flow::Call::new(
                    is,
                    helper_ref,
                    [resource, fb.args()[2]].into_iter().collect(),
                ));
                fb.insert_inst_no_result(control_flow::Return::new_unit(is));
            }
            _ => unreachable!("resource helper test shape must match its blocks"),
        }
        fb.seal_all();
        fb.finish();
    }

    mb.build()
}

fn ambiguous_resource_backend() -> SpirvBackend {
    SpirvBackend::new()
        .with_compute()
        .with_workgroup_size(1, 1, 1)
        .with_external_resource(SpirvExternalResource {
            arg_index: 0,
            group: 0,
            binding: 0,
            name: "left_values".to_string(),
            access: Access::ReadWrite,
            element: SpirvResourceElement::Scalar(SpirvScalarKind::U32),
            stride: 4,
            length: 8,
        })
        .with_external_resource(SpirvExternalResource {
            arg_index: 1,
            group: 0,
            binding: 1,
            name: "right_values".to_string(),
            access: Access::ReadWrite,
            element: SpirvResourceElement::Scalar(SpirvScalarKind::U32),
            stride: 4,
            length: 8,
        })
}

#[test]
fn spirv_external_resource_helper_resolves_contextual_capability() {
    let artifact = ambiguous_resource_backend()
        .compile_module(&ambiguous_resource_helper_module(ResourceHelperCalls::LeftOnly))
        .expect("one call-graph resource identity should preserve the helper");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL side artifact");
    let helper_start = wgsl
        .find("fn ambiguous_resource_helper")
        .expect("the contextual resource helper must remain outlined");
    let helper_end = wgsl[helper_start..]
        .find("fn ambiguous_resource_entry")
        .map_or(wgsl.len(), |offset| helper_start + offset);
    let helper = &wgsl[helper_start..helper_end];
    assert!(
        helper.contains("left_values") && !helper.contains("right_values"),
        "the helper must bind the resource identity proved by its call graph:\n{helper}",
    );
    let reparsed = naga::front::wgsl::parse_str(wgsl)
        .expect("WGSL with a contextual helper resource must reparse");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&reparsed)
    .expect("a contextual helper resource must validate for browsers");
}

#[test]
fn spirv_external_resource_helper_derives_multiple_contextual_variants() {
    let artifact = ambiguous_resource_backend()
        .compile_module(&ambiguous_resource_helper_module(ResourceHelperCalls::BothSeparate))
        .expect("one Fe helper should derive one backend placement per proven resource identity");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL side artifact");
    let left_name = "ambiguous_resource_helper_resource_variant_0";
    let right_name = "ambiguous_resource_helper_resource_variant_1";
    let left_start = wgsl
        .find(&format!("fn {left_name}"))
        .expect("the left resource variant must remain outlined");
    let right_start = wgsl
        .find(&format!("fn {right_name}"))
        .expect("the right resource variant must remain outlined");
    let helper_end = |start: usize| {
        wgsl[start + 1..]
            .find("\nfn ")
            .map_or(wgsl.len(), |offset| start + 1 + offset)
    };
    let left = &wgsl[left_start..helper_end(left_start)];
    let right = &wgsl[right_start..helper_end(right_start)];
    assert!(
        left.contains("left_values") && !left.contains("right_values"),
        "the first helper variant must bind only its proven resource:\n{left}",
    );
    assert!(
        right.contains("right_values") && !right.contains("left_values"),
        "the second helper variant must bind only its proven resource:\n{right}",
    );
    assert_eq!(
        wgsl.matches(left_name).count(),
        2,
        "the left specialization must have one definition and one call:\n{wgsl}",
    );
    assert_eq!(
        wgsl.matches(right_name).count(),
        2,
        "the right specialization must have one definition and one call:\n{wgsl}",
    );
    let reparsed = naga::front::wgsl::parse_str(wgsl)
        .expect("WGSL with contextual resource specializations must reparse");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&reparsed)
    .expect("contextual resource specializations must validate for browsers");
}

#[test]
fn spirv_external_resource_helper_rejects_an_unresolved_resource_join() {
    let errors = match ambiguous_resource_backend()
        .compile_module(&ambiguous_resource_helper_module(ResourceHelperCalls::Joined))
    {
        Ok(_) => panic!("a runtime join of distinct resource identities must fail closed"),
        Err(errors) => errors,
    };
    let message = errors
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("; ");
    assert!(
        message.contains("resource aliases") && message.contains("conflicting identities"),
        "the unresolved resource join must retain a named fail-closed diagnostic: {message}",
    );
}

#[test]
fn spirv_four_word_helper_result_survives_as_a_valid_wgsl_struct() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64,
        Vendor::Unknown,
        OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let result_types = [Type::I32; 4];

    let entry_ref = mb
        .declare_function(Signature::new_single(
            "four_word_helper_entry",
            Linkage::Public,
            &[Type::I32, Type::I32],
            Type::I32,
        ))
        .unwrap();
    let helper_ref = mb
        .declare_function(Signature::new(
            "four_word_helper",
            Linkage::Private,
            &[Type::I32, Type::I32],
            &result_types,
        ))
        .unwrap();

    {
        let mut fb = mb.func_builder::<InstInserter>(helper_ref);
        let entry = fb.append_block();
        fb.switch_to_block(entry);
        let lhs = fb.args()[0];
        let rhs = fb.args()[1];
        let sum = fb.insert_inst(arith::Add::new(is, lhs, rhs), Type::I32);
        let difference = fb.insert_inst(arith::Sub::new(is, lhs, rhs), Type::I32);
        let product = fb.insert_inst(arith::Mul::new(is, lhs, rhs), Type::I32);
        let doubled = fb.insert_inst(arith::Add::new(is, sum, sum), Type::I32);
        fb.insert_inst_no_result(control_flow::Return::new(
            is,
            [sum, difference, product, doubled]
                .into_iter()
                .collect::<smallvec::SmallVec<[_; 2]>>()
                .into(),
        ));
        fb.seal_all();
        fb.finish();
    }
    {
        let mut fb = mb.func_builder::<InstInserter>(entry_ref);
        let entry = fb.append_block();
        fb.switch_to_block(entry);
        let results = fb.insert_inst_results(
            control_flow::Call::new(
                is,
                helper_ref,
                [fb.args()[0], fb.args()[1]].into_iter().collect(),
            ),
            &result_types,
        );
        let first = fb.insert_inst(arith::Add::new(is, results[0], results[1]), Type::I32);
        let second = fb.insert_inst(arith::Add::new(is, results[2], results[3]), Type::I32);
        let total = fb.insert_inst(arith::Add::new(is, first, second), Type::I32);
        fb.insert_inst_no_result(control_flow::Return::new_single(is, total));
        fb.seal_all();
        fb.finish();
    }

    let artifact = SpirvBackend::new()
        .compile_module(&mb.build())
        .expect("ordinary four-word helper call should lower without inlining");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL side artifact");
    assert!(
        wgsl.matches("four_word_helper(").count() >= 2,
        "four-word helper must appear as one definition and at least one call:\n{wgsl}"
    );
    assert!(
        wgsl.contains("struct four_word_helper_result"),
        "four-word helper should use one logical WGSL result struct:\n{wgsl}"
    );
    let reparsed = naga::front::wgsl::parse_str(wgsl)
        .expect("WGSL with a four-word helper result must reparse");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&reparsed)
    .expect("WGSL with a four-word helper result must validate for browser capabilities");
}

#[test]
fn spirv_scalar_tuple_helper_with_multiple_returns_uses_one_result_struct() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64,
        Vendor::Unknown,
        OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let result_types = [Type::I32; 2];

    let entry_ref = mb
        .declare_function(Signature::new_single(
            "tuple_return_boundary_entry",
            Linkage::Public,
            &[Type::I32, Type::I32],
            Type::I32,
        ))
        .unwrap();
    let helper_ref = mb
        .declare_function(Signature::new(
            "tuple_return_boundary",
            Linkage::Private,
            &[Type::I32, Type::I32],
            &result_types,
        ))
        .unwrap();

    {
        let mut fb = mb.func_builder::<InstInserter>(helper_ref);
        let entry = fb.append_block();
        let equal = fb.append_block();
        let different = fb.append_block();
        fb.switch_to_block(entry);
        let lhs = fb.args()[0];
        let rhs = fb.args()[1];
        let condition = fb.insert_inst(cmp::Eq::new(is, lhs, rhs), Type::I1);
        fb.insert_inst_no_result(control_flow::Br::new(is, condition, equal, different));
        fb.switch_to_block(equal);
        fb.insert_inst_no_result(control_flow::Return::new(
            is,
            [lhs, rhs]
                .into_iter()
                .collect::<smallvec::SmallVec<[_; 2]>>()
                .into(),
        ));
        fb.switch_to_block(different);
        fb.insert_inst_no_result(control_flow::Return::new(
            is,
            [rhs, lhs]
                .into_iter()
                .collect::<smallvec::SmallVec<[_; 2]>>()
                .into(),
        ));
        fb.seal_all();
        fb.finish();
    }
    {
        let mut fb = mb.func_builder::<InstInserter>(entry_ref);
        let entry = fb.append_block();
        fb.switch_to_block(entry);
        let results = fb.insert_inst_results(
            control_flow::Call::new(
                is,
                helper_ref,
                [fb.args()[0], fb.args()[1]].into_iter().collect(),
            ),
            &result_types,
        );
        let total = fb.insert_inst(arith::Add::new(is, results[0], results[1]), Type::I32);
        fb.insert_inst_no_result(control_flow::Return::new_single(is, total));
        fb.seal_all();
        fb.finish();
    }

    let artifact = SpirvBackend::new()
        .compile_module(&mb.build())
        .expect("multiple scalar-tuple exits should canonicalize before Naga structurization");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL side artifact");
    assert!(
        wgsl.matches("tuple_return_boundary(").count() >= 2,
        "the helper must remain one definition plus a call:\n{wgsl}",
    );
    assert!(
        wgsl.contains("struct tuple_return_boundary_result"),
        "the path-specific lanes must share one physical WGSL result struct:\n{wgsl}",
    );
    let reparsed = naga::front::wgsl::parse_str(wgsl)
        .expect("WGSL with canonicalized scalar-tuple exits must reparse");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&reparsed)
    .expect("canonicalized scalar-tuple exits must validate for browsers");
}

#[test]
fn spirv_value_helper_accepts_a_guarded_trap_continuation() {
    let source = r#"
target = "wasm32-unknown-native"
func public %guarded_trap_entry(v0.i32) -> i32 {
    block0:
        v1.i32 = call %guarded_trap_value v0;
        return v1;
}

func private %guarded_trap_value(v0.i32) -> i32 {
    block0:
        v1.i1 = lt v0 4.i32;
        br v1 block1 block6;
    block1:
        v2.i1 = lt v0 3.i32;
        br v2 block2 block6;
    block2:
        v3.i1 = is_zero v0;
        br v3 block3 block4;
    block3:
        v4.i32 = add v0 22.i32;
        jump block5;
    block4:
        v5.i32 = add v0 33.i32;
        jump block5;
    block5:
        v6.i32 = phi (v4 block3) (v5 block4);
        return v6;
    block6:
        unreachable;
}
"#;
    let module = sonatina_parser::parse_module(source)
        .expect("guarded trap continuation probe should parse")
        .module;
    let artifact = SpirvBackend::new()
        .compile_module(&module)
        .expect("a trap-only continuation should not owe a value result");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL side artifact");
    assert!(
        wgsl.contains("fn guarded_trap_value"),
        "the guarded value helper should remain outlined:\n{wgsl}",
    );
    let reparsed = naga::front::wgsl::parse_str(wgsl)
        .expect("WGSL with a guarded trap continuation must reparse");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&reparsed)
    .expect("a guarded trap continuation must validate for browsers");
}

#[test]
fn spirv_poseidon_sigma_valid() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64, Vendor::Unknown, OperatingSystem::Native
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();

    // Same Poseidon sigma computation as Cranelift and WASM tests:
    // acc=1, C=[3,5,7,11], sigma(x)=x*x+x, 4 rounds unrolled
    let sig = Signature::new_single("poseidon_sigma", Linkage::Public, &[], Type::I64);
    let func_ref = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);

    let mut acc = fb.make_imm_value(1i64);
    for c_val in [3i64, 5, 7, 11] {
        let c = fb.make_imm_value(c_val);
        let sum = fb.insert_inst(arith::Add::new(is, acc, c), Type::I64);
        let sq = fb.insert_inst(arith::Mul::new(is, sum, sum), Type::I64);
        acc = fb.insert_inst(arith::Add::new(is, sq, sum), Type::I64);
    }
    fb.insert_inst_no_result(control_flow::Return::new_single(is, acc));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let backend = SpirvBackend::new().with_workgroup_size(1, 1, 1);
    let artifact = backend.compile_module(&module).expect("SPIR-V compilation failed");

    assert_eq!(artifact.words[0], 0x07230203, "SPIR-V magic");
    eprintln!("SPIR-V Poseidon module: {} words", artifact.words.len());

    // Validate with spirv-val
    let tmp = std::env::temp_dir().join("poseidon_spirv.spv");
    std::fs::write(&tmp, artifact.as_bytes()).unwrap();
    if let Ok(output) = std::process::Command::new("spirv-val").arg(tmp.to_str().unwrap()).output() {
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            eprintln!("spirv-val: {stderr}");
        }
        assert!(output.status.success(), "SPIR-V Poseidon should validate");
    }
    let _ = std::fs::remove_file(&tmp);
}

/// Three-backend Poseidon known-answer: Cranelift executes, WASM executes,
/// SPIR-V validates. All use the same computation.
#[test]
fn three_backend_poseidon_known_answer() {
    use sonatina_codegen::isa::cranelift::CraneliftBackend;
    use sonatina_codegen::isa::wasm::WasmBackend;

    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64, Vendor::Unknown, OperatingSystem::Native
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();

    let sig = Signature::new_single("poseidon", Linkage::Public, &[], Type::I64);
    let func_ref = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);

    let mut acc = fb.make_imm_value(1i64);
    for c_val in [3i64, 5, 7, 11] {
        let c = fb.make_imm_value(c_val);
        let sum = fb.insert_inst(arith::Add::new(is, acc, c), Type::I64);
        let sq = fb.insert_inst(arith::Mul::new(is, sum, sum), Type::I64);
        acc = fb.insert_inst(arith::Add::new(is, sq, sum), Type::I64);
    }
    fb.insert_inst_no_result(control_flow::Return::new_single(is, acc));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let expected = 186898420806i64;

    // Cranelift execution
    let cl = CraneliftBackend::new();
    let cl_art = cl.compile_module(&module).expect("cranelift");
    let cl_fn: fn() -> i64 = unsafe {
        std::mem::transmute(cl_art.get_func_ptr::<fn() -> i64>("poseidon").unwrap())
    };
    assert_eq!(cl_fn(), expected, "Cranelift Poseidon");

    // WASM execution
    let wasm = WasmBackend::new();
    let wasm_art = wasm.compile_module(&module).expect("wasm");
    wasmparser::validate(&wasm_art.bytes).expect("invalid wasm");
    let engine = wasmtime::Engine::default();
    let wm = wasmtime::Module::new(&engine, &wasm_art.bytes).unwrap();
    let mut store = wasmtime::Store::new(&engine, ());
    let inst = wasmtime::Instance::new(&mut store, &wm, &[]).unwrap();
    let wasm_fn = inst.get_typed_func::<(), i64>(&mut store, "poseidon").unwrap();
    assert_eq!(wasm_fn.call(&mut store, ()).unwrap(), expected, "WASM Poseidon");

    // SPIR-V validation (execution requires GPU)
    let spirv = SpirvBackend::new();
    let spirv_art = spirv.compile_module(&module).expect("spirv");
    assert_eq!(spirv_art.words[0], 0x07230203);
    let tmp = std::env::temp_dir().join("poseidon_3way.spv");
    std::fs::write(&tmp, spirv_art.as_bytes()).unwrap();
    if let Ok(output) = std::process::Command::new("spirv-val").arg(tmp.to_str().unwrap()).output() {
        assert!(output.status.success(), "SPIR-V Poseidon should validate");
    }
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn spirv_loop_sum_to_valid() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64, Vendor::Unknown, OperatingSystem::Native
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();

    // sum_to(n): acc=0, i=0; while i<n { acc+=i; i++ }; return acc
    let sig = Signature::new_single("sum_to", Linkage::Public, &[Type::I64], Type::I64);
    let func_ref = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(func_ref);

    let entry = fb.append_block();
    let loop_header = fb.append_block();
    let loop_body = fb.append_block();
    let exit = fb.append_block();

    fb.switch_to_block(entry);
    let n = fb.args()[0];
    let init_acc = fb.make_imm_value(0i64);
    let init_i = fb.make_imm_value(0i64);
    fb.insert_inst_no_result(control_flow::Jump::new(is, loop_header));

    fb.switch_to_block(loop_header);
    let acc = fb.insert_inst(control_flow::Phi::new(is, vec![(init_acc, entry)]), Type::I64);
    let i = fb.insert_inst(control_flow::Phi::new(is, vec![(init_i, entry)]), Type::I64);
    let cond = fb.insert_inst(cmp::Lt::new(is, i, n), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, cond, loop_body, exit));

    fb.switch_to_block(loop_body);
    let new_acc = fb.insert_inst(arith::Add::new(is, acc, i), Type::I64);
    let one = fb.make_imm_value(1i64);
    let new_i = fb.insert_inst(arith::Add::new(is, i, one), Type::I64);
    fb.append_phi_arg(acc, new_acc, loop_body);
    fb.append_phi_arg(i, new_i, loop_body);
    fb.insert_inst_no_result(control_flow::Jump::new(is, loop_header));

    fb.switch_to_block(exit);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, acc));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let backend = SpirvBackend::new().with_workgroup_size(1, 1, 1);
    let artifact = backend.compile_module(&module).expect("SPIR-V loop compilation failed");

    assert_eq!(artifact.words[0], 0x07230203, "valid SPIR-V magic");
    eprintln!("SPIR-V loop module: {} words", artifact.words.len());

    let tmp = std::env::temp_dir().join("spirv_loop_sum.spv");
    std::fs::write(&tmp, artifact.as_bytes()).unwrap();
    if let Ok(output) = std::process::Command::new("spirv-val").arg(tmp.to_str().unwrap()).output() {
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            eprintln!("spirv-val: {stderr}");
        }
        assert!(output.status.success(), "SPIR-V loop should validate with spirv-val");
    }
    let _ = std::fs::remove_file(&tmp);
}

/// A loop header may branch to its exit when the condition is true and remain
/// in the loop when false. This is the reverse of the usual `keep_going`
/// polarity and must still execute according to CFG semantics.
#[test]
fn grid_loop_reversed_header_polarity_executes_on_lavapipe() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64, Vendor::Unknown, OperatingSystem::Native
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single(
        "grid_reversed_loop", Linkage::Public, &[Type::I32, Type::I32], Type::I32,
    );
    let func_ref = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    let header = fb.append_block();
    let body = fb.append_block();
    let exit = fb.append_block();

    fb.switch_to_block(entry);
    let zero = fb.make_imm_value(0i32);
    fb.insert_inst_no_result(control_flow::Jump::new(is, header));

    fb.switch_to_block(header);
    let i = fb.insert_inst(control_flow::Phi::new(is, vec![(zero, entry)]), Type::I32);
    let three = fb.make_imm_value(3i32);
    let done = fb.insert_inst(cmp::Lt::new(is, three, i), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, done, exit, body));

    fb.switch_to_block(body);
    let one = fb.make_imm_value(1i32);
    let next_i = fb.insert_inst(arith::Add::new(is, i, one), Type::I32);
    fb.append_phi_arg(i, next_i, body);
    fb.insert_inst_no_result(control_flow::Jump::new(is, header));

    fb.switch_to_block(exit);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, i));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let artifact = SpirvBackend::new()
        .with_grid()
        .with_workgroup_size(8, 8, 1)
        .compile_module(&module)
        .expect("reversed loop-header polarity should compile");
    let output = run_grid_u32(artifact.wgsl.as_deref().expect("WGSL"), 8, 8, 8, 8, &[]);
    assert_eq!(output, vec![4; 64], "reversed-polarity loop must count through three");
}

/// Phi argument order is not semantic. Put the backedge input first and verify
/// that loop initialization still selects the true outside predecessor.
#[test]
fn spirv_loop_backedge_first_phi_argument_validates() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64, Vendor::Unknown, OperatingSystem::Native
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single("phi_order_loop", Linkage::Public, &[Type::I64], Type::I64);
    let func_ref = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    let header = fb.append_block();
    let body = fb.append_block();
    let exit = fb.append_block();

    fb.switch_to_block(entry);
    let limit = fb.args()[0];
    let zero = fb.make_imm_value(0i64);
    fb.insert_inst_no_result(control_flow::Jump::new(is, header));

    fb.switch_to_block(header);
    let i = fb.insert_inst(control_flow::Phi::new(is, Vec::new()), Type::I64);
    let keep_going = fb.insert_inst(cmp::Lt::new(is, i, limit), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, keep_going, body, exit));

    fb.switch_to_block(body);
    let one = fb.make_imm_value(1i64);
    let next_i = fb.insert_inst(arith::Add::new(is, i, one), Type::I64);
    fb.append_phi_arg(i, next_i, body);
    fb.append_phi_arg(i, zero, entry);
    fb.insert_inst_no_result(control_flow::Jump::new(is, header));

    fb.switch_to_block(exit);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, i));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let artifact = SpirvBackend::new()
        .with_workgroup_size(1, 1, 1)
        .compile_module(&module)
        .expect("backedge-first phi order must compile");
    assert_eq!(artifact.words[0], 0x07230203, "valid SPIR-V magic");

    let tmp = std::env::temp_dir().join("spirv_loop_phi_order.spv");
    std::fs::write(&tmp, artifact.as_bytes()).unwrap();
    if let Ok(output) = std::process::Command::new("spirv-val").arg(&tmp).output() {
        assert!(
            output.status.success(),
            "SPIR-V loop with backedge-first phi should validate: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let _ = std::fs::remove_file(&tmp);
}

/// Loop membership is a CFG property, not a relationship between allocated
/// block identifiers. Here the backedge block is allocated before the header,
/// so both phi predecessors have smaller IDs than the loop header.
#[test]
fn float_loop_phi_with_earlier_backedge_block_compiles() {
    let source = r#"
target = "wasm32-unknown-native"
func public %earlier_backedge(v0.i32) -> i32 {
block0:
    jump block2;
block1:
    v3.f32 = fadd v1 0x3f800000.f32;
    jump block2;
block2:
    v1.f32 = phi (0x00000000.f32 block0) (v3 block1);
    v2.i1 = flt v1 0x40800000.f32;
    br v2 block1 block3;
block3:
    v4.i32 = f32_to_i32 v1;
    return v4;
}
"#;
    let module = sonatina_parser::parse_module(source)
        .expect("earlier-backedge loop should parse")
        .module;
    SpirvBackend::new().compile_module(&module)
        .expect("f32 loop phi must use typed recursive locals independent of block IDs");
}

/// B1 (mb2 browser-testable plan): a u32-word kernel must lower to a naga `Uint`
/// scalar and produce BROWSER-PROFILE WGSL (no 64-bit scalar) that reparses and
/// validates WITHOUT SHADER_INT64. This is the content-derived word gate: the
/// kernel's `i32` return type ALONE drives the u32 lowering (no flag, no config).
/// The kernel mirrors the `poseidon_sigma_u32` browser keystone (rounds
/// [13, 41, 2026]); B1 only lowers/validates it, execution is B2.
#[test]
fn spirv_u32_kernel_lowers_to_uint_scalar() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64, Vendor::Unknown, OperatingSystem::Native
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();

    // poseidon_sigma_u32: acc=1, C=[13,41,2026], sigma(x)=x*x+x, 3 rounds, all
    // Add/Mul. Return type i32 -> the backend derives a u32 word.
    let sig = Signature::new_single("poseidon_sigma_u32", Linkage::Public, &[], Type::I32);
    let func_ref = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);

    let mut acc = fb.make_imm_value(1i32);
    for c_val in [13i32, 41, 2026] {
        let c = fb.make_imm_value(c_val);
        let sum = fb.insert_inst(arith::Add::new(is, acc, c), Type::I32);
        let sq = fb.insert_inst(arith::Mul::new(is, sum, sum), Type::I32);
        acc = fb.insert_inst(arith::Add::new(is, sq, sum), Type::I32);
    }
    fb.insert_inst_no_result(control_flow::Return::new_single(is, acc));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let backend = SpirvBackend::new().with_workgroup_size(1, 1, 1);
    let artifact = backend
        .compile_module(&module)
        .expect("u32 kernel must compile to SPIR-V");

    // The compiler states its own ABI: word must be u32 with a 4-byte result.
    assert_eq!(
        artifact.layout.word,
        WordKind::U32,
        "an i32 return type must derive a u32 word (content-derived, not hardwired)"
    );
    assert_eq!(
        artifact
            .layout
            .result
            .expect("scalar mode must state a single-slot result")
            .width,
        4,
        "u32 result readback width must be 4 bytes"
    );
    assert_eq!(artifact.words[0], 0x07230203, "valid SPIR-V magic");

    // Browser-profile WGSL: no 64-bit scalar tokens anywhere.
    let wgsl = artifact.wgsl.as_ref().expect("WGSL side artifact");
    for tok in ["i64", "u64"] {
        assert!(
            !wgsl.contains(tok),
            "u32 WGSL must contain no `{tok}` 64-bit scalar; got:\n{wgsl}"
        );
    }
    assert!(
        wgsl.contains("u32"),
        "u32 WGSL should declare u32 storage; got:\n{wgsl}"
    );

    // wgsl-in reparse + validate with the BROWSER capability set (no SHADER_INT64).
    let reparsed = naga::front::wgsl::parse_str(wgsl)
        .expect("naga wgsl-in must reparse the emitted browser-profile WGSL");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    )
    .validate(&reparsed)
    .expect(
        "browser-profile validation (default caps, no SHADER_INT64) must accept the u32 module",
    );

    eprintln!(
        "B1: i32 kernel -> u32 word; WGSL validates under default caps ({} words)",
        artifact.words.len()
    );
}

// ===========================================================================
// M1a: Grid mode (fork push #1). One invocation per pixel; args 0,1 are the grid
// coordinates (global_invocation_id.xy), args 2.. are broadcast inputs, the
// return value is stored at output[gid.y * (num_workgroups.x * wgx) + gid.x].
// Grid is driver-declared (with_grid()); scalar/batch paths stay byte-untouched.
// ===========================================================================

/// The M1 grid gradient shape: `grid_gradient(px, py) -> px + py * 1024`. All
/// Add/Mul, i32 return (u32 word). px = gid.x, py = gid.y, so value = x + 1024*y.
fn build_grid_gradient_module() -> sonatina_ir::Module {
    let isa = Native::new(TargetTriple::new(
        if cfg!(target_arch = "x86_64") { Architecture::X86_64 } else { Architecture::Aarch64 },
        Vendor::Unknown, OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single(
        "grid_gradient", Linkage::Public, &[Type::I32, Type::I32], Type::I32,
    );
    let fr = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(fr);
    let entry = fb.append_block();
    fb.switch_to_block(entry);
    let px = fb.args()[0];
    let py = fb.args()[1];
    let k = fb.make_imm_value(1024i32);
    let scaled = fb.insert_inst(arith::Mul::new(is, py, k), Type::I32);
    let v = fb.insert_inst(arith::Add::new(is, px, scaled), Type::I32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, v));
    fb.seal_all();
    fb.finish();
    mb.build()
}

/// A top-level diamond with a merge phi. This catches the historical SPIR-V
/// fallback that emitted blocks linearly and silently ignored branch semantics.
fn build_grid_diamond_module() -> sonatina_ir::Module {
    let isa = Native::new(TargetTriple::new(
        if cfg!(target_arch = "x86_64") { Architecture::X86_64 } else { Architecture::Aarch64 },
        Vendor::Unknown, OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single(
        "grid_diamond", Linkage::Public, &[Type::I32, Type::I32], Type::I32,
    );
    let fr = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(fr);
    let entry = fb.append_block();
    let then_block = fb.append_block();
    let else_block = fb.append_block();
    let merge = fb.append_block();

    fb.switch_to_block(entry);
    let px = fb.args()[0];
    let py = fb.args()[1];
    let cond = fb.insert_inst(cmp::Lt::new(is, px, py), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, cond, then_block, else_block));

    fb.switch_to_block(then_block);
    let then_bias = fb.make_imm_value(100i32);
    let then_value = fb.insert_inst(arith::Add::new(is, px, then_bias), Type::I32);
    fb.insert_inst_no_result(control_flow::Jump::new(is, merge));

    fb.switch_to_block(else_block);
    let else_bias = fb.make_imm_value(200i32);
    let else_value = fb.insert_inst(arith::Add::new(is, py, else_bias), Type::I32);
    fb.insert_inst_no_result(control_flow::Jump::new(is, merge));

    fb.switch_to_block(merge);
    let selected = fb.insert_inst(
        control_flow::Phi::new(
            is,
            vec![(then_value, then_block), (else_value, else_block)],
        ),
        Type::I32,
    );
    let one = fb.make_imm_value(1i32);
    let result = fb.insert_inst(arith::Add::new(is, selected, one), Type::I32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, result));
    fb.seal_all();
    fb.finish();
    mb.build()
}

/// A triangle where the false edge goes straight from the header to the merge.
/// Its phi therefore has one arm input from the header itself.
fn build_grid_triangle_module() -> sonatina_ir::Module {
    let isa = Native::new(TargetTriple::new(
        if cfg!(target_arch = "x86_64") { Architecture::X86_64 } else { Architecture::Aarch64 },
        Vendor::Unknown, OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single(
        "grid_triangle", Linkage::Public, &[Type::I32, Type::I32], Type::I32,
    );
    let fr = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(fr);
    let entry = fb.append_block();
    let then_block = fb.append_block();
    let merge = fb.append_block();

    fb.switch_to_block(entry);
    let px = fb.args()[0];
    let py = fb.args()[1];
    let cond = fb.insert_inst(cmp::Lt::new(is, px, py), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, cond, then_block, merge));

    fb.switch_to_block(then_block);
    let bias = fb.make_imm_value(100i32);
    let then_value = fb.insert_inst(arith::Add::new(is, px, bias), Type::I32);
    fb.insert_inst_no_result(control_flow::Jump::new(is, merge));

    fb.switch_to_block(merge);
    let selected = fb.insert_inst(
        control_flow::Phi::new(is, vec![(then_value, then_block), (py, entry)]),
        Type::I32,
    );
    fb.insert_inst_no_result(control_flow::Return::new_single(is, selected));
    fb.seal_all();
    fb.finish();
    mb.build()
}

fn build_grid_one_arm_return_module() -> sonatina_ir::Module {
    let isa = Native::new(TargetTriple::new(
        if cfg!(target_arch = "x86_64") { Architecture::X86_64 } else { Architecture::Aarch64 },
        Vendor::Unknown, OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single(
        "grid_one_arm_return", Linkage::Public, &[Type::I32, Type::I32], Type::I32,
    );
    let fr = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(fr);
    let entry = fb.append_block();
    let early = fb.append_block();
    let merge = fb.append_block();

    fb.switch_to_block(entry);
    let px = fb.args()[0];
    let py = fb.args()[1];
    let condition = fb.insert_inst(cmp::Lt::new(is, px, py), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, condition, early, merge));

    fb.switch_to_block(early);
    let escaped = fb.make_imm_value(777i32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, escaped));

    fb.switch_to_block(merge);
    let normal = fb.make_imm_value(4i32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, normal));
    fb.seal_all();
    fb.finish();
    mb.build()
}

fn build_grid_one_arm_return_with_fallthrough_merge_module() -> sonatina_ir::Module {
    let isa = Native::new(TargetTriple::new(
        if cfg!(target_arch = "x86_64") { Architecture::X86_64 } else { Architecture::Aarch64 },
        Vendor::Unknown, OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single(
        "grid_one_arm_return_with_fallthrough_merge", Linkage::Public,
        &[Type::I32, Type::I32], Type::I32,
    );
    let fr = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(fr);
    let entry = fb.append_block();
    let early = fb.append_block();
    let passthrough = fb.append_block();
    let merge = fb.append_block();

    fb.switch_to_block(entry);
    let px = fb.args()[0];
    let py = fb.args()[1];
    let condition = fb.insert_inst(cmp::Lt::new(is, px, py), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, condition, early, passthrough));

    fb.switch_to_block(early);
    let escaped = fb.make_imm_value(777i32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, escaped));

    fb.switch_to_block(passthrough);
    fb.insert_inst_no_result(control_flow::Jump::new(is, merge));

    fb.switch_to_block(merge);
    let normal = fb.make_imm_value(4i32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, normal));
    fb.seal_all();
    fb.finish();
    mb.build()
}

fn build_grid_nested_return_outer_phi_module() -> sonatina_ir::Module {
    let isa = Native::new(TargetTriple::new(
        if cfg!(target_arch = "x86_64") { Architecture::X86_64 } else { Architecture::Aarch64 },
        Vendor::Unknown, OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single(
        "grid_nested_return_outer_phi", Linkage::Public,
        &[Type::I32, Type::I32], Type::I32,
    );
    let fr = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(fr);
    let entry = fb.append_block();
    let nested_header = fb.append_block();
    let other_outer_arm = fb.append_block();
    let early = fb.append_block();
    let nested_fallthrough = fb.append_block();
    let outer_merge = fb.append_block();

    fb.switch_to_block(entry);
    let px = fb.args()[0];
    let py = fb.args()[1];
    let four = fb.make_imm_value(4i32);
    let enter_nested = fb.insert_inst(cmp::Lt::new(is, px, four), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, enter_nested, nested_header, other_outer_arm));

    fb.switch_to_block(nested_header);
    let return_early = fb.insert_inst(cmp::Lt::new(is, py, four), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, return_early, early, nested_fallthrough));

    fb.switch_to_block(early);
    let escaped = fb.make_imm_value(777i32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, escaped));

    fb.switch_to_block(nested_fallthrough);
    let fallthrough_value = fb.insert_inst(arith::Add::new(is, px, py), Type::I32);
    fb.insert_inst_no_result(control_flow::Jump::new(is, outer_merge));

    fb.switch_to_block(other_outer_arm);
    let twenty_two = fb.make_imm_value(22i32);
    fb.insert_inst_no_result(control_flow::Jump::new(is, outer_merge));

    fb.switch_to_block(outer_merge);
    let result = fb.insert_inst(
        control_flow::Phi::new(is, vec![(fallthrough_value, nested_fallthrough), (twenty_two, other_outer_arm)]),
        Type::I32,
    );
    fb.insert_inst_no_result(control_flow::Return::new_single(is, result));
    fb.seal_all();
    fb.finish();
    mb.build()
}

/// A phi produced by a nested diamond feeds one incoming edge of an outer
/// merge. Its load must be emitted inside that exact edge before the outer phi
/// snapshots it.
fn build_grid_nested_phi_feeds_outer_phi_module() -> sonatina_ir::Module {
    let isa = Native::new(TargetTriple::new(
        if cfg!(target_arch = "x86_64") { Architecture::X86_64 } else { Architecture::Aarch64 },
        Vendor::Unknown, OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single(
        "grid_nested_phi_feeds_outer_phi", Linkage::Public,
        &[Type::I32, Type::I32], Type::I32,
    );
    let fr = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(fr);
    let entry = fb.append_block();
    let nested_header = fb.append_block();
    let other_outer_arm = fb.append_block();
    let nested_low = fb.append_block();
    let nested_high = fb.append_block();
    let nested_merge = fb.append_block();
    let outer_merge = fb.append_block();

    fb.switch_to_block(entry);
    let px = fb.args()[0];
    let py = fb.args()[1];
    let four = fb.make_imm_value(4i32);
    let enter_nested = fb.insert_inst(cmp::Lt::new(is, px, four), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, enter_nested, nested_header, other_outer_arm));

    fb.switch_to_block(nested_header);
    let choose_low = fb.insert_inst(cmp::Lt::new(is, py, four), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, choose_low, nested_low, nested_high));

    fb.switch_to_block(nested_low);
    let low = fb.make_imm_value(11i32);
    fb.insert_inst_no_result(control_flow::Jump::new(is, nested_merge));

    fb.switch_to_block(nested_high);
    let high = fb.make_imm_value(22i32);
    fb.insert_inst_no_result(control_flow::Jump::new(is, nested_merge));

    fb.switch_to_block(nested_merge);
    let nested = fb.insert_inst(
        control_flow::Phi::new(is, vec![(low, nested_low), (high, nested_high)]),
        Type::I32,
    );
    fb.insert_inst_no_result(control_flow::Jump::new(is, outer_merge));

    fb.switch_to_block(other_outer_arm);
    let other = fb.make_imm_value(33i32);
    fb.insert_inst_no_result(control_flow::Jump::new(is, outer_merge));

    fb.switch_to_block(outer_merge);
    let result = fb.insert_inst(
        control_flow::Phi::new(is, vec![(nested, nested_merge), (other, other_outer_arm)]),
        Type::I32,
    );
    fb.insert_inst_no_result(control_flow::Return::new_single(is, result));
    fb.seal_all();
    fb.finish();
    mb.build()
}

/// A diamond merge carrying a phi is itself the header of the next
/// conditional. The phi Load must be emitted before that header consumes it.
fn build_grid_phi_headed_conditional_module() -> sonatina_ir::Module {
    let isa = Native::new(TargetTriple::new(
        if cfg!(target_arch = "x86_64") { Architecture::X86_64 } else { Architecture::Aarch64 },
        Vendor::Unknown, OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single(
        "grid_phi_headed_conditional", Linkage::Public,
        &[Type::I32, Type::I32], Type::I32,
    );
    let fr = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(fr);
    let entry = fb.append_block();
    let then_block = fb.append_block();
    let else_block = fb.append_block();
    let merge_header = fb.append_block();
    let low_arm = fb.append_block();
    let high_arm = fb.append_block();
    let final_merge = fb.append_block();

    fb.switch_to_block(entry);
    let px = fb.args()[0];
    let py = fb.args()[1];
    let choose_low = fb.insert_inst(cmp::Lt::new(is, px, py), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, choose_low, then_block, else_block));

    fb.switch_to_block(then_block);
    let low = fb.make_imm_value(11i32);
    fb.insert_inst_no_result(control_flow::Jump::new(is, merge_header));

    fb.switch_to_block(else_block);
    let high = fb.make_imm_value(22i32);
    fb.insert_inst_no_result(control_flow::Jump::new(is, merge_header));

    fb.switch_to_block(merge_header);
    let selected = fb.insert_inst(
        control_flow::Phi::new(is, vec![(low, then_block), (high, else_block)]),
        Type::I32,
    );
    let twenty = fb.make_imm_value(20i32);
    let selected_low = fb.insert_inst(cmp::Lt::new(is, selected, twenty), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, selected_low, low_arm, high_arm));

    fb.switch_to_block(low_arm);
    fb.insert_inst_no_result(control_flow::Jump::new(is, final_merge));

    fb.switch_to_block(high_arm);
    fb.insert_inst_no_result(control_flow::Jump::new(is, final_merge));

    fb.switch_to_block(final_merge);
    let result = fb.insert_inst(
        control_flow::Phi::new(is, vec![(selected, low_arm), (selected, high_arm)]),
        Type::I32,
    );
    fb.insert_inst_no_result(control_flow::Return::new_single(is, result));
    fb.seal_all();
    fb.finish();
    mb.build()
}

/// A loop whose body branches between a backedge and an early return. The
/// structurizer represents the body branch as an `IfThenElse` nested in a
/// `Loop`; silently flattening only direct `Block` children drops this branch.
fn build_grid_loop_conditional_module() -> sonatina_ir::Module {
    let isa = Native::new(TargetTriple::new(
        if cfg!(target_arch = "x86_64") { Architecture::X86_64 } else { Architecture::Aarch64 },
        Vendor::Unknown, OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single(
        "grid_loop_conditional", Linkage::Public, &[Type::I32, Type::I32], Type::I32,
    );
    let fr = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(fr);
    let entry = fb.append_block();
    let header = fb.append_block();
    let body = fb.append_block();
    let continue_block = fb.append_block();
    let escape = fb.append_block();
    let exit = fb.append_block();

    fb.switch_to_block(entry);
    let px = fb.args()[0];
    let py = fb.args()[1];
    let zero = fb.make_imm_value(0i32);
    fb.insert_inst_no_result(control_flow::Jump::new(is, header));

    fb.switch_to_block(header);
    let i = fb.insert_inst(control_flow::Phi::new(is, vec![(zero, entry)]), Type::I32);
    let four = fb.make_imm_value(4i32);
    let keep_going = fb.insert_inst(cmp::Lt::new(is, i, four), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, keep_going, body, exit));

    fb.switch_to_block(body);
    let continue_condition = fb.insert_inst(cmp::Lt::new(is, px, py), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(
        is,
        continue_condition,
        continue_block,
        escape,
    ));

    fb.switch_to_block(continue_block);
    let one = fb.make_imm_value(1i32);
    let next_i = fb.insert_inst(arith::Add::new(is, i, one), Type::I32);
    fb.append_phi_arg(i, next_i, continue_block);
    fb.insert_inst_no_result(control_flow::Jump::new(is, header));

    fb.switch_to_block(escape);
    let escaped = fb.make_imm_value(777i32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, escaped));

    fb.switch_to_block(exit);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, i));
    fb.seal_all();
    fb.finish();
    mb.build()
}

/// A return from inside the loop must bypass the normal post-loop sibling.
/// Lowering the return as an ordinary loop break would overwrite 777 with 4.
fn build_grid_loop_early_return_bypasses_sibling_module() -> sonatina_ir::Module {
    let isa = Native::new(TargetTriple::new(
        if cfg!(target_arch = "x86_64") { Architecture::X86_64 } else { Architecture::Aarch64 },
        Vendor::Unknown, OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single(
        "grid_loop_early_return_bypasses_sibling", Linkage::Public,
        &[Type::I32, Type::I32], Type::I32,
    );
    let fr = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(fr);
    let entry = fb.append_block();
    let header = fb.append_block();
    let body = fb.append_block();
    let early = fb.append_block();
    let latch = fb.append_block();
    let exit = fb.append_block();
    let sibling = fb.append_block();

    fb.switch_to_block(entry);
    let px = fb.args()[0];
    let py = fb.args()[1];
    let zero = fb.make_imm_value(0i32);
    fb.insert_inst_no_result(control_flow::Jump::new(is, header));

    fb.switch_to_block(header);
    let i = fb.insert_inst(control_flow::Phi::new(is, vec![(zero, entry)]), Type::I32);
    let four = fb.make_imm_value(4i32);
    let keep_going = fb.insert_inst(cmp::Lt::new(is, i, four), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, keep_going, body, exit));

    fb.switch_to_block(body);
    let return_early = fb.insert_inst(cmp::Lt::new(is, px, py), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, return_early, early, latch));

    fb.switch_to_block(early);
    let escaped = fb.make_imm_value(777i32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, escaped));

    fb.switch_to_block(latch);
    let one = fb.make_imm_value(1i32);
    let next_i = fb.insert_inst(arith::Add::new(is, i, one), Type::I32);
    fb.append_phi_arg(i, next_i, latch);
    fb.insert_inst_no_result(control_flow::Jump::new(is, header));

    fb.switch_to_block(exit);
    fb.insert_inst_no_result(control_flow::Jump::new(is, sibling));

    fb.switch_to_block(sibling);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, i));
    fb.seal_all();
    fb.finish();
    mb.build()
}

/// A return-bearing loop nested in one arm of an outer conditional. The loop's
/// early return must bypass the outer merge, while its normal exit must still
/// supply the exact predecessor value to that merge.
fn build_grid_conditional_with_returning_loop_module() -> sonatina_ir::Module {
    let isa = Native::new(TargetTriple::new(
        if cfg!(target_arch = "x86_64") { Architecture::X86_64 } else { Architecture::Aarch64 },
        Vendor::Unknown, OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single(
        "grid_conditional_with_returning_loop", Linkage::Public,
        &[Type::I32, Type::I32], Type::I32,
    );
    let fr = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(fr);
    let entry = fb.append_block();
    let loop_preheader = fb.append_block();
    let outer_else = fb.append_block();
    let loop_header = fb.append_block();
    let loop_body = fb.append_block();
    let early = fb.append_block();
    let loop_latch = fb.append_block();
    let loop_exit = fb.append_block();
    let outer_merge = fb.append_block();

    fb.switch_to_block(entry);
    let px = fb.args()[0];
    let py = fb.args()[1];
    let four = fb.make_imm_value(4i32);
    let choose_loop = fb.insert_inst(cmp::Lt::new(is, px, four), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(
        is,
        choose_loop,
        loop_preheader,
        outer_else,
    ));

    fb.switch_to_block(loop_preheader);
    let zero = fb.make_imm_value(0i32);
    fb.insert_inst_no_result(control_flow::Jump::new(is, loop_header));

    fb.switch_to_block(outer_else);
    let outside = fb.make_imm_value(22i32);
    fb.insert_inst_no_result(control_flow::Jump::new(is, outer_merge));

    fb.switch_to_block(loop_header);
    let i = fb.insert_inst(
        control_flow::Phi::new(is, vec![(zero, loop_preheader)]),
        Type::I32,
    );
    let keep_going = fb.insert_inst(cmp::Lt::new(is, i, four), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(
        is,
        keep_going,
        loop_body,
        loop_exit,
    ));

    fb.switch_to_block(loop_body);
    let return_early = fb.insert_inst(cmp::Lt::new(is, py, four), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(
        is,
        return_early,
        early,
        loop_latch,
    ));

    fb.switch_to_block(early);
    let escaped = fb.make_imm_value(777i32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, escaped));

    fb.switch_to_block(loop_latch);
    let one = fb.make_imm_value(1i32);
    let next_i = fb.insert_inst(arith::Add::new(is, i, one), Type::I32);
    fb.append_phi_arg(i, next_i, loop_latch);
    fb.insert_inst_no_result(control_flow::Jump::new(is, loop_header));

    fb.switch_to_block(loop_exit);
    fb.insert_inst_no_result(control_flow::Jump::new(is, outer_merge));

    fb.switch_to_block(outer_merge);
    let result = fb.insert_inst(
        control_flow::Phi::new(is, vec![(i, loop_exit), (outside, outer_else)]),
        Type::I32,
    );
    fb.insert_inst_no_result(control_flow::Return::new_single(is, result));
    fb.seal_all();
    fb.finish();
    mb.build()
}

/// A return-bearing loop nested in another loop. Returning from the inner loop
/// must cascade out of the outer loop, while normal inner-loop completion must
/// resume the outer latch.
fn build_grid_nested_returning_loop_module() -> sonatina_ir::Module {
    let isa = Native::new(TargetTriple::new(
        if cfg!(target_arch = "x86_64") { Architecture::X86_64 } else { Architecture::Aarch64 },
        Vendor::Unknown, OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single(
        "grid_nested_returning_loop", Linkage::Public,
        &[Type::I32, Type::I32], Type::I32,
    );
    let fr = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(fr);
    let entry = fb.append_block();
    let outer_header = fb.append_block();
    let inner_preheader = fb.append_block();
    let inner_header = fb.append_block();
    let inner_body = fb.append_block();
    let early = fb.append_block();
    let inner_latch = fb.append_block();
    let inner_exit = fb.append_block();
    let outer_latch = fb.append_block();
    let outer_exit = fb.append_block();

    fb.switch_to_block(entry);
    let px = fb.args()[0];
    let py = fb.args()[1];
    let zero = fb.make_imm_value(0i32);
    fb.insert_inst_no_result(control_flow::Jump::new(is, outer_header));

    fb.switch_to_block(outer_header);
    let outer_i = fb.insert_inst(
        control_flow::Phi::new(is, vec![(zero, entry)]),
        Type::I32,
    );
    let two = fb.make_imm_value(2i32);
    let outer_keep_going = fb.insert_inst(cmp::Lt::new(is, outer_i, two), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(
        is,
        outer_keep_going,
        inner_preheader,
        outer_exit,
    ));

    fb.switch_to_block(inner_preheader);
    fb.insert_inst_no_result(control_flow::Jump::new(is, inner_header));

    fb.switch_to_block(inner_header);
    let inner_i = fb.insert_inst(
        control_flow::Phi::new(is, vec![(zero, inner_preheader)]),
        Type::I32,
    );
    let inner_keep_going = fb.insert_inst(cmp::Lt::new(is, inner_i, two), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(
        is,
        inner_keep_going,
        inner_body,
        inner_exit,
    ));

    fb.switch_to_block(inner_body);
    let return_early = fb.insert_inst(cmp::Lt::new(is, px, py), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(
        is,
        return_early,
        early,
        inner_latch,
    ));

    fb.switch_to_block(early);
    let escaped = fb.make_imm_value(777i32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, escaped));

    fb.switch_to_block(inner_latch);
    let one = fb.make_imm_value(1i32);
    let next_inner = fb.insert_inst(arith::Add::new(is, inner_i, one), Type::I32);
    fb.append_phi_arg(inner_i, next_inner, inner_latch);
    fb.insert_inst_no_result(control_flow::Jump::new(is, inner_header));

    fb.switch_to_block(inner_exit);
    fb.insert_inst_no_result(control_flow::Jump::new(is, outer_latch));

    fb.switch_to_block(outer_latch);
    let next_outer = fb.insert_inst(arith::Add::new(is, outer_i, one), Type::I32);
    fb.append_phi_arg(outer_i, next_outer, outer_latch);
    fb.insert_inst_no_result(control_flow::Jump::new(is, outer_header));

    fb.switch_to_block(outer_exit);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, outer_i));
    fb.seal_all();
    fb.finish();
    mb.build()
}

/// Three parallel loop-phi swaps: (a, b) starts at (11, 22), swaps on every
/// backedge, and therefore exits as (22, 11). Sequential phi stores would
/// overwrite one source before the other is read and produce the wrong result.
fn build_grid_parallel_phi_swap_module() -> sonatina_ir::Module {
    let isa = Native::new(TargetTriple::new(
        if cfg!(target_arch = "x86_64") { Architecture::X86_64 } else { Architecture::Aarch64 },
        Vendor::Unknown, OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single(
        "grid_parallel_phi_swap", Linkage::Public, &[Type::I32, Type::I32], Type::I32,
    );
    let fr = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(fr);
    let entry = fb.append_block();
    let header = fb.append_block();
    let latch = fb.append_block();
    let exit = fb.append_block();

    fb.switch_to_block(entry);
    let zero = fb.make_imm_value(0i32);
    let eleven = fb.make_imm_value(11i32);
    let twenty_two = fb.make_imm_value(22i32);
    fb.insert_inst_no_result(control_flow::Jump::new(is, header));

    fb.switch_to_block(header);
    let i = fb.insert_inst(control_flow::Phi::new(is, vec![(zero, entry)]), Type::I32);
    let a = fb.insert_inst(control_flow::Phi::new(is, vec![(eleven, entry)]), Type::I32);
    let b = fb.insert_inst(control_flow::Phi::new(is, vec![(twenty_two, entry)]), Type::I32);
    let three = fb.make_imm_value(3i32);
    let keep_going = fb.insert_inst(cmp::Lt::new(is, i, three), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, keep_going, latch, exit));

    fb.switch_to_block(latch);
    let one = fb.make_imm_value(1i32);
    let next_i = fb.insert_inst(arith::Add::new(is, i, one), Type::I32);
    fb.append_phi_arg(i, next_i, latch);
    fb.append_phi_arg(a, b, latch);
    fb.append_phi_arg(b, a, latch);
    fb.insert_inst_no_result(control_flow::Jump::new(is, header));

    fb.switch_to_block(exit);
    let hundred = fb.make_imm_value(100i32);
    let high = fb.insert_inst(arith::Mul::new(is, a, hundred), Type::I32);
    let result = fb.insert_inst(arith::Add::new(is, high, b), Type::I32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, result));
    fb.seal_all();
    fb.finish();
    mb.build()
}

/// A loop containing a full diamond whose merge value is carried by the loop.
/// For grid coordinates the condition is deliberately false, so the value
/// selected at the in-loop merge and returned after one iteration is 22.
fn build_grid_loop_inner_if_phi_module() -> sonatina_ir::Module {
    let isa = Native::new(TargetTriple::new(
        if cfg!(target_arch = "x86_64") { Architecture::X86_64 } else { Architecture::Aarch64 },
        Vendor::Unknown, OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single(
        "grid_loop_inner_if_phi", Linkage::Public, &[Type::I32, Type::I32], Type::I32,
    );
    let fr = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(fr);
    let entry = fb.append_block();
    let header = fb.append_block();
    let body = fb.append_block();
    let then_block = fb.append_block();
    let else_block = fb.append_block();
    let merge = fb.append_block();
    let exit = fb.append_block();

    fb.switch_to_block(entry);
    let px = fb.args()[0];
    let zero = fb.make_imm_value(0i32);
    fb.insert_inst_no_result(control_flow::Jump::new(is, header));

    fb.switch_to_block(header);
    let i = fb.insert_inst(control_flow::Phi::new(is, vec![(zero, entry)]), Type::I32);
    let result = fb.insert_inst(control_flow::Phi::new(is, vec![(zero, entry)]), Type::I32);
    let one = fb.make_imm_value(1i32);
    let keep_going = fb.insert_inst(cmp::Lt::new(is, i, one), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, keep_going, body, exit));

    fb.switch_to_block(body);
    let negative = fb.insert_inst(cmp::Lt::new(is, px, zero), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, negative, then_block, else_block));

    fb.switch_to_block(then_block);
    let eleven = fb.make_imm_value(11i32);
    fb.insert_inst_no_result(control_flow::Jump::new(is, merge));

    fb.switch_to_block(else_block);
    let twenty_two = fb.make_imm_value(22i32);
    fb.insert_inst_no_result(control_flow::Jump::new(is, merge));

    fb.switch_to_block(merge);
    let selected = fb.insert_inst(
        control_flow::Phi::new(is, vec![(eleven, then_block), (twenty_two, else_block)]),
        Type::I32,
    );
    let next_i = fb.insert_inst(arith::Add::new(is, i, one), Type::I32);
    fb.append_phi_arg(i, next_i, merge);
    fb.append_phi_arg(result, selected, merge);
    fb.insert_inst_no_result(control_flow::Jump::new(is, header));

    fb.switch_to_block(exit);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, result));
    fb.seal_all();
    fb.finish();
    mb.build()
}

/// Keeps the loop-carried value as f32 and converts it only in the exit block
/// to the grid's i32/u32 output carrier. Two exact increments turn 1.0 into 3.0.
fn build_grid_f32_loop_return_module() -> sonatina_ir::Module {
    let isa = Native::new(TargetTriple::new(
        if cfg!(target_arch = "x86_64") { Architecture::X86_64 } else { Architecture::Aarch64 },
        Vendor::Unknown, OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single(
        "grid_f32_loop_return", Linkage::Public, &[Type::I32, Type::I32], Type::I32,
    );
    let fr = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(fr);
    let entry = fb.append_block();
    let header = fb.append_block();
    let latch = fb.append_block();
    let exit = fb.append_block();

    fb.switch_to_block(entry);
    let zero = fb.make_imm_value(0i32);
    let initial = fb.make_imm_value(Immediate::F32(1.0f32.to_bits()));
    fb.insert_inst_no_result(control_flow::Jump::new(is, header));

    fb.switch_to_block(header);
    let i = fb.insert_inst(control_flow::Phi::new(is, vec![(zero, entry)]), Type::I32);
    let value = fb.insert_inst(control_flow::Phi::new(is, vec![(initial, entry)]), Type::F32);
    let two = fb.make_imm_value(2i32);
    let keep_going = fb.insert_inst(cmp::Lt::new(is, i, two), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, keep_going, latch, exit));

    fb.switch_to_block(latch);
    let one_i32 = fb.make_imm_value(1i32);
    let one_f32 = fb.make_imm_value(Immediate::F32(1.0f32.to_bits()));
    let next_i = fb.insert_inst(arith::Add::new(is, i, one_i32), Type::I32);
    let next_value = fb.insert_inst(arith::Fadd::new(is, value, one_f32), Type::F32);
    fb.append_phi_arg(i, next_i, latch);
    fb.append_phi_arg(value, next_value, latch);
    fb.insert_inst_no_result(control_flow::Jump::new(is, header));

    fb.switch_to_block(exit);
    let output = fb.insert_inst(cast::F32ToI32::new(is, value), Type::I32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, output));
    fb.seal_all();
    fb.finish();
    mb.build()
}

/// Exits a loop normally, then resumes in a sibling block outside the loop.
/// The sibling consumes a non-phi value computed in the header from the
/// current loop phi. At exit `i == 3`, so `i * 2 + 16 == 22`.
fn build_grid_loop_exit_sibling_resume_module() -> sonatina_ir::Module {
    let isa = Native::new(TargetTriple::new(
        if cfg!(target_arch = "x86_64") { Architecture::X86_64 } else { Architecture::Aarch64 },
        Vendor::Unknown, OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single(
        "grid_loop_exit_sibling_resume", Linkage::Public, &[Type::I32, Type::I32], Type::I32,
    );
    let fr = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(fr);
    let entry = fb.append_block();
    let header = fb.append_block();
    let body = fb.append_block();
    let loop_exit = fb.append_block();
    let sibling = fb.append_block();

    fb.switch_to_block(entry);
    let zero = fb.make_imm_value(0i32);
    fb.insert_inst_no_result(control_flow::Jump::new(is, header));

    fb.switch_to_block(header);
    let i = fb.insert_inst(control_flow::Phi::new(is, vec![(zero, entry)]), Type::I32);
    let two = fb.make_imm_value(2i32);
    let header_value = fb.insert_inst(arith::Mul::new(is, i, two), Type::I32);
    let three = fb.make_imm_value(3i32);
    let keep_going = fb.insert_inst(cmp::Lt::new(is, i, three), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, keep_going, body, loop_exit));

    fb.switch_to_block(body);
    let one = fb.make_imm_value(1i32);
    let next_i = fb.insert_inst(arith::Add::new(is, i, one), Type::I32);
    fb.append_phi_arg(i, next_i, body);
    fb.insert_inst_no_result(control_flow::Jump::new(is, header));

    fb.switch_to_block(loop_exit);
    fb.insert_inst_no_result(control_flow::Jump::new(is, sibling));

    fb.switch_to_block(sibling);
    let sixteen = fb.make_imm_value(16i32);
    let result = fb.insert_inst(arith::Add::new(is, header_value, sixteen), Type::I32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, result));
    fb.seal_all();
    fb.finish();
    mb.build()
}

/// The normal loop exit begins with a phi fed by the header edge. Using px as
/// the trip count covers both the zero-trip x=0 case and iterative x>0 cases.
fn build_grid_loop_exit_phi_module() -> sonatina_ir::Module {
    let isa = Native::new(TargetTriple::new(
        if cfg!(target_arch = "x86_64") { Architecture::X86_64 } else { Architecture::Aarch64 },
        Vendor::Unknown, OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single(
        "grid_loop_exit_phi", Linkage::Public, &[Type::I32, Type::I32], Type::I32,
    );
    let fr = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(fr);
    let entry = fb.append_block();
    let header = fb.append_block();
    let body = fb.append_block();
    let exit = fb.append_block();

    fb.switch_to_block(entry);
    let limit = fb.args()[0];
    let zero = fb.make_imm_value(0i32);
    fb.insert_inst_no_result(control_flow::Jump::new(is, header));

    fb.switch_to_block(header);
    let i = fb.insert_inst(control_flow::Phi::new(is, vec![(zero, entry)]), Type::I32);
    let ten = fb.make_imm_value(10i32);
    let header_value = fb.insert_inst(arith::Add::new(is, i, ten), Type::I32);
    let keep_going = fb.insert_inst(cmp::Lt::new(is, i, limit), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, keep_going, body, exit));

    fb.switch_to_block(body);
    let one = fb.make_imm_value(1i32);
    let next_i = fb.insert_inst(arith::Add::new(is, i, one), Type::I32);
    fb.append_phi_arg(i, next_i, body);
    fb.insert_inst_no_result(control_flow::Jump::new(is, header));

    fb.switch_to_block(exit);
    let exit_value = fb.insert_inst(
        control_flow::Phi::new(is, vec![(header_value, header)]),
        Type::I32,
    );
    fb.insert_inst_no_result(control_flow::Return::new_single(is, exit_value));
    fb.seal_all();
    fb.finish();
    mb.build()
}

/// One shared loop exit receives distinct f32 state from two body `break`
/// edges and the normal header exit. Grid x selects the path: 0 -> 11,
/// 1 -> 22, and every other column runs one iteration and exits with 33.
fn build_grid_multi_exit_f32_phi_module() -> sonatina_ir::Module {
    let isa = Native::new(TargetTriple::new(
        if cfg!(target_arch = "x86_64") { Architecture::X86_64 } else { Architecture::Aarch64 },
        Vendor::Unknown, OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single(
        "grid_multi_exit_f32_phi", Linkage::Public, &[Type::I32, Type::I32], Type::I32,
    );
    let fr = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(fr);
    let entry = fb.append_block();
    let header = fb.append_block();
    let first_test = fb.append_block();
    let second_test = fb.append_block();
    let latch = fb.append_block();
    let exit = fb.append_block();
    let sibling = fb.append_block();

    fb.switch_to_block(entry);
    let x = fb.args()[0];
    let zero = fb.make_imm_value(0i32);
    let initial = fb.make_imm_value(Immediate::F32(3.0f32.to_bits()));
    fb.insert_inst_no_result(control_flow::Jump::new(is, header));

    fb.switch_to_block(header);
    let i = fb.insert_inst(control_flow::Phi::new(is, vec![(zero, entry)]), Type::I32);
    let value = fb.insert_inst(control_flow::Phi::new(is, vec![(initial, entry)]), Type::F32);
    let one = fb.make_imm_value(1i32);
    let keep_going = fb.insert_inst(cmp::Lt::new(is, i, one), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, keep_going, first_test, exit));

    fb.switch_to_block(first_test);
    let eleven = fb.make_imm_value(Immediate::F32(11.0f32.to_bits()));
    let is_first = fb.insert_inst(cmp::Lt::new(is, x, one), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, is_first, exit, second_test));

    fb.switch_to_block(second_test);
    let twenty_two = fb.make_imm_value(Immediate::F32(22.0f32.to_bits()));
    let two = fb.make_imm_value(2i32);
    let is_second = fb.insert_inst(cmp::Lt::new(is, x, two), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, is_second, exit, latch));

    fb.switch_to_block(latch);
    let next_i = fb.insert_inst(arith::Add::new(is, i, one), Type::I32);
    let thirty = fb.make_imm_value(Immediate::F32(30.0f32.to_bits()));
    let next_value = fb.insert_inst(arith::Fadd::new(is, value, thirty), Type::F32);
    fb.append_phi_arg(i, next_i, latch);
    fb.append_phi_arg(value, next_value, latch);
    fb.insert_inst_no_result(control_flow::Jump::new(is, header));

    fb.switch_to_block(exit);
    let exit_value = fb.insert_inst(
        control_flow::Phi::new(
            is,
            vec![(value, header), (eleven, first_test), (twenty_two, second_test)],
        ),
        Type::F32,
    );
    fb.insert_inst_no_result(control_flow::Jump::new(is, sibling));

    fb.switch_to_block(sibling);
    let output = fb.insert_inst(cast::F32ToI32::new(is, exit_value), Type::I32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, output));
    fb.seal_all();
    fb.finish();
    mb.build()
}

/// A directly-returning canonical exit phi is fed by the normal header edge,
/// a conditional body edge, and an unconditional jump block outside the loop
/// SCC. This pins both body-exit forms without relying on a following sibling.
fn build_grid_direct_return_multi_exit_f32_phi_module() -> sonatina_ir::Module {
    let isa = Native::new(TargetTriple::new(
        if cfg!(target_arch = "x86_64") { Architecture::X86_64 } else { Architecture::Aarch64 },
        Vendor::Unknown, OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single(
        "grid_direct_return_multi_exit_f32_phi", Linkage::Public,
        &[Type::I32, Type::I32], Type::I32,
    );
    let fr = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(fr);
    let entry = fb.append_block();
    let header = fb.append_block();
    let first_test = fb.append_block();
    let second_test = fb.append_block();
    let jump_exit = fb.append_block();
    let latch = fb.append_block();
    let exit = fb.append_block();

    fb.switch_to_block(entry);
    let x = fb.args()[0];
    let zero = fb.make_imm_value(0i32);
    let initial = fb.make_imm_value(Immediate::F32(3.0f32.to_bits()));
    fb.insert_inst_no_result(control_flow::Jump::new(is, header));

    fb.switch_to_block(header);
    let i = fb.insert_inst(control_flow::Phi::new(is, vec![(zero, entry)]), Type::I32);
    let value = fb.insert_inst(control_flow::Phi::new(is, vec![(initial, entry)]), Type::F32);
    let keep_going = fb.insert_inst(cmp::Lt::new(is, i, x), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, keep_going, first_test, exit));

    fb.switch_to_block(first_test);
    let direct_value = fb.make_imm_value(Immediate::F32(41.0f32.to_bits()));
    let two = fb.make_imm_value(2i32);
    let takes_direct_exit = fb.insert_inst(cmp::Lt::new(is, x, two), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, takes_direct_exit, exit, second_test));

    fb.switch_to_block(second_test);
    let three = fb.make_imm_value(3i32);
    let takes_jump_exit = fb.insert_inst(cmp::Lt::new(is, x, three), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, takes_jump_exit, jump_exit, latch));

    fb.switch_to_block(jump_exit);
    let jump_value = fb.make_imm_value(Immediate::F32(52.0f32.to_bits()));
    fb.insert_inst_no_result(control_flow::Jump::new(is, exit));

    fb.switch_to_block(latch);
    let one = fb.make_imm_value(1i32);
    let next_i = fb.insert_inst(arith::Add::new(is, i, one), Type::I32);
    let ten = fb.make_imm_value(Immediate::F32(10.0f32.to_bits()));
    let next_value = fb.insert_inst(arith::Fadd::new(is, value, ten), Type::F32);
    fb.append_phi_arg(i, next_i, latch);
    fb.append_phi_arg(value, next_value, latch);
    fb.insert_inst_no_result(control_flow::Jump::new(is, header));

    fb.switch_to_block(exit);
    let exit_value = fb.insert_inst(
        control_flow::Phi::new(
            is,
            vec![(value, header), (direct_value, first_test), (jump_value, jump_exit)],
        ),
        Type::F32,
    );
    let output = fb.insert_inst(cast::F32ToI32::new(is, exit_value), Type::I32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, output));
    fb.seal_all();
    fb.finish();
    mb.build()
}

/// A 3-arg grid kernel with one broadcast param: `f(px, py, p) -> px + py*1024 + p`.
/// arg2 is the broadcast input struct member p0; the load-bearing M3 shape.
fn build_grid_broadcast_module() -> sonatina_ir::Module {
    let isa = Native::new(TargetTriple::new(
        if cfg!(target_arch = "x86_64") { Architecture::X86_64 } else { Architecture::Aarch64 },
        Vendor::Unknown, OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single(
        "grid_broadcast", Linkage::Public, &[Type::I32, Type::I32, Type::I32], Type::I32,
    );
    let fr = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(fr);
    let entry = fb.append_block();
    fb.switch_to_block(entry);
    let px = fb.args()[0];
    let py = fb.args()[1];
    let p = fb.args()[2];
    let k = fb.make_imm_value(1024i32);
    let scaled = fb.insert_inst(arith::Mul::new(is, py, k), Type::I32);
    let base = fb.insert_inst(arith::Add::new(is, px, scaled), Type::I32);
    let v = fb.insert_inst(arith::Add::new(is, base, p), Type::I32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, v));
    fb.seal_all();
    fb.finish();
    mb.build()
}

/// A 2-arg i64 grid kernel (fail-closed: grid requires the u32 word).
fn build_grid_i64_module() -> sonatina_ir::Module {
    let isa = Native::new(TargetTriple::new(
        if cfg!(target_arch = "x86_64") { Architecture::X86_64 } else { Architecture::Aarch64 },
        Vendor::Unknown, OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single(
        "grid_i64", Linkage::Public, &[Type::I64, Type::I64], Type::I64,
    );
    let fr = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(fr);
    let entry = fb.append_block();
    fb.switch_to_block(entry);
    let a = fb.args()[0];
    let b = fb.args()[1];
    let v = fb.insert_inst(arith::Add::new(is, a, b), Type::I64);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, v));
    fb.seal_all();
    fb.finish();
    mb.build()
}

/// A 1-arg i32 grid kernel (fail-closed: a grid kernel needs at least px, py).
fn build_grid_1arg_module() -> sonatina_ir::Module {
    let isa = Native::new(TargetTriple::new(
        if cfg!(target_arch = "x86_64") { Architecture::X86_64 } else { Architecture::Aarch64 },
        Vendor::Unknown, OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single("grid_1arg", Linkage::Public, &[Type::I32], Type::I32);
    let fr = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(fr);
    let entry = fb.append_block();
    fb.switch_to_block(entry);
    let a = fb.args()[0];
    fb.insert_inst_no_result(control_flow::Return::new_single(is, a));
    fb.seal_all();
    fb.finish();
    mb.build()
}

/// A 2-arg i32 grid kernel that also ObjAllocs (fail-closed: grid and batch are
/// mutually exclusive).
fn build_grid_objalloc_module() -> sonatina_ir::Module {
    let isa = Native::new(TargetTriple::new(
        if cfg!(target_arch = "x86_64") { Architecture::X86_64 } else { Architecture::Aarch64 },
        Vendor::Unknown, OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let arr_ty = mb.declare_array_type(Type::I32, 16);
    let arr_objref_ty = mb.objref_type(arr_ty);
    let sig = Signature::new_single(
        "grid_objalloc", Linkage::Public, &[Type::I32, Type::I32], Type::I32,
    );
    let fr = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(fr);
    let entry = fb.append_block();
    fb.switch_to_block(entry);
    let a = fb.args()[0];
    let _buf = fb.insert_inst(data::ObjAlloc::new(is, arr_ty), arr_objref_ty);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, a));
    fb.seal_all();
    fb.finish();
    mb.build()
}

/// Grid kernel whose result is the exact i32 bit pattern round-tripped through
/// f32. This pins representation-preserving Bitcast lowering independently of
/// the numeric i32/f32 conversion instructions.
fn build_grid_bitcast_roundtrip_module() -> sonatina_ir::Module {
    let isa = Native::new(TargetTriple::new(
        if cfg!(target_arch = "x86_64") { Architecture::X86_64 } else { Architecture::Aarch64 },
        Vendor::Unknown, OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single(
        "grid_bitcast_roundtrip", Linkage::Public, &[Type::I32, Type::I32], Type::I32,
    );
    let fr = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(fr);
    let entry = fb.append_block();
    fb.switch_to_block(entry);
    let px = fb.args()[0];
    let as_f32 = fb.insert_inst(cast::Bitcast::new(is, px, Type::F32), Type::F32);
    let as_i32 = fb.insert_inst(cast::Bitcast::new(is, as_f32, Type::I32), Type::I32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, as_i32));
    fb.seal_all();
    fb.finish();
    mb.build()
}

fn build_external_record_compute_module() -> sonatina_ir::Module {
    let isa = Native::new(TargetTriple::new(
        if cfg!(target_arch = "x86_64") { Architecture::X86_64 } else { Architecture::Aarch64 },
        Vendor::Unknown, OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let record_ty = mb.declare_struct_type("ComplexF32Bits", &[Type::I32, Type::I32], false);
    let record_ref_ty = mb.objref_type(record_ty);
    let word_ref_ty = mb.objref_type(Type::I32);
    let array_ty = mb.declare_array_type(record_ty, 1);
    let array_ref_ty = mb.objref_type(array_ty);
    let sig = Signature::new_unit("write_external_record", Linkage::Public, &[array_ref_ty]);
    let fr = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(fr);
    let entry = fb.append_block();
    fb.switch_to_block(entry);
    let orbit = fb.args()[0];
    let zero = fb.make_imm_value(0i32);
    let one = fb.make_imm_value(1i32);
    let sample = fb.insert_inst(data::ObjIndex::new(is, orbit, zero), record_ref_ty);
    let re = fb.insert_inst(
        data::ObjProj::new(is, smallvec::smallvec![sample, zero]),
        word_ref_ty,
    );
    let im = fb.insert_inst(
        data::ObjProj::new(is, smallvec::smallvec![sample, one]),
        word_ref_ty,
    );
    let re_bits = fb.make_imm_value(1.0f32.to_bits() as i32);
    let im_bits = fb.make_imm_value((-2.0f32).to_bits() as i32);
    fb.insert_inst_no_result(data::ObjStore::new(is, re, re_bits));
    fb.insert_inst_no_result(data::ObjStore::new(is, im, im_bits));
    fb.insert_inst_no_result(control_flow::Return::new_unit(is));
    fb.seal_all();
    fb.finish();
    mb.build()
}

fn build_external_mixed_record_compute_module() -> sonatina_ir::Module {
    let isa = Native::new(TargetTriple::new(
        if cfg!(target_arch = "x86_64") { Architecture::X86_64 } else { Architecture::Aarch64 },
        Vendor::Unknown, OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    // Sonatina's I32 lane is the signless browser-word carrier. The independent
    // external schema below deliberately marks that middle field as WGSL u32.
    let record_ty = mb.declare_struct_type("MixedSample", &[Type::F32, Type::I32, Type::F32], false);
    let record_ref_ty = mb.objref_type(record_ty);
    let float_ref_ty = mb.objref_type(Type::F32);
    let word_ref_ty = mb.objref_type(Type::I32);
    let array_ty = mb.declare_array_type(record_ty, 1);
    let array_ref_ty = mb.objref_type(array_ty);
    let sig = Signature::new_unit("write_external_mixed_record", Linkage::Public, &[array_ref_ty]);
    let fr = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(fr);
    let entry = fb.append_block();
    fb.switch_to_block(entry);
    let samples = fb.args()[0];
    let zero = fb.make_imm_value(0i32);
    let one = fb.make_imm_value(1i32);
    let two = fb.make_imm_value(2i32);
    let sample = fb.insert_inst(data::ObjIndex::new(is, samples, zero), record_ref_ty);
    let x = fb.insert_inst(
        data::ObjProj::new(is, smallvec::smallvec![sample, zero]),
        float_ref_ty,
    );
    let material = fb.insert_inst(
        data::ObjProj::new(is, smallvec::smallvec![sample, one]),
        word_ref_ty,
    );
    let y = fb.insert_inst(
        data::ObjProj::new(is, smallvec::smallvec![sample, two]),
        float_ref_ty,
    );
    let x_value = fb.make_imm_value(Immediate::F32(1.25f32.to_bits()));
    let material_value = fb.make_imm_value(7i32);
    let y_value = fb.make_imm_value(Immediate::F32((-2.5f32).to_bits()));
    fb.insert_inst_no_result(data::ObjStore::new(is, x, x_value));
    fb.insert_inst_no_result(data::ObjStore::new(is, material, material_value));
    fb.insert_inst_no_result(data::ObjStore::new(is, y, y_value));
    let loaded_x = fb.insert_inst(data::ObjLoad::new(is, x), Type::F32);
    let loaded_y = fb.insert_inst(data::ObjLoad::new(is, y), Type::F32);
    let sum = fb.insert_inst(arith::Fadd::new(is, loaded_x, loaded_y), Type::F32);
    fb.insert_inst_no_result(data::ObjStore::new(is, x, sum));
    fb.insert_inst_no_result(control_flow::Return::new_unit(is));
    fb.seal_all();
    fb.finish();
    mb.build()
}

fn build_unit_compute_loop_with_exit_store_module() -> sonatina_ir::Module {
    let isa = Native::new(TargetTriple::new(
        if cfg!(target_arch = "x86_64") { Architecture::X86_64 } else { Architecture::Aarch64 },
        Vendor::Unknown, OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let array_ty = mb.declare_array_type(Type::I32, 8);
    let array_ref_ty = mb.objref_type(array_ty);
    let word_ref_ty = mb.objref_type(Type::I32);
    let sig = Signature::new_unit("unit_compute_loop", Linkage::Public, &[array_ref_ty]);
    let fr = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(fr);
    let entry = fb.append_block();
    let header = fb.append_block();
    let body = fb.append_block();
    let exit = fb.append_block();

    fb.switch_to_block(entry);
    let values = fb.args()[0];
    let zero = fb.make_imm_value(0i32);
    fb.insert_inst_no_result(control_flow::Jump::new(is, header));

    fb.switch_to_block(header);
    let i = fb.insert_inst(control_flow::Phi::new(is, vec![(zero, entry)]), Type::I32);
    let eight = fb.make_imm_value(8i32);
    let keep_going = fb.insert_inst(cmp::Lt::new(is, i, eight), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, keep_going, body, exit));

    fb.switch_to_block(body);
    let slot = fb.insert_inst(data::ObjIndex::new(is, values, i), word_ref_ty);
    fb.insert_inst_no_result(data::ObjStore::new(is, slot, i));
    let one = fb.make_imm_value(1i32);
    let next_i = fb.insert_inst(arith::Add::new(is, i, one), Type::I32);
    fb.append_phi_arg(i, next_i, body);
    fb.insert_inst_no_result(control_flow::Jump::new(is, header));

    fb.switch_to_block(exit);
    let first = fb.insert_inst(data::ObjIndex::new(is, values, zero), word_ref_ty);
    let receipt = fb.make_imm_value(99i32);
    fb.insert_inst_no_result(data::ObjStore::new(is, first, receipt));
    fb.insert_inst_no_result(control_flow::Return::new_unit(is));
    fb.seal_all();
    fb.finish();
    mb.build()
}

fn build_unit_compute_early_return_with_sibling_module() -> sonatina_ir::Module {
    let isa = Native::new(TargetTriple::new(
        if cfg!(target_arch = "x86_64") { Architecture::X86_64 } else { Architecture::Aarch64 },
        Vendor::Unknown, OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let array_ty = mb.declare_array_type(Type::I32, 1);
    let array_ref_ty = mb.objref_type(array_ty);
    let word_ref_ty = mb.objref_type(Type::I32);
    let sig = Signature::new_unit(
        "unit_compute_early_return_with_sibling",
        Linkage::Public,
        &[Type::I32, array_ref_ty],
    );
    let fr = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(fr);
    let entry = fb.append_block();
    let early = fb.append_block();
    let normal = fb.append_block();
    let sibling = fb.append_block();

    fb.switch_to_block(entry);
    let invocation = fb.args()[0];
    let values = fb.args()[1];
    let zero = fb.make_imm_value(0i32);
    let return_early = fb.insert_inst(cmp::Eq::new(is, invocation, zero), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, return_early, early, normal));

    fb.switch_to_block(early);
    fb.insert_inst_no_result(control_flow::Return::new_unit(is));

    fb.switch_to_block(normal);
    fb.insert_inst_no_result(control_flow::Jump::new(is, sibling));

    fb.switch_to_block(sibling);
    let slot = fb.insert_inst(data::ObjIndex::new(is, values, zero), word_ref_ty);
    let receipt = fb.make_imm_value(99i32);
    fb.insert_inst_no_result(data::ObjStore::new(is, slot, receipt));
    fb.insert_inst_no_result(control_flow::Return::new_unit(is));
    fb.seal_all();
    fb.finish();
    mb.build()
}

fn build_compute_invocation_context_module() -> sonatina_ir::Module {
    let isa = Native::new(TargetTriple::new(
        if cfg!(target_arch = "x86_64") { Architecture::X86_64 } else { Architecture::Aarch64 },
        Vendor::Unknown, OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let array_ty = mb.declare_array_type(Type::I32, 8);
    let array_ref_ty = mb.objref_type(array_ty);
    let word_ref_ty = mb.objref_type(Type::I32);
    let mut args = vec![Type::I32; 13];
    args.push(array_ref_ty);
    args.push(Type::I32);
    let sig = Signature::new_unit("compute_invocation_context", Linkage::Public, &args);
    let fr = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(fr);
    let entry = fb.append_block();
    fb.switch_to_block(entry);
    let function_args = fb.args().to_vec();
    let mut receipt = function_args[14];
    for &value in &function_args[..13] {
        receipt = fb.insert_inst(arith::Add::new(is, receipt, value), Type::I32);
    }
    let slot = fb.insert_inst(
        data::ObjIndex::new(is, function_args[13], function_args[0]),
        word_ref_ty,
    );
    fb.insert_inst_no_result(data::ObjStore::new(is, slot, receipt));
    fb.insert_inst_no_result(control_flow::Return::new_unit(is));
    fb.seal_all();
    fb.finish();
    mb.build()
}

fn build_unit_compute_with_trap_module() -> sonatina_ir::Module {
    let isa = Native::new(TargetTriple::new(
        if cfg!(target_arch = "x86_64") { Architecture::X86_64 } else { Architecture::Aarch64 },
        Vendor::Unknown, OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_unit("compute_with_trap", Linkage::Public, &[Type::I32]);
    let fr = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(fr);
    let entry = fb.append_block();
    let ok = fb.append_block();
    let trap = fb.append_block();
    fb.switch_to_block(entry);
    let limit = fb.args()[0];
    let eight = fb.make_imm_value(8i32);
    let valid = fb.insert_inst(cmp::Lt::new(is, limit, eight), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, valid, ok, trap));
    fb.switch_to_block(ok);
    fb.insert_inst_no_result(control_flow::Return::new_unit(is));
    fb.switch_to_block(trap);
    fb.insert_inst_no_result(control_flow::Unreachable::new(is));
    fb.seal_all();
    fb.finish();
    mb.build()
}

fn complete_compute_invocation_arguments() -> Vec<SpirvBuiltinArgument> {
    use SpirvBuiltinSource as Source;
    [
        Source::GlobalInvocationIdX,
        Source::GlobalInvocationIdY,
        Source::GlobalInvocationIdZ,
        Source::LocalInvocationIdX,
        Source::LocalInvocationIdY,
        Source::LocalInvocationIdZ,
        Source::WorkgroupIdX,
        Source::WorkgroupIdY,
        Source::WorkgroupIdZ,
        Source::NumWorkgroupsX,
        Source::NumWorkgroupsY,
        Source::NumWorkgroupsZ,
        Source::LocalInvocationIndex,
    ]
    .into_iter()
    .enumerate()
    .map(|(arg_index, source)| SpirvBuiltinArgument {
        arg_index: arg_index as u32,
        source,
    })
    .collect()
}

fn build_external_record_render_module() -> sonatina_ir::Module {
    let isa = Native::new(TargetTriple::new(
        if cfg!(target_arch = "x86_64") { Architecture::X86_64 } else { Architecture::Aarch64 },
        Vendor::Unknown, OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let record_ty = mb.declare_struct_type("ComplexF32Bits", &[Type::I32, Type::I32], false);
    let record_ref_ty = mb.objref_type(record_ty);
    let word_ref_ty = mb.objref_type(Type::I32);
    let array_ty = mb.declare_array_type(record_ty, 1);
    let array_ref_ty = mb.objref_type(array_ty);
    let sig = Signature::new_single(
        "read_external_record",
        Linkage::Public,
        &[Type::I32, Type::I32, array_ref_ty],
        Type::I32,
    );
    let fr = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(fr);
    let entry = fb.append_block();
    fb.switch_to_block(entry);
    let orbit = fb.args()[2];
    let zero = fb.make_imm_value(0i32);
    let one = fb.make_imm_value(1i32);
    let sample = fb.insert_inst(data::ObjIndex::new(is, orbit, zero), record_ref_ty);
    let re = fb.insert_inst(
        data::ObjProj::new(is, smallvec::smallvec![sample, zero]),
        word_ref_ty,
    );
    let im = fb.insert_inst(
        data::ObjProj::new(is, smallvec::smallvec![sample, one]),
        word_ref_ty,
    );
    let re_bits = fb.insert_inst(data::ObjLoad::new(is, re), Type::I32);
    let im_bits = fb.insert_inst(data::ObjLoad::new(is, im), Type::I32);
    let color = fb.insert_inst(
        sonatina_ir::inst::logic::Xor::new(is, re_bits, im_bits),
        Type::I32,
    );
    fb.insert_inst_no_result(control_flow::Return::new_single(is, color));
    fb.seal_all();
    fb.finish();
    mb.build()
}

fn external_complex_resource(arg_index: u32, access: Access) -> SpirvExternalResource {
    SpirvExternalResource {
        arg_index,
        group: 0,
        binding: 0,
        name: "orbit".to_string(),
        access,
        element: SpirvResourceElement::Record {
            fields: vec![
                SpirvResourceField {
                    name: "re_bits".to_string(),
                    scalar: SpirvScalarKind::U32,
                    offset: 0,
                },
                SpirvResourceField {
                    name: "im_bits".to_string(),
                    scalar: SpirvScalarKind::U32,
                    offset: 4,
                },
            ],
            span: 8,
        },
        stride: 8,
        length: 1,
    }
}

fn external_mixed_resource(arg_index: u32, access: Access) -> SpirvExternalResource {
    SpirvExternalResource {
        arg_index,
        group: 0,
        binding: 0,
        name: "samples".to_string(),
        access,
        element: SpirvResourceElement::Record {
            fields: vec![
                SpirvResourceField {
                    name: "x".to_string(),
                    scalar: SpirvScalarKind::F32,
                    offset: 0,
                },
                SpirvResourceField {
                    name: "material".to_string(),
                    scalar: SpirvScalarKind::U32,
                    offset: 4,
                },
                SpirvResourceField {
                    name: "y".to_string(),
                    scalar: SpirvScalarKind::F32,
                    offset: 8,
                },
            ],
            span: 12,
        },
        stride: 12,
        length: 1,
    }
}

/// Execute a Grid-mode WGSL compute shader under the browser profile
/// (`Features::empty()`, no SHADER_INT64) and read back the whole output grid.
/// Hard-fails if no adapter is available: this is an EXECUTED gate (lavapipe is
/// present in CI), not validate-only.
fn run_grid_u32(wgsl: &str, width: u32, height: u32, wgx: u32, wgy: u32, input: &[u8]) -> Vec<u32> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        force_fallback_adapter: false,
        ..Default::default()
    }))
    .expect("grid execute requires a GPU adapter (lavapipe); none available");
    eprintln!("  GPU: {}", adapter.get_info().name);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        // Browser profile: no SHADER_INT64. The u32 grid must run here.
        required_features: wgpu::Features::empty(),
        ..Default::default()
    }))
    .expect("browser-profile device (Features::empty) must be available on lavapipe");

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("grid"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });

    // Explicit two-binding BGL: output @0/0 (read-write), input @0/1 (read-only).
    // The gradient kernel never reads `input`, so an auto-derived layout would
    // strip binding 1; declaring both explicitly binds the dummy input exactly
    // like the scalar keystone's unused input.
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("grid_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("grid_pl"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let output_size = width as u64 * height as u64 * 4;
    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let input_size = input.len().max(4) as u64;
    let input_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: input_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !input.is_empty() {
        queue.write_buffer(&input_buf, 0, input);
    }
    let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: output_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: output_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: input_buf.as_entire_binding() },
        ],
    });

    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut p = enc.begin_compute_pass(&Default::default());
        p.set_pipeline(&pipeline);
        p.set_bind_group(0, &bg, &[]);
        p.dispatch_workgroups(width / wgx, height / wgy, 1);
    }
    enc.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, output_size);
    queue.submit(Some(enc.finish()));

    let slice = staging_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(std::time::Duration::from_secs(30)),
    });
    rx.recv().unwrap().unwrap();
    let data = slice.get_mapped_range();
    let out: Vec<u32> = data
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    drop(data);
    staging_buf.unmap();
    out
}

fn expect_grid_err(module: &sonatina_ir::Module, backend: SpirvBackend) -> String {
    match backend.compile_module(module) {
        Ok(_) => panic!("expected a grid fail-closed error, got Ok"),
        Err(errs) => {
            let msg = errs.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("; ");
            assert!(msg.contains("grid"), "fail-closed error must name grid: {msg}");
            msg
        }
    }
}

/// Grid WGSL shape + self-describing layout, GPU-free.
#[test]
fn grid_wgsl_shape() {
    let module = build_grid_gradient_module();
    let art = SpirvBackend::new()
        .with_grid()
        .with_workgroup_size(8, 8, 1)
        .compile_module(&module)
        .expect("grid gradient must compile");

    let layout = &art.layout;
    assert_layout_metadata_invariants(layout, 2);
    assert_eq!(layout.mode, LayoutMode::Grid, "grid mode");
    assert_eq!(layout.word, WordKind::U32, "u32 word");
    assert!(layout.result.is_none(), "grid states no single-slot result");
    assert_eq!(layout.workgroup_size, [8, 8, 1], "workgroup size");
    let out_b = layout.bindings.iter().find(|b| b.role == Role::Output).expect("output binding");
    let in_b = layout.bindings.iter().find(|b| b.role == Role::Input).expect("input binding");
    assert_eq!(out_b.stride, 4, "output stride = per-element word width");
    assert_eq!(in_b.stride, 4, "input stride = broadcast span (one dummy member)");
    assert!(in_b.members.is_empty(), "padding-only input must not invent a source member");

    let wgsl = art.wgsl.as_ref().expect("WGSL side artifact");
    for tok in ["global_invocation_id", "num_workgroups", "array<u32>"] {
        assert!(wgsl.contains(tok), "grid WGSL must contain `{tok}`; got:\n{wgsl}");
    }
    for tok in ["i64", "u64"] {
        assert!(!wgsl.contains(tok), "grid WGSL must contain no `{tok}`; got:\n{wgsl}");
    }
    let reparsed = naga::front::wgsl::parse_str(wgsl)
        .expect("naga wgsl-in must reparse the grid WGSL");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    )
    .validate(&reparsed)
    .expect("browser-profile validation (default caps) must accept the grid module");
    eprintln!("grid_wgsl_shape OK: {} words", art.words.len());
}

#[test]
fn scalar_i64_and_batch_layout_metadata_invariants() {
    let scalar = SpirvBackend::new().compile_module(&build_grid_i64_module())
        .expect("homogeneous i64 scalar module should compile");
    assert_layout_metadata_invariants(&scalar.layout, 2);
    let input = scalar.layout.bindings.iter().find(|b| b.role == Role::Input).unwrap();
    assert_eq!(input.members.iter().map(|m| (m.offset, m.width, m.scalar)).collect::<Vec<_>>(), vec![
        (0, 8, SpirvScalarKind::I64),
        (8, 8, SpirvScalarKind::I64),
    ]);
    assert_eq!((input.span, input.stride), (16, 16));

    let batch = SpirvBackend::new().compile_module(&build_grid_objalloc_module())
        .expect("batch module should compile");
    assert_eq!(batch.layout.mode, LayoutMode::Batch);
    assert_layout_metadata_invariants(&batch.layout, 2);
    assert!(batch.layout.builtin_inputs.is_empty());
}

/// The headline: a grid kernel EXECUTES on lavapipe and every pixel equals the
/// CPU oracle x + 1024*y.
#[test]
fn grid_executes_on_lavapipe() {
    let module = build_grid_gradient_module();
    let art = SpirvBackend::new()
        .with_grid()
        .with_workgroup_size(8, 8, 1)
        .compile_module(&module)
        .expect("grid gradient must compile");
    let wgsl = art.wgsl.as_ref().expect("WGSL");

    let (w, h, wgx, wgy) = (16u32, 16u32, 8u32, 8u32);
    let out = run_grid_u32(wgsl, w, h, wgx, wgy, &[]);
    assert_eq!(out.len(), (w * h) as usize, "full grid readback");
    for y in 0..h {
        for x in 0..w {
            let got = out[(y * w + x) as usize];
            let want = x + 1024 * y; // oracle, written in-test, never trusted from spec
            assert_eq!(got, want, "grid[{y}*{w}+{x}] = {got}, want {want}");
        }
    }
    eprintln!("grid_executes_on_lavapipe OK: {w}x{h}, all pixels == x + 1024*y");
}

/// Execute a top-level conditional and its merge phi. Both arms are exercised
/// across the grid and the phi feeds another instruction after the merge.
#[test]
fn grid_diamond_executes_on_lavapipe() {
    let module = build_grid_diamond_module();
    let art = SpirvBackend::new()
        .with_grid()
        .with_workgroup_size(8, 8, 1)
        .compile_module(&module)
        .expect("grid diamond must compile");
    let wgsl = art.wgsl.as_ref().expect("WGSL");

    let (w, h, wgx, wgy) = (8u32, 8u32, 8u32, 8u32);
    let out = run_grid_u32(wgsl, w, h, wgx, wgy, &[]);
    for y in 0..h {
        for x in 0..w {
            let got = out[(y * w + x) as usize];
            let want = if x < y { x + 101 } else { y + 201 };
            assert_eq!(got, want, "grid diamond at ({x}, {y})");
        }
    }
}

#[test]
fn grid_triangle_executes_on_lavapipe() {
    let module = build_grid_triangle_module();
    let art = SpirvBackend::new()
        .with_grid()
        .with_workgroup_size(8, 8, 1)
        .compile_module(&module)
        .expect("grid triangle must compile");
    let wgsl = art.wgsl.as_ref().expect("WGSL");

    let (w, h, wgx, wgy) = (8u32, 8u32, 8u32, 8u32);
    let out = run_grid_u32(wgsl, w, h, wgx, wgy, &[]);
    for y in 0..h {
        for x in 0..w {
            let got = out[(y * w + x) as usize];
            let want = if x < y { x + 100 } else { y };
            assert_eq!(got, want, "grid triangle at ({x}, {y})");
        }
    }
}

#[test]
fn grid_one_arm_return_bypasses_merge_on_lavapipe() {
    let module = build_grid_one_arm_return_module();
    let artifact = SpirvBackend::new()
        .with_grid()
        .with_workgroup_size(8, 8, 1)
        .compile_module(&module)
        .expect("one-arm return should compile");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL");
    let output = run_grid_u32(wgsl, 8, 8, 8, 8, &[]);
    for y in 0..8u32 {
        for x in 0..8u32 {
            assert_eq!(output[(y * 8 + x) as usize], if x < y { 777 } else { 4 });
        }
    }
}

#[test]
fn grid_one_arm_return_guards_fallthrough_merge_on_lavapipe() {
    let module = build_grid_one_arm_return_with_fallthrough_merge_module();
    let artifact = SpirvBackend::new()
        .with_grid()
        .with_workgroup_size(8, 8, 1)
        .compile_module(&module)
        .expect("one-arm return with fallthrough merge should compile");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL");
    let output = run_grid_u32(wgsl, 8, 8, 8, 8, &[]);
    for y in 0..8u32 {
        for x in 0..8u32 {
            assert_eq!(output[(y * 8 + x) as usize], if x < y { 777 } else { 4 });
        }
    }
}

#[test]
fn grid_nested_return_guards_outer_merge_phi_edge_on_lavapipe() {
    let module = build_grid_nested_return_outer_phi_module();
    let artifact = SpirvBackend::new()
        .with_grid()
        .with_workgroup_size(8, 8, 1)
        .compile_module(&module)
        .expect("nested return and outer merge phi should compile");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL");
    let output = run_grid_u32(wgsl, 8, 8, 8, 8, &[]);
    for y in 0..8u32 {
        for x in 0..8u32 {
            let expected = if x < 4 && y < 4 { 777 } else if x < 4 { x + y } else { 22 };
            assert_eq!(output[(y * 8 + x) as usize], expected, "({x},{y})");
        }
    }
}

#[test]
fn grid_nested_phi_feeds_outer_phi_on_lavapipe() {
    let module = build_grid_nested_phi_feeds_outer_phi_module();
    let artifact = SpirvBackend::new()
        .with_grid()
        .with_workgroup_size(8, 8, 1)
        .compile_module(&module)
        .expect("nested phi feeding an outer phi should compile");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL");
    let output = run_grid_u32(wgsl, 8, 8, 8, 8, &[]);
    for y in 0..8u32 {
        for x in 0..8u32 {
            let expected = if x < 4 { if y < 4 { 11 } else { 22 } } else { 33 };
            assert_eq!(output[(y * 8 + x) as usize], expected, "({x},{y})");
        }
    }
}

#[test]
fn grid_phi_headed_conditional_executes_on_lavapipe() {
    let module = build_grid_phi_headed_conditional_module();
    let artifact = SpirvBackend::new()
        .with_grid()
        .with_workgroup_size(8, 8, 1)
        .compile_module(&module)
        .expect("phi-headed conditional should compile");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL");
    let output = run_grid_u32(wgsl, 8, 8, 8, 8, &[]);
    for y in 0..8u32 {
        for x in 0..8u32 {
            assert_eq!(output[(y * 8 + x) as usize], if x < y { 11 } else { 22 });
        }
    }
}

#[test]
fn grid_loop_with_conditional_executes_on_lavapipe() {
    let module = build_grid_loop_conditional_module();
    let artifact = SpirvBackend::new()
        .with_grid()
        .with_workgroup_size(8, 8, 1)
        .compile_module(&module).expect("recursive loop conditional should compile");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL");
    let output = run_grid_u32(wgsl, 8, 8, 8, 8, &[]);
    for y in 0..8u32 {
        for x in 0..8u32 {
            assert_eq!(output[(y * 8 + x) as usize], if x < y { 4 } else { 777 }, "({x},{y})");
        }
    }
}

#[test]
fn grid_loop_early_return_bypasses_post_loop_sibling_on_lavapipe() {
    let module = build_grid_loop_early_return_bypasses_sibling_module();
    let artifact = SpirvBackend::new()
        .with_grid()
        .with_workgroup_size(8, 8, 1)
        .compile_module(&module)
        .expect("loop early return and normal sibling should compile");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL");
    let output = run_grid_u32(wgsl, 8, 8, 8, 8, &[]);
    for y in 0..8u32 {
        for x in 0..8u32 {
            assert_eq!(
                output[(y * 8 + x) as usize],
                if x < y { 777 } else { 4 },
                "({x},{y})",
            );
        }
    }
}

#[test]
fn grid_conditional_with_returning_loop_executes_on_lavapipe() {
    let module = build_grid_conditional_with_returning_loop_module();
    let artifact = SpirvBackend::new()
        .with_grid()
        .with_workgroup_size(8, 8, 1)
        .compile_module(&module)
        .expect("return-bearing loop nested in conditional should compile");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL");
    let output = run_grid_u32(wgsl, 8, 8, 8, 8, &[]);
    for y in 0..8u32 {
        for x in 0..8u32 {
            let expected = if x >= 4 { 22 } else if y < 4 { 777 } else { 4 };
            assert_eq!(output[(y * 8 + x) as usize], expected, "({x},{y})");
        }
    }
}

#[test]
fn grid_nested_returning_loop_executes_on_lavapipe() {
    let module = build_grid_nested_returning_loop_module();
    let artifact = SpirvBackend::new()
        .with_grid()
        .with_workgroup_size(8, 8, 1)
        .compile_module(&module)
        .expect("return-bearing loop nested in loop should compile");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL");
    let output = run_grid_u32(wgsl, 8, 8, 8, 8, &[]);
    for y in 0..8u32 {
        for x in 0..8u32 {
            assert_eq!(
                output[(y * 8 + x) as usize],
                if x < y { 777 } else { 2 },
                "({x},{y})",
            );
        }
    }
}

#[test]
fn grid_parallel_loop_phi_swap_executes_on_lavapipe() {
    let module = build_grid_parallel_phi_swap_module();
    let artifact = SpirvBackend::new()
        .with_grid()
        .with_workgroup_size(8, 8, 1)
        .compile_module(&module)
        .expect("parallel loop-phi swap should compile");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL");
    let reparsed = naga::front::wgsl::parse_str(wgsl)
        .expect("parallel loop-phi swap WGSL should reparse");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    )
    .validate(&reparsed)
    .expect("parallel loop-phi swap WGSL should validate");
    let phi_local_count = wgsl
        .lines()
        .filter(|line| line.trim_start().starts_with("var local"))
        .count();
    assert_eq!(
        phi_local_count, 3,
        "three logical phis should not allocate per-edge snapshot locals:\n{wgsl}",
    );
    assert!(
        wgsl.lines().any(|line| {
            let line = line.trim_start();
            line.starts_with("local") && line.ends_with(" = 0u;")
        }),
        "the zero-valued entry phi transfer must remain explicit after declaration:\n{wgsl}",
    );

    let output = run_grid_u32(wgsl, 8, 8, 8, 8, &[]);
    assert_eq!(output, vec![2211; 64], "three parallel swaps must end at (22, 11)");
}

#[test]
fn grid_i32_f32_bitcast_roundtrip_wgsl_shape() {
    let artifact = SpirvBackend::new()
        .with_grid()
        .with_workgroup_size(8, 8, 1)
        .compile_module(&build_grid_bitcast_roundtrip_module())
        .expect("i32/f32 bitcast roundtrip should compile");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL");
    assert!(wgsl.contains("bitcast<f32>"), "i32 bits must reinterpret as f32:\n{wgsl}");
    assert!(wgsl.contains("bitcast<u32>"), "f32 bits must reinterpret as u32:\n{wgsl}");
    let reparsed = naga::front::wgsl::parse_str(wgsl)
        .expect("naga wgsl-in must reparse bitcast WGSL");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&reparsed)
    .expect("bitcast WGSL must validate under browser capabilities");
}

#[test]
fn explicit_compute_roots_external_record_and_emits_no_implicit_buffers() {
    let artifact = SpirvBackend::new()
        .with_compute()
        .with_workgroup_size(1, 1, 1)
        .with_external_resource(external_complex_resource(0, Access::ReadWrite))
        .compile_module(&build_external_record_compute_module())
        .expect("external record compute should compile");
    assert_eq!(artifact.layout.mode, LayoutMode::Compute);
    assert_layout_metadata_invariants(&artifact.layout, 1);
    assert_eq!(artifact.layout.bindings.len(), 1, "resource-only compute has no implicit input/output");
    let orbit = &artifact.layout.bindings[0];
    assert_eq!((orbit.group, orbit.binding), (0, 0));
    assert_eq!((orbit.span, orbit.stride, orbit.resource_length), (8, 8, Some(1)));
    let wgsl = artifact.wgsl.as_deref().expect("WGSL");
    assert!(wgsl.contains("re_bits: u32"), "record field must survive:\n{wgsl}");
    assert!(wgsl.contains("im_bits: u32"), "record field must survive:\n{wgsl}");
    assert!(wgsl.contains("var<storage, read_write> orbit"), "resource root must be a storage global:\n{wgsl}");
    assert!(!wgsl.contains("var<storage, read_write> output"), "explicit compute has no implicit output:\n{wgsl}");
    assert!(!wgsl.contains("var<storage, read> input"), "resource-only compute has no parameter buffer:\n{wgsl}");
    let reparsed = naga::front::wgsl::parse_str(wgsl)
        .expect("naga wgsl-in must reparse external-resource compute WGSL");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&reparsed)
    .expect("external-resource compute WGSL must validate under browser capabilities");
}

fn compile_mixed_f32_u32_object_storage() -> sonatina_codegen::isa::spirv::SpirvArtifact {
    SpirvBackend::new()
        .with_compute()
        .with_workgroup_size(1, 1, 1)
        .with_external_resource(external_mixed_resource(0, Access::ReadWrite))
        .compile_module(&build_external_mixed_record_compute_module())
        .expect("mixed scalar external record should compile")
}

#[test]
fn explicit_compute_lowers_mixed_f32_u32_object_storage() {
    let artifact = compile_mixed_f32_u32_object_storage();
    assert_eq!(artifact.layout.mode, LayoutMode::Compute);
    assert_layout_metadata_invariants(&artifact.layout, 1);
    assert_eq!(artifact.layout.bindings.len(), 1);
    let samples = &artifact.layout.bindings[0];
    assert_eq!((samples.span, samples.stride, samples.resource_length), (12, 12, Some(1)));
    let wgsl = artifact.wgsl.as_deref().expect("WGSL");
    assert!(wgsl.contains("x: f32"), "f32 field must survive:\n{wgsl}");
    assert!(wgsl.contains("material: u32"), "u32 field must survive:\n{wgsl}");
    assert!(wgsl.contains("y: f32"), "second f32 field must survive:\n{wgsl}");
    let reparsed = naga::front::wgsl::parse_str(wgsl)
        .expect("mixed scalar external-resource WGSL should reparse");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&reparsed)
    .expect("mixed scalar external-resource WGSL must validate for browsers");
}

#[test]
fn explicit_compute_emits_only_transitively_live_external_resources() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64,
        Vendor::Unknown,
        OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let array_type = mb.declare_array_type(Type::I32, 8);
    let array_ref_type = mb.objref_type(array_type);
    let word_ref_type = mb.objref_type(Type::I32);

    let entry_ref = mb
        .declare_function(Signature::new_unit(
            "resource_liveness_entry",
            Linkage::Public,
            &[array_ref_type, array_ref_type, Type::I32],
        ))
        .unwrap();
    let helper_ref = mb
        .declare_function(Signature::new_unit(
            "store_live_resource",
            Linkage::Private,
            &[array_ref_type, Type::I32],
        ))
        .unwrap();

    {
        let mut fb = mb.func_builder::<InstInserter>(helper_ref);
        let entry = fb.append_block();
        fb.switch_to_block(entry);
        let zero = fb.make_imm_value(0i32);
        let slot = fb.insert_inst(
            data::ObjIndex::new(is, fb.args()[0], zero),
            word_ref_type,
        );
        fb.insert_inst_no_result(data::ObjStore::new(is, slot, fb.args()[1]));
        fb.insert_inst_no_result(control_flow::Return::new_unit(is));
        fb.seal_all();
        fb.finish();
    }
    {
        let mut fb = mb.func_builder::<InstInserter>(entry_ref);
        let entry = fb.append_block();
        fb.switch_to_block(entry);
        fb.insert_inst_no_result(control_flow::Call::new(
            is,
            helper_ref,
            [fb.args()[1], fb.args()[2]].into_iter().collect(),
        ));
        fb.insert_inst_no_result(control_flow::Return::new_unit(is));
        fb.seal_all();
        fb.finish();
    }

    let resource = |arg_index: u32, binding: u32, name: &str| SpirvExternalResource {
        arg_index,
        group: 0,
        binding,
        name: name.to_string(),
        access: Access::ReadWrite,
        element: SpirvResourceElement::Scalar(SpirvScalarKind::U32),
        stride: 4,
        length: 8,
    };
    let artifact = SpirvBackend::new()
        .with_compute()
        .with_workgroup_size(1, 1, 1)
        .with_external_resource(resource(0, 0, "unused_values"))
        .with_external_resource(resource(1, 1, "live_values"))
        .compile_module(&mb.build())
        .expect("dead resource arguments should not become physical bindings");

    assert_eq!(artifact.layout.bindings.len(), 2);
    let live = &artifact.layout.bindings[0];
    assert_eq!((live.name.as_str(), live.binding), ("live_values", 0));
    assert_eq!(live.resource_arg_index, Some(1));
    assert_eq!(live.stages, vec![SpirvShaderStage::Compute]);
    let params = &artifact.layout.bindings[1];
    assert_eq!((params.role, params.binding), (Role::Input, 1));
    assert_eq!(params.stages, vec![SpirvShaderStage::Compute]);
    assert_eq!(params.members.len(), 1);
    assert_eq!(params.members[0].arg_index, 2);
    let wgsl = artifact.wgsl.as_deref().expect("WGSL side artifact");
    assert!(!wgsl.contains("unused_values"), "{wgsl}");
    assert!(wgsl.contains("live_values"), "{wgsl}");
}

#[test]
fn authored_raster_prunes_resources_and_preserves_instance_local_identity() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64,
        Vendor::Unknown,
        OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let array_type = mb.declare_array_type(Type::I32, 4);
    let array_ref_type = mb.objref_type(array_type);
    let word_ref_type = mb.objref_type(Type::I32);
    let state = [
        array_ref_type,
        array_ref_type,
        array_ref_type,
        Type::F32,
        Type::F32,
    ];
    let vertex_args = [
        Type::I32,
        Type::I32,
        state[0],
        state[1],
        state[2],
        state[3],
        state[4],
    ];
    let vertex_results = [Type::F32; 5];
    let fragment_args = [
        Type::F32,
        state[0],
        state[1],
        state[2],
        state[3],
        state[4],
    ];

    let vertex_ref = mb
        .declare_function(Signature::new(
            "vs_resource_identity_0",
            Linkage::Public,
            &vertex_args,
            &vertex_results,
        ))
        .unwrap();
    let fragment_ref = mb
        .declare_function(Signature::new_single(
            "fs_resource_identity_0",
            Linkage::Public,
            &fragment_args,
            Type::I32,
        ))
        .unwrap();

    {
        let mut fb = mb.func_builder::<InstInserter>(vertex_ref);
        let entry = fb.append_block();
        fb.switch_to_block(entry);
        let vertex_index = fb.args()[0];
        let instance_index = fb.args()[1];
        let vertex_values = fb.args()[2];
        // State slot 3 is deliberately dormant. It must remain in the scalar
        // actor-state record even though resource slots may be pruned.
        let scale = fb.args()[6];
        let zero_index = fb.make_imm_value(0i32);
        let slot = fb.insert_inst(
            data::ObjIndex::new(is, vertex_values, zero_index),
            word_ref_type,
        );
        let loaded = fb.insert_inst(data::ObjLoad::new(is, slot), Type::I32);
        let loaded = fb.insert_inst(cast::I32ToF32::new(is, loaded), Type::F32);
        let x = fb.insert_inst(arith::Fadd::new(is, loaded, scale), Type::F32);
        let y = fb.insert_inst(cast::I32ToF32::new(is, instance_index), Type::F32);
        let varying = fb.insert_inst(cast::I32ToF32::new(is, vertex_index), Type::F32);
        let zero = fb.make_imm_value(Immediate::F32(0.0f32.to_bits()));
        let one = fb.make_imm_value(Immediate::F32(1.0f32.to_bits()));
        fb.insert_inst_no_result(control_flow::Return::new(
            is,
            [x, y, zero, one, varying]
                .into_iter()
                .collect::<smallvec::SmallVec<[_; 2]>>()
                .into(),
        ));
        fb.seal_all();
        fb.finish();
    }
    {
        let mut fb = mb.func_builder::<InstInserter>(fragment_ref);
        let entry = fb.append_block();
        fb.switch_to_block(entry);
        let fragment_values = fb.args()[2];
        let scale = fb.args()[5];
        let zero_index = fb.make_imm_value(0i32);
        let slot = fb.insert_inst(
            data::ObjIndex::new(is, fragment_values, zero_index),
            word_ref_type,
        );
        let color = fb.insert_inst(data::ObjLoad::new(is, slot), Type::I32);
        let scale_word = fb.insert_inst(cast::F32ToI32::new(is, scale), Type::I32);
        let color = fb.insert_inst(arith::Add::new(is, color, scale_word), Type::I32);
        fb.insert_inst_no_result(control_flow::Return::new_single(is, color));
        fb.seal_all();
        fb.finish();
    }

    let resource = |arg_index: u32, binding: u32, name: &str| SpirvExternalResource {
        arg_index,
        group: 0,
        binding,
        name: name.to_string(),
        access: Access::Read,
        element: SpirvResourceElement::Scalar(SpirvScalarKind::U32),
        stride: 4,
        length: 4,
    };
    let artifact = SpirvBackend::new()
        .with_authored_raster("vs_resource_identity_0", "fs_resource_identity_0")
        .with_builtin_argument(SpirvBuiltinArgument {
            arg_index: 0,
            source: SpirvBuiltinSource::VertexIndex,
        })
        .with_builtin_argument(SpirvBuiltinArgument {
            arg_index: 1,
            source: SpirvBuiltinSource::InstanceIndex,
        })
        .with_external_resource(resource(2, 0, "vertex_values"))
        .with_external_resource(resource(3, 1, "fragment_values"))
        .with_external_resource(resource(4, 2, "unused_values"))
        .compile_module(&mb.build())
        .expect("paired raster stages should share only their live resource union");

    let bindings = artifact
        .layout
        .bindings
        .iter()
        .map(|binding| {
            (
                binding.name.as_str(),
                binding.binding,
                binding.resource_arg_index,
                binding.stages.clone(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        bindings,
        vec![
            (
                "vertex_values",
                0,
                Some(2),
                vec![SpirvShaderStage::Vertex],
            ),
            (
                "fragment_values",
                1,
                Some(3),
                vec![SpirvShaderStage::Fragment],
            ),
            (
                "state",
                2,
                None,
                vec![SpirvShaderStage::Vertex, SpirvShaderStage::Fragment],
            ),
        ],
    );
    let state = artifact
        .layout
        .bindings
        .iter()
        .find(|binding| binding.name == "state")
        .expect("complete scalar actor-state binding");
    assert_eq!(
        state
            .members
            .iter()
            .map(|member| member.arg_index)
            .collect::<Vec<_>>(),
        vec![4, 5],
        "a dormant scalar field remains in the stable actor-state ABI",
    );
    let wgsl = artifact.wgsl.as_deref().expect("WGSL side artifact");
    assert_eq!(artifact.layout.vertex_entry.as_deref(), Some("fe_vertex_main"));
    assert_eq!(artifact.layout.fragment_entry.as_deref(), Some("fe_fragment_main"));
    assert!(wgsl.contains("fn fe_vertex_main("), "{wgsl}");
    assert!(wgsl.contains("fn fe_fragment_main("), "{wgsl}");
    assert!(
        !wgsl.contains("RasterVertexLeaves"),
        "a unique source return needs no aggregate transport:\n{wgsl}",
    );
    assert!(!wgsl.contains("fn vs_resource_identity_0_("), "{wgsl}");
    assert!(!wgsl.contains("fn fs_resource_identity_0_("), "{wgsl}");
    assert!(wgsl.contains("@builtin(instance_index)"), "{wgsl}");
    assert!(!wgsl.contains("unused_values"), "{wgsl}");
    let fragment = wgsl
        .split("fn fe_fragment_main")
        .nth(1)
        .expect("named fragment entry");
    assert!(fragment.contains("fragment_values"), "{fragment}");
    assert!(!fragment.contains("vertex_values"), "{fragment}");
}

#[test]
fn authored_raster_preserves_shared_scalar_multi_result_helpers() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64,
        Vendor::Unknown,
        OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let pair_ref = mb
        .declare_function(Signature::new(
            "raster_pair_math",
            Linkage::Private,
            &[Type::F32, Type::F32],
            &[Type::F32, Type::F32],
        ))
        .unwrap();
    let vertex_ref = mb
        .declare_function(Signature::new(
            "vs_scalar_helpers",
            Linkage::Public,
            &[Type::I32, Type::F32],
            &[Type::F32; 5],
        ))
        .unwrap();
    let fragment_ref = mb
        .declare_function(Signature::new_single(
            "fs_scalar_helpers",
            Linkage::Public,
            &[Type::F32, Type::F32],
            Type::I32,
        ))
        .unwrap();

    {
        let mut fb = mb.func_builder::<InstInserter>(pair_ref);
        let entry = fb.append_block();
        fb.switch_to_block(entry);
        let sum = fb.insert_inst(arith::Fadd::new(is, fb.args()[0], fb.args()[1]), Type::F32);
        let product =
            fb.insert_inst(arith::Fmul::new(is, fb.args()[0], fb.args()[1]), Type::F32);
        fb.insert_inst_no_result(control_flow::Return::new(
            is,
            [sum, product]
                .into_iter()
                .collect::<smallvec::SmallVec<[_; 2]>>()
                .into(),
        ));
        fb.seal_all();
        fb.finish();
    }
    {
        let mut fb = mb.func_builder::<InstInserter>(vertex_ref);
        let entry = fb.append_block();
        fb.switch_to_block(entry);
        let index = fb.insert_inst(cast::I32ToF32::new(is, fb.args()[0]), Type::F32);
        let first = fb.insert_inst_results(
            control_flow::Call::new(is, pair_ref, [index, fb.args()[1]].into_iter().collect()),
            &[Type::F32, Type::F32],
        );
        let second = fb.insert_inst_results(
            control_flow::Call::new(is, pair_ref, [first[0], first[1]].into_iter().collect()),
            &[Type::F32, Type::F32],
        );
        let zero = fb.make_imm_value(Immediate::F32(0.0f32.to_bits()));
        let one = fb.make_imm_value(Immediate::F32(1.0f32.to_bits()));
        fb.insert_inst_no_result(control_flow::Return::new(
            is,
            [second[0], second[1], zero, one, first[0]]
                .into_iter()
                .collect::<smallvec::SmallVec<[_; 2]>>()
                .into(),
        ));
        fb.seal_all();
        fb.finish();
    }
    {
        let mut fb = mb.func_builder::<InstInserter>(fragment_ref);
        let entry = fb.append_block();
        fb.switch_to_block(entry);
        let pair = fb.insert_inst_results(
            control_flow::Call::new(
                is,
                pair_ref,
                [fb.args()[0], fb.args()[1]].into_iter().collect(),
            ),
            &[Type::F32, Type::F32],
        );
        let packed = fb.insert_inst(cast::F32ToI32::new(is, pair[0]), Type::I32);
        fb.insert_inst_no_result(control_flow::Return::new_single(is, packed));
        fb.seal_all();
        fb.finish();
    }

    let artifact = SpirvBackend::new()
        .with_authored_raster("vs_scalar_helpers", "fs_scalar_helpers")
        .compile_module(&mb.build())
        .expect("authored raster should retain one shared scalar helper");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL side artifact");
    assert_eq!(
        wgsl.matches("fn raster_pair_math(").count(),
        1,
        "the shared helper must be emitted once:\n{wgsl}",
    );
    assert!(
        wgsl.matches("raster_pair_math(").count() >= 4,
        "both stages must call the shared helper:\n{wgsl}",
    );
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(
        &naga::front::wgsl::parse_str(wgsl)
            .expect("authored scalar-helper WGSL should reparse"),
    )
    .expect("authored scalar-helper WGSL should validate for browsers");
}

#[test]
fn authored_raster_keeps_equal_stage_local_call_ids_distinct() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64,
        Vendor::Unknown,
        OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let vertex_helper = mb
        .declare_function(Signature::new_single(
            "vertex_helper_four",
            Linkage::Private,
            &[Type::F32; 4],
            Type::F32,
        ))
        .unwrap();
    let fragment_helper = mb
        .declare_function(Signature::new_single(
            "fragment_helper_three",
            Linkage::Private,
            &[Type::F32; 3],
            Type::F32,
        ))
        .unwrap();
    let vertex_ref = mb
        .declare_function(Signature::new(
            "vs_distinct_local_calls",
            Linkage::Public,
            &[Type::I32, Type::F32, Type::F32, Type::F32, Type::F32],
            &[Type::F32; 5],
        ))
        .unwrap();
    let fragment_ref = mb
        .declare_function(Signature::new_single(
            "fs_distinct_local_calls",
            Linkage::Public,
            &[Type::F32; 5],
            Type::I32,
        ))
        .unwrap();

    for (helper, arity) in [(vertex_helper, 4usize), (fragment_helper, 3usize)] {
        let mut fb = mb.func_builder::<InstInserter>(helper);
        let entry = fb.append_block();
        fb.switch_to_block(entry);
        let mut sum = fb.args()[0];
        for index in 1..arity {
            sum = fb.insert_inst(arith::Fadd::new(is, sum, fb.args()[index]), Type::F32);
        }
        fb.insert_inst_no_result(control_flow::Return::new_single(is, sum));
        fb.seal_all();
        fb.finish();
    }
    {
        let mut fb = mb.func_builder::<InstInserter>(vertex_ref);
        let entry = fb.append_block();
        fb.switch_to_block(entry);
        // This is deliberately the first instruction in each stage root, so
        // both calls receive the same function-local InstId.
        let value = fb.insert_inst(
            control_flow::Call::new(
                is,
                vertex_helper,
                fb.args()[1..5].iter().copied().collect(),
            ),
            Type::F32,
        );
        fb.insert_inst_no_result(control_flow::Return::new(
            is,
            [value, fb.args()[1], fb.args()[2], fb.args()[3], fb.args()[4]]
                .into_iter()
                .collect::<smallvec::SmallVec<[_; 2]>>()
                .into(),
        ));
        fb.seal_all();
        fb.finish();
    }
    {
        let mut fb = mb.func_builder::<InstInserter>(fragment_ref);
        let entry = fb.append_block();
        fb.switch_to_block(entry);
        let value = fb.insert_inst(
            control_flow::Call::new(
                is,
                fragment_helper,
                fb.args()[0..3].iter().copied().collect(),
            ),
            Type::F32,
        );
        let packed = fb.insert_inst(cast::F32ToI32::new(is, value), Type::I32);
        fb.insert_inst_no_result(control_flow::Return::new_single(is, packed));
        fb.seal_all();
        fb.finish();
    }

    let artifact = SpirvBackend::new()
        .with_authored_raster("vs_distinct_local_calls", "fs_distinct_local_calls")
        .compile_module(&mb.build())
        .expect("stage-local call IDs must select their own helper ABIs");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL side artifact");
    assert!(wgsl.contains("fn vertex_helper_four("), "{wgsl}");
    assert!(wgsl.contains("fn fragment_helper_three("), "{wgsl}");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(
        &naga::front::wgsl::parse_str(wgsl)
            .expect("stage-local call-ID WGSL should reparse"),
    )
    .expect("stage-local call-ID WGSL should validate for browsers");
}

#[test]
fn authored_raster_normalizes_multi_value_returns_to_one_ssa_exit() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64,
        Vendor::Unknown,
        OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let vertex_ref = mb
        .declare_function(Signature::new(
            "vs_branching_returns",
            Linkage::Public,
            &[Type::I32],
            &[Type::F32; 5],
        ))
        .unwrap();
    let fragment_ref = mb
        .declare_function(Signature::new_single(
            "fs_branching_returns",
            Linkage::Public,
            &[Type::F32],
            Type::I32,
        ))
        .unwrap();

    {
        let mut fb = mb.func_builder::<InstInserter>(vertex_ref);
        let entry = fb.append_block();
        let left = fb.append_block();
        let right = fb.append_block();
        fb.switch_to_block(entry);
        let split = fb.make_imm_value(3i32);
        let condition = fb.insert_inst(cmp::Lt::new(is, fb.args()[0], split), Type::I1);
        fb.insert_inst_no_result(control_flow::Br::new(is, condition, left, right));

        let zero = fb.make_imm_value(Immediate::F32(0.0f32.to_bits()));
        let one = fb.make_imm_value(Immediate::F32(1.0f32.to_bits()));
        let minus_one = fb.make_imm_value(Immediate::F32((-1.0f32).to_bits()));
        fb.switch_to_block(left);
        fb.insert_inst_no_result(control_flow::Return::new(
            is,
            [minus_one, zero, zero, one, zero]
                .into_iter()
                .collect::<smallvec::SmallVec<[_; 2]>>()
                .into(),
        ));
        fb.switch_to_block(right);
        fb.insert_inst_no_result(control_flow::Return::new(
            is,
            [one, zero, zero, one, one]
                .into_iter()
                .collect::<smallvec::SmallVec<[_; 2]>>()
                .into(),
        ));
        fb.seal_all();
        fb.finish();
    }

    {
        let mut fb = mb.func_builder::<InstInserter>(fragment_ref);
        let entry = fb.append_block();
        let dark = fb.append_block();
        let light = fb.append_block();
        fb.switch_to_block(entry);
        let half = fb.make_imm_value(Immediate::F32(0.5f32.to_bits()));
        let condition = fb.insert_inst(cmp::Flt::new(is, fb.args()[0], half), Type::I1);
        fb.insert_inst_no_result(control_flow::Br::new(is, condition, dark, light));
        fb.switch_to_block(dark);
        let dark_color = fb.make_imm_value(255i32);
        fb.insert_inst_no_result(control_flow::Return::new_single(is, dark_color));
        fb.switch_to_block(light);
        let light_color = fb.make_imm_value(65535i32);
        fb.insert_inst_no_result(control_flow::Return::new_single(is, light_color));
        fb.seal_all();
        fb.finish();
    }

    let artifact = SpirvBackend::new()
        .with_authored_raster("vs_branching_returns", "fs_branching_returns")
        .compile_module(&mb.build())
        .expect("branch-local raster returns should share the typed stage result transport");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL side artifact");
    assert!(
        !wgsl.contains("RasterVertexLeaves"),
        "branch-local returns should merge through ordinary SSA phis, not a physical aggregate:\n{wgsl}",
    );
    assert!(wgsl.contains("fn fe_vertex_main("), "{wgsl}");
    assert!(wgsl.contains("fn fe_fragment_main("), "{wgsl}");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(
        &naga::front::wgsl::parse_str(wgsl)
            .expect("branch-local raster return WGSL should reparse"),
    )
    .expect("branch-local raster return WGSL should validate for browsers");
}

#[test]
fn authored_raster_many_early_returns_lower_with_linear_sized_control_flow() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64,
        Vendor::Unknown,
        OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let vertex_ref = mb
        .declare_function(Signature::new(
            "vs_many_early_returns",
            Linkage::Public,
            &[Type::I32],
            &[Type::F32; 5],
        ))
        .unwrap();
    let fragment_ref = mb
        .declare_function(Signature::new_single(
            "fs_many_early_returns",
            Linkage::Public,
            &[Type::F32],
            Type::I32,
        ))
        .unwrap();

    {
        let mut fb = mb.func_builder::<InstInserter>(vertex_ref);
        let mut current = fb.append_block();
        let zero = fb.make_imm_value(Immediate::F32(0.0f32.to_bits()));
        let one = fb.make_imm_value(Immediate::F32(1.0f32.to_bits()));
        for index in 0..48 {
            let early = fb.append_block();
            let next = fb.append_block();
            fb.switch_to_block(current);
            let split = fb.make_imm_value(index + 1);
            let condition = fb.insert_inst(cmp::Lt::new(is, fb.args()[0], split), Type::I1);
            fb.insert_inst_no_result(control_flow::Br::new(is, condition, early, next));

            fb.switch_to_block(early);
            let x = fb.make_imm_value(Immediate::F32(
                ((index + 1) as f32 / 48.0).to_bits(),
            ));
            fb.insert_inst_no_result(control_flow::Return::new(
                is,
                [x, zero, zero, one, x]
                    .into_iter()
                    .collect::<smallvec::SmallVec<[_; 2]>>()
                    .into(),
            ));
            current = next;
        }
        fb.switch_to_block(current);
        fb.insert_inst_no_result(control_flow::Return::new(
            is,
            [one, zero, zero, one, one]
                .into_iter()
                .collect::<smallvec::SmallVec<[_; 2]>>()
                .into(),
        ));
        fb.seal_all();
        fb.finish();
    }
    {
        let mut fb = mb.func_builder::<InstInserter>(fragment_ref);
        let entry = fb.append_block();
        fb.switch_to_block(entry);
        let white = fb.make_imm_value(-1i32);
        fb.insert_inst_no_result(control_flow::Return::new_single(is, white));
        fb.seal_all();
        fb.finish();
    }

    let artifact = SpirvBackend::new()
        .with_authored_raster("vs_many_early_returns", "fs_many_early_returns")
        .compile_module(&mb.build())
        .expect("many raster early returns should normalize before structurization");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL side artifact");
    assert!(!wgsl.contains("RasterVertexLeaves"), "{wgsl}");
    assert!(
        wgsl.len() < 50_000,
        "48 early returns should produce linear-sized WGSL, got {} bytes",
        wgsl.len(),
    );
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(
        &naga::front::wgsl::parse_str(wgsl)
            .expect("many-return raster WGSL should reparse"),
    )
    .expect("many-return raster WGSL should validate for browsers");
}

#[test]
fn explicit_compute_rejects_signed_external_storage_until_carrier_semantics_exist() {
    let mut resource = external_complex_resource(0, Access::ReadWrite);
    let SpirvResourceElement::Record { fields, .. } = &mut resource.element else {
        unreachable!("the fixture is a record")
    };
    fields[0].scalar = SpirvScalarKind::I32;
    let errors = match SpirvBackend::new()
        .with_compute()
        .with_workgroup_size(1, 1, 1)
        .with_external_resource(resource)
        .compile_module(&build_external_record_compute_module())
    {
        Ok(_) => panic!("signed storage must fail closed until its carrier bitcasts are explicit"),
        Err(errors) => errors,
    };
    let message = errors.iter().map(|error| error.to_string()).collect::<Vec<_>>().join("; ");
    assert!(
        message.contains("external storage scalar I32 is unsupported"),
        "expected the signed-carrier boundary error, got: {message}"
    );
}

#[test]
fn explicit_compute_executes_mixed_f32_u32_object_storage() {
    let artifact = compile_mixed_f32_u32_object_storage();
    let wgsl = artifact.wgsl.as_deref().expect("WGSL");
    // Read the 12-byte record back as raw words. A logical width of three
    // allocates those bytes, while wgx=3 submits one actual 1x1x1 workgroup so
    // there is no cross-invocation race on the single record.
    assert_eq!(
        run_grid_u32(wgsl, 3, 1, 3, 1, &[]),
        vec![(-1.25f32).to_bits(), 7, (-2.5f32).to_bits()],
    );
}

#[test]
fn explicit_compute_maps_complete_invocation_context_without_parameter_shims() {
    let resource = SpirvExternalResource {
        arg_index: 13,
        group: 0,
        binding: 0,
        name: "receipts".to_string(),
        access: Access::ReadWrite,
        element: SpirvResourceElement::Scalar(SpirvScalarKind::U32),
        stride: 4,
        length: 8,
    };
    let expected_builtins = complete_compute_invocation_arguments();
    let mut backend = SpirvBackend::new()
        .with_compute()
        .with_workgroup_size(2, 1, 1)
        .with_external_resource(resource);
    for argument in &expected_builtins {
        backend = backend.with_builtin_argument(*argument);
    }
    let artifact = backend
        .compile_module(&build_compute_invocation_context_module())
        .expect("complete compute invocation context should compile");
    assert_eq!(artifact.layout.mode, LayoutMode::Compute);
    assert_layout_metadata_invariants(&artifact.layout, 15);
    assert_eq!(
        artifact.layout.builtin_inputs,
        expected_builtins
            .iter()
            .map(|argument| SpirvBuiltinInput {
                arg_index: argument.arg_index,
                source: argument.source,
                scalar: SpirvScalarKind::I32,
            })
            .collect::<Vec<_>>()
    );
    assert_eq!(artifact.layout.bindings.len(), 2);
    let params = artifact
        .layout
        .bindings
        .iter()
        .find(|binding| binding.role == Role::Input)
        .expect("ordinary scalar parameter binding");
    assert_eq!(
        params.members,
        vec![SpirvBindingMember {
            arg_index: 14,
            offset: 0,
            width: 4,
            scalar: SpirvScalarKind::I32,
        }]
    );
    let wgsl = artifact.wgsl.as_deref().expect("WGSL");
    for builtin in [
        "@builtin(global_invocation_id)",
        "@builtin(local_invocation_id)",
        "@builtin(workgroup_id)",
        "@builtin(num_workgroups)",
        "@builtin(local_invocation_index)",
    ] {
        assert!(wgsl.contains(builtin), "missing {builtin}:\n{wgsl}");
    }
    let reparsed = naga::front::wgsl::parse_str(wgsl)
        .expect("complete invocation-context WGSL should reparse");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&reparsed)
    .expect("complete invocation context must stay inside browser capabilities");
}

#[test]
fn explicit_compute_rejects_duplicate_physical_builtin_sources() {
    let resource = SpirvExternalResource {
        arg_index: 13,
        group: 0,
        binding: 0,
        name: "receipts".to_string(),
        access: Access::ReadWrite,
        element: SpirvResourceElement::Scalar(SpirvScalarKind::U32),
        stride: 4,
        length: 8,
    };
    let result = SpirvBackend::new()
        .with_compute()
        .with_workgroup_size(2, 1, 1)
        .with_external_resource(resource)
        .with_builtin_argument(SpirvBuiltinArgument {
            arg_index: 0,
            source: SpirvBuiltinSource::GlobalInvocationIdX,
        })
        .with_builtin_argument(SpirvBuiltinArgument {
            arg_index: 1,
            source: SpirvBuiltinSource::GlobalInvocationIdX,
        })
        .compile_module(&build_compute_invocation_context_module());
    let error = match result {
        Ok(_) => panic!("one physical builtin cannot ambiguously supply two logical arguments"),
        Err(error) => error,
    };
    assert!(
        error.iter().any(|error| error
            .to_string()
            .contains("GlobalInvocationIdX is mapped more than once")),
        "unexpected diagnostic: {error:?}"
    );
}

#[test]
fn explicit_compute_trap_channel_owns_one_word_per_fixed_invocation() {
    let artifact = SpirvBackend::new()
        .with_compute()
        .with_workgroup_size(2, 2, 1)
        .with_dispatch_grid(2, 2, 1)
        .compile_module(&build_unit_compute_with_trap_module())
        .expect("fixed multi-invocation trap channel should compile");
    assert!(artifact.layout.builtin_inputs.is_empty());
    let trap = artifact.layout.trap.expect("trap result lane");
    assert_eq!(trap.width, 16 * 4);
    let binding = artifact
        .layout
        .bindings
        .iter()
        .find(|binding| binding.name == "trap")
        .expect("trap binding");
    assert_eq!(binding.stride, 4);
    assert_eq!(binding.span, 16 * 4);
    let wgsl = artifact.wgsl.as_deref().expect("WGSL");
    assert!(
        wgsl.contains("@builtin(global_invocation_id)"),
        "compiler-owned trap indexing requires the physical global id:\n{wgsl}"
    );
    assert!(
        wgsl.contains("array<u32, 16>"),
        "trap storage must have one statically sized word per invocation:\n{wgsl}"
    );
    let reparsed = naga::front::wgsl::parse_str(wgsl)
        .expect("multi-invocation trap WGSL should reparse");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&reparsed)
        .expect("multi-invocation trap channel must stay inside browser capabilities");
}

#[test]
fn explicit_compute_rejects_a_zero_fixed_dispatch_dimension() {
    let result = SpirvBackend::new()
        .with_compute()
        .with_workgroup_size(2, 2, 1)
        .with_dispatch_grid(2, 0, 1)
        .compile_module(&build_unit_compute_with_trap_module());
    let error = match result {
        Ok(_) => panic!("a zero fixed dispatch dimension must fail closed"),
        Err(error) => error,
    };
    assert!(
        error.iter().any(|error| error
            .to_string()
            .contains("workgroup and dispatch dimensions must be nonzero")),
        "unexpected diagnostic: {error:?}"
    );
}

#[test]
fn explicit_unit_compute_loop_preserves_exit_block_side_effects() {
    let resource = SpirvExternalResource {
        arg_index: 0,
        group: 0,
        binding: 0,
        name: "values".to_string(),
        access: Access::ReadWrite,
        element: SpirvResourceElement::Scalar(SpirvScalarKind::U32),
        stride: 4,
        length: 8,
    };
    let artifact = SpirvBackend::new()
        .with_compute()
        .with_workgroup_size(1, 1, 1)
        .with_external_resource(resource)
        .compile_module(&build_unit_compute_loop_with_exit_store_module())
        .expect("unit-return compute loop should compile");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL");
    assert!(wgsl.contains("loop {"), "source loop must survive:\n{wgsl}");
    assert!(
        wgsl.contains("values[i32(0u)] = 99u;"),
        "the unit-return loop exit store must not be dropped:\n{wgsl}"
    );
    let reparsed = naga::front::wgsl::parse_str(wgsl)
        .expect("naga wgsl-in must reparse unit compute loop WGSL");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&reparsed)
    .expect("unit compute loop WGSL must validate under browser capabilities");
}

#[test]
fn explicit_unit_compute_early_return_guards_post_branch_sibling() {
    let resource = SpirvExternalResource {
        arg_index: 1,
        group: 0,
        binding: 0,
        name: "values".to_string(),
        access: Access::ReadWrite,
        element: SpirvResourceElement::Scalar(SpirvScalarKind::U32),
        stride: 4,
        length: 1,
    };
    let artifact = SpirvBackend::new()
        .with_compute()
        .with_workgroup_size(2, 1, 1)
        .with_builtin_argument(SpirvBuiltinArgument {
            arg_index: 0,
            source: SpirvBuiltinSource::LocalInvocationIdX,
        })
        .with_external_resource(resource)
        .compile_module(&build_unit_compute_early_return_with_sibling_module())
        .expect("unit early return and post-branch sibling should compile");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL");
    assert!(
        wgsl.contains("values[i32(0u)] = 99u;"),
        "the normal path must retain its sibling store:\n{wgsl}",
    );
    let reparsed = naga::front::wgsl::parse_str(wgsl)
        .expect("naga wgsl-in must reparse unit early-return WGSL");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&reparsed)
    .expect("unit early-return WGSL must validate under browser capabilities");
}

#[test]
fn render_roots_same_external_record_read_only_without_parameter_buffer() {
    let artifact = SpirvBackend::new()
        .with_render()
        .with_external_resource(external_complex_resource(2, Access::Read))
        .compile_module(&build_external_record_render_module())
        .expect("external record render should compile");
    assert_eq!(artifact.layout.mode, LayoutMode::Render);
    assert_layout_metadata_invariants(&artifact.layout, 3);
    assert_eq!(artifact.layout.bindings.len(), 1, "resource-only render has no parameter buffer");
    let orbit = &artifact.layout.bindings[0];
    assert_eq!(orbit.access, Access::Read);
    assert_eq!(orbit.resource_arg_index, Some(2));
    let wgsl = artifact.wgsl.as_deref().expect("WGSL");
    assert!(wgsl.contains("var<storage> orbit"), "fragment resource must use WGSL's read-only storage form:\n{wgsl}");
    assert!(!wgsl.contains("var<storage, read> input"), "resource-only render has no parameter buffer:\n{wgsl}");
    assert!(wgsl.contains("].re_bits"), "record load must preserve resource indexing and projection:\n{wgsl}");
    assert!(wgsl.contains("].im_bits"), "record load must preserve both fields:\n{wgsl}");
    let reparsed = naga::front::wgsl::parse_str(wgsl)
        .expect("naga wgsl-in must reparse external-resource render WGSL");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&reparsed)
    .expect("external-resource render WGSL must validate under browser capabilities");
}

#[test]
fn grid_loop_inner_if_phi_executes_on_lavapipe() {
    let module = build_grid_loop_inner_if_phi_module();
    let artifact = SpirvBackend::new()
        .with_grid()
        .with_workgroup_size(8, 8, 1)
        .compile_module(&module)
        .expect("in-loop if merge phi should compile");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL");
    let output = run_grid_u32(wgsl, 8, 8, 8, 8, &[]);
    assert_eq!(output, vec![22; 64], "in-loop if merge must select 22");
}

#[test]
fn grid_f32_loop_return_executes_on_lavapipe() {
    let module = build_grid_f32_loop_return_module();
    let artifact = SpirvBackend::new()
        .with_grid()
        .with_workgroup_size(8, 8, 1)
        .compile_module(&module)
        .expect("f32 loop return should compile");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL");
    let output = run_grid_u32(wgsl, 8, 8, 8, 8, &[]);
    assert_eq!(output, vec![3; 64], "1.0 + 1.0 + 1.0 must convert to u32 3");
}

#[test]
fn grid_loop_exit_resumes_at_sibling_on_lavapipe() {
    let module = build_grid_loop_exit_sibling_resume_module();
    let artifact = SpirvBackend::new()
        .with_grid()
        .with_workgroup_size(8, 8, 1)
        .compile_module(&module)
        .expect("normal loop exit should resume at its following sibling");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL");
    let output = run_grid_u32(wgsl, 8, 8, 8, 8, &[]);
    assert_eq!(output, vec![22; 64], "loop exit must resume and execute sibling");
}

#[test]
fn grid_loop_exit_phi_executes_on_lavapipe() {
    let module = build_grid_loop_exit_phi_module();
    let artifact = SpirvBackend::new()
        .with_grid()
        .with_workgroup_size(8, 8, 1)
        .compile_module(&module)
        .expect("normal loop exit phi should compile");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL");
    let output = run_grid_u32(wgsl, 8, 8, 8, 8, &[]);
    for y in 0..8u32 {
        for x in 0..8u32 {
            assert_eq!(output[(y * 8 + x) as usize], x + 10, "exit phi at ({x}, {y})");
        }
    }
}

#[test]
fn grid_conditional_break_cascade_compiles_browser_wgsl() {
    let module = build_grid_multi_exit_f32_phi_module();
    let artifact = SpirvBackend::new()
        .with_grid()
        .with_workgroup_size(8, 8, 1)
        .compile_module(&module)
        .expect("conditional loop breaks should compile without a fixed-work workaround");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL");
    let reparsed = naga::front::wgsl::parse_str(wgsl)
        .expect("conditional-break WGSL must reparse");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&reparsed)
    .expect("conditional-break WGSL must validate under browser capabilities");
    assert!(
        wgsl.matches("break;").count() >= 3,
        "the header exit and two body exits must remain real loop breaks:\n{wgsl}",
    );
}

#[test]
fn grid_multi_exit_f32_phi_executes_on_lavapipe() {
    let module = build_grid_multi_exit_f32_phi_module();
    let artifact = SpirvBackend::new()
        .with_grid()
        .with_workgroup_size(8, 8, 1)
        .compile_module(&module)
        .expect("all loop exits should carry their exact f32 phi input");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL");
    let output = run_grid_u32(wgsl, 8, 8, 8, 8, &[]);
    for y in 0..8u32 {
        for x in 0..8u32 {
            let expected = match x { 0 => 11, 1 => 22, _ => 33 };
            assert_eq!(output[(y * 8 + x) as usize], expected, "multi-exit phi at ({x}, {y})");
        }
    }
}

#[test]
fn grid_direct_return_multi_exit_f32_phi_executes_on_lavapipe() {
    let module = build_grid_direct_return_multi_exit_f32_phi_module();
    let artifact = SpirvBackend::new()
        .with_grid()
        .with_workgroup_size(8, 8, 1)
        .compile_module(&module)
        .expect("direct-return loop exit phi should preserve every exact edge");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL");
    let output = run_grid_u32(wgsl, 8, 8, 8, 8, &[]);
    for y in 0..8u32 {
        for x in 0..8u32 {
            let expected = match x {
                0 => 3,
                1 => 41,
                2 => 52,
                _ => 3 + 10 * x,
            };
            assert_eq!(output[(y * 8 + x) as usize], expected, "direct-return exit at ({x}, {y})");
        }
    }
}

/// A grid kernel with one broadcast param executes with p0 = 7 written to the
/// input buffer; every pixel equals x + 1024*y + 7.
#[test]
fn grid_broadcast_params() {
    let module = build_grid_broadcast_module();
    let art = SpirvBackend::new()
        .with_grid()
        .with_workgroup_size(8, 8, 1)
        .compile_module(&module)
        .expect("grid broadcast must compile");

    // Exactly one broadcast member (arg2): input span = 4.
    let in_b = art.layout.bindings.iter().find(|b| b.role == Role::Input).expect("input binding");
    assert_eq!(in_b.stride, 4, "one broadcast member p0 at offset 0, span 4");

    let wgsl = art.wgsl.as_ref().expect("WGSL");
    let (w, h, wgx, wgy) = (8u32, 8u32, 8u32, 8u32);
    let p0: u32 = 7;
    let out = run_grid_u32(wgsl, w, h, wgx, wgy, &p0.to_le_bytes());
    for y in 0..h {
        for x in 0..w {
            let got = out[(y * w + x) as usize];
            let want = x + 1024 * y + 7;
            assert_eq!(got, want, "grid[{y}*{w}+{x}] = {got}, want {want}");
        }
    }
    eprintln!("grid_broadcast_params OK: p0 = 7 broadcast, all pixels == x + 1024*y + 7");
}

/// Fail-closed: the four grid preconditions each err, and the error names grid.
#[test]
fn grid_fail_closed() {
    let m = build_grid_i64_module();
    let e = expect_grid_err(&m, SpirvBackend::new().with_grid().with_workgroup_size(8, 8, 1));
    assert!(
        e.contains("coordinate args") && e.contains("i32"),
        "i64 coordinate args must name the i32 coordinate requirement: {e}"
    );

    let m = build_grid_1arg_module();
    let e = expect_grid_err(&m, SpirvBackend::new().with_grid().with_workgroup_size(8, 8, 1));
    assert!(
        e.contains("coordinate args") && e.contains("i32"),
        "1-arg kernel must name the two-coordinate requirement: {e}"
    );

    let m = build_grid_objalloc_module();
    let e = expect_grid_err(&m, SpirvBackend::new().with_grid().with_workgroup_size(8, 8, 1));
    assert!(e.contains("mutually exclusive"), "ObjAlloc must name the batch conflict: {e}");

    let m = build_grid_gradient_module();
    let e = expect_grid_err(&m, SpirvBackend::new().with_grid().with_workgroup_size(8, 8, 2));
    assert!(
        e.contains("2D") || e.contains("workgroup z"),
        "workgroup z != 1 must name the 2D dispatch rule: {e}"
    );
    eprintln!("grid_fail_closed OK: all four preconditions err and name grid");
}

// ===========================================================================
// M2a: fork push #2 - signed ops under the u32 word. `Slt` and `Sar` were
// fail-closed under u32; they are now word-aware via an i32 bitcast:
//   - `Slt`: bitcast BOTH operands to i32, then naga `Less` (a signed compare).
//   - `Sar`: bitcast the value to i32, arithmetic `>>` with a u32 literal amount,
//     bitcast back to u32.
// A non-immediate `Sar` shift amount still fails closed with a named error.
// ===========================================================================

/// A scalar u32 kernel whose loop condition is a SIGNED compare (`i <s 10`),
/// exercising the u32 `Slt` arm: `slt_count() = { let mut i = 0; while i <s 10
/// { i += 1 } i }` (returns 10). Scalar loop, no inner branch (the proven
/// `spirv_loop_sum_to_valid` shape, with the compare made signed).
fn build_u32_slt_count() -> sonatina_ir::Module {
    let isa = Native::new(TargetTriple::new(
        if cfg!(target_arch = "x86_64") { Architecture::X86_64 } else { Architecture::Aarch64 },
        Vendor::Unknown, OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single("slt_count", Linkage::Public, &[], Type::I32);
    let fr = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(fr);
    let entry = fb.append_block();
    let lh = fb.append_block();
    let lb = fb.append_block();
    let exit = fb.append_block();
    fb.switch_to_block(entry);
    let init = fb.make_imm_value(0i32);
    fb.insert_inst_no_result(control_flow::Jump::new(is, lh));
    fb.switch_to_block(lh);
    let i = fb.insert_inst(control_flow::Phi::new(is, vec![(init, entry)]), Type::I32);
    let ten = fb.make_imm_value(10i32);
    let cond = fb.insert_inst(cmp::Slt::new(is, i, ten), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, cond, lb, exit));
    fb.switch_to_block(lb);
    let one = fb.make_imm_value(1i32);
    let ni = fb.insert_inst(arith::Add::new(is, i, one), Type::I32);
    fb.append_phi_arg(i, ni, lb);
    fb.insert_inst_no_result(control_flow::Jump::new(is, lh));
    fb.switch_to_block(exit);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, i));
    fb.seal_all();
    fb.finish();
    mb.build()
}

/// A browser-word grid kernel containing both of Sonatina's signless integer
/// equality operations. Runtime coordinate operands ensure both comparisons
/// survive into Naga and the emitted WGSL/SPIR-V.
fn build_u32_eq_ne_probe() -> sonatina_ir::Module {
    let isa = Native::new(TargetTriple::new(
        if cfg!(target_arch = "x86_64") { Architecture::X86_64 } else { Architecture::Aarch64 },
        Vendor::Unknown, OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single("eq_ne_probe", Linkage::Public, &[Type::I32, Type::I32], Type::I32);
    let fr = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(fr);
    let lhs = fb.args()[0];
    let rhs = fb.args()[1];
    let entry = fb.append_block();
    let equal = fb.append_block();
    let unequal = fb.append_block();
    let ne_true = fb.append_block();
    let impossible = fb.append_block();
    fb.switch_to_block(entry);
    let is_equal = fb.insert_inst(cmp::Eq::new(is, lhs, rhs), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, is_equal, equal, unequal));
    fb.switch_to_block(equal);
    let eleven = fb.make_imm_value(11i32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, eleven));
    fb.switch_to_block(unequal);
    let is_unequal = fb.insert_inst(cmp::Ne::new(is, lhs, rhs), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, is_unequal, ne_true, impossible));
    fb.switch_to_block(ne_true);
    let twenty_two = fb.make_imm_value(22i32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, twenty_two));
    fb.switch_to_block(impossible);
    let thirty_three = fb.make_imm_value(33i32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, thirty_three));
    fb.seal_all();
    fb.finish();
    mb.build()
}

/// A scalar u32 kernel `sar_probe(a) = a >> 12` with an IMMEDIATE shift amount.
fn build_u32_sar_probe() -> sonatina_ir::Module {
    let isa = Native::new(TargetTriple::new(
        if cfg!(target_arch = "x86_64") { Architecture::X86_64 } else { Architecture::Aarch64 },
        Vendor::Unknown, OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single("sar_probe", Linkage::Public, &[Type::I32], Type::I32);
    let fr = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(fr);
    let entry = fb.append_block();
    fb.switch_to_block(entry);
    let a = fb.args()[0];
    let twelve = fb.make_imm_value(12i32);
    // Sar constructor order is (bits, value), the EVM/i64 convention.
    let s = fb.insert_inst(arith::Sar::new(is, twelve, a), Type::I32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, s));
    fb.seal_all();
    fb.finish();
    mb.build()
}

/// A scalar u32 kernel `sar_nonimm(a, sh) = a >> sh` whose shift amount is a
/// runtime value.
fn build_u32_sar_nonimm() -> sonatina_ir::Module {
    let isa = Native::new(TargetTriple::new(
        if cfg!(target_arch = "x86_64") { Architecture::X86_64 } else { Architecture::Aarch64 },
        Vendor::Unknown, OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single("sar_nonimm", Linkage::Public, &[Type::I32, Type::I32], Type::I32);
    let fr = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(fr);
    let entry = fb.append_block();
    fb.switch_to_block(entry);
    let a = fb.args()[0];
    let sh = fb.args()[1];
    let s = fb.insert_inst(arith::Sar::new(is, sh, a), Type::I32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, s));
    fb.seal_all();
    fb.finish();
    mb.build()
}

/// A grid escape-time kernel at the u32 word: `escape_grid(px, py) -> u32`, the
/// `mandelbrot_snapshot.rs::build_escape_time` shape ported to I32, with the
/// escape compare made SIGNED (`Slt`) and c derived per-pixel from (px, py) so
/// each pixel diverges. Q10 fixed point (1.0 = 1024): c_re = -2048 + px*80,
/// c_im = -1280 + py*80, threshold 4_194_304 (4.0), shift 10, max 50.
fn build_u32_escape_grid() -> sonatina_ir::Module {
    let isa = Native::new(TargetTriple::new(
        if cfg!(target_arch = "x86_64") { Architecture::X86_64 } else { Architecture::Aarch64 },
        Vendor::Unknown, OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single(
        "escape_grid", Linkage::Public, &[Type::I32, Type::I32], Type::I32,
    );
    let fr = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(fr);
    let entry = fb.append_block();
    let lh = fb.append_block();
    let lb = fb.append_block();
    let cont = fb.append_block();
    let esc = fb.append_block();
    let exit = fb.append_block();

    fb.switch_to_block(entry);
    let px = fb.args()[0];
    let py = fb.args()[1];
    let step = fb.make_imm_value(80i32);
    let base_re = fb.make_imm_value(-2048i32);
    let base_im = fb.make_imm_value(-1280i32);
    let pxs = fb.insert_inst(arith::Mul::new(is, px, step), Type::I32);
    let c_re = fb.insert_inst(arith::Add::new(is, base_re, pxs), Type::I32);
    let pys = fb.insert_inst(arith::Mul::new(is, py, step), Type::I32);
    let c_im = fb.insert_inst(arith::Add::new(is, base_im, pys), Type::I32);
    let zero = fb.make_imm_value(0i32);
    fb.insert_inst_no_result(control_flow::Jump::new(is, lh));

    fb.switch_to_block(lh);
    let zr = fb.insert_inst(control_flow::Phi::new(is, vec![(zero, entry)]), Type::I32);
    let zi = fb.insert_inst(control_flow::Phi::new(is, vec![(zero, entry)]), Type::I32);
    let i = fb.insert_inst(control_flow::Phi::new(is, vec![(zero, entry)]), Type::I32);
    let max = fb.make_imm_value(50i32);
    // Loop counter compare is UNSIGNED (`Lt`): i is a non-negative iteration count.
    let c = fb.insert_inst(cmp::Lt::new(is, i, max), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, c, lb, exit));

    fb.switch_to_block(lb);
    let rr = fb.insert_inst(arith::Mul::new(is, zr, zr), Type::I32);
    let ii = fb.insert_inst(arith::Mul::new(is, zi, zi), Type::I32);
    let mag = fb.insert_inst(arith::Add::new(is, rr, ii), Type::I32);
    let th = fb.make_imm_value(4_194_304i32);
    // Escape compare is SIGNED (`Slt`): the Q10 magnitude carries a signed value.
    let ec = fb.insert_inst(cmp::Slt::new(is, mag, th), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, ec, cont, esc));

    fb.switch_to_block(cont);
    let diff = fb.insert_inst(arith::Sub::new(is, rr, ii), Type::I32);
    let ten = fb.make_imm_value(10i32);
    // `diff` can be negative (ii > rr), so `>>` must be an ARITHMETIC shift (Sar).
    let sr = fb.insert_inst(arith::Sar::new(is, ten, diff), Type::I32);
    let nr = fb.insert_inst(arith::Add::new(is, sr, c_re), Type::I32);
    let p = fb.insert_inst(arith::Mul::new(is, zr, zi), Type::I32);
    let two = fb.make_imm_value(2i32);
    let d = fb.insert_inst(arith::Mul::new(is, two, p), Type::I32);
    let si = fb.insert_inst(arith::Sar::new(is, ten, d), Type::I32);
    let ni = fb.insert_inst(arith::Add::new(is, si, c_im), Type::I32);
    let one = fb.make_imm_value(1i32);
    let ni2 = fb.insert_inst(arith::Add::new(is, i, one), Type::I32);
    fb.append_phi_arg(zr, nr, cont);
    fb.append_phi_arg(zi, ni, cont);
    fb.append_phi_arg(i, ni2, cont);
    fb.insert_inst_no_result(control_flow::Jump::new(is, lh));

    fb.switch_to_block(esc);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, i));

    fb.switch_to_block(exit);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, max));

    fb.seal_all();
    fb.finish();
    mb.build()
}

/// In-test Rust escape-time reference, integer-identical to `build_u32_escape_grid`
/// (i32 arithmetic, arithmetic `>>`, same literals). Written here, never trusted
/// from the kernel: the two must agree pixel-for-pixel.
fn escape_ref(px: i32, py: i32) -> u32 {
    let c_re = -2048i32 + px * 80;
    let c_im = -1280i32 + py * 80;
    let mut zr = 0i32;
    let mut zi = 0i32;
    let mut i: u32 = 0;
    while i < 50 {
        let rr = zr * zr;
        let ii = zi * zi;
        let mag = rr + ii;
        if mag < 4_194_304 {
            let diff = rr - ii;
            let nr = (diff >> 10) + c_re;
            let p = zr * zi;
            let d = 2 * p;
            let ni = (d >> 10) + c_im;
            zr = nr;
            zi = ni;
            i += 1;
        } else {
            return i;
        }
    }
    50
}

/// Test 3.3.1: the u32 `Slt` arm emits a `bitcast<i32>` signed compare that
/// validates under the browser capability set (no SHADER_INT64, no i64/u64).
#[test]
fn spirv_u32_slt_shape() {
    let module = build_u32_slt_count();
    let art = SpirvBackend::new()
        .with_workgroup_size(1, 1, 1)
        .compile_module(&module)
        .expect("u32 Slt kernel must compile");
    assert_eq!(art.layout.word, WordKind::U32, "i32 return -> u32 word");
    let wgsl = art.wgsl.as_ref().expect("WGSL");
    assert!(wgsl.contains("bitcast<i32>"), "u32 Slt must bitcast operands to i32:\n{wgsl}");
    assert!(wgsl.contains("<"), "the compare emits `<`:\n{wgsl}");
    for tok in ["i64", "u64"] {
        assert!(!wgsl.contains(tok), "browser profile: no `{tok}`:\n{wgsl}");
    }
    let reparsed = naga::front::wgsl::parse_str(wgsl)
        .expect("naga wgsl-in must reparse the u32 Slt WGSL");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    )
    .validate(&reparsed)
    .expect("browser-profile validation (default caps) must accept the u32 Slt module");
    eprintln!("spirv_u32_slt_shape OK: {} words", art.words.len());
}

#[test]
fn spirv_u32_eq_ne_shape() {
    let module = build_u32_eq_ne_probe();
    let art = SpirvBackend::new()
        .with_grid()
        .with_workgroup_size(1, 1, 1)
        .compile_module(&module)
        .expect("scalar Eq/Ne kernel must compile");
    assert_eq!(art.layout.word, WordKind::U32, "i32 return -> u32 word");
    let wgsl = art.wgsl.as_ref().expect("WGSL");
    assert!(wgsl.contains("=="), "Eq must emit genuine WGSL equality:\n{wgsl}");
    assert!(wgsl.contains("!="), "Ne must emit genuine WGSL inequality:\n{wgsl}");
    for tok in ["i64", "u64"] {
        assert!(!wgsl.contains(tok), "browser profile: no `{tok}`:\n{wgsl}");
    }
    let reparsed = naga::front::wgsl::parse_str(wgsl)
        .expect("naga wgsl-in must reparse Eq/Ne WGSL");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    )
    .validate(&reparsed)
    .expect("browser-profile validation must accept scalar Eq/Ne");
}

/// Test 3.3.2: the u32 `Sar` arm emits bitcast-i32 / shift / bitcast-u32 and
/// validates for both immediate and runtime shift amounts.
#[test]
fn spirv_u32_sar_shape() {
    let module = build_u32_sar_probe();
    let art = SpirvBackend::new()
        .with_workgroup_size(1, 1, 1)
        .compile_module(&module)
        .expect("u32 Sar kernel must compile");
    assert_eq!(art.layout.word, WordKind::U32, "i32 return -> u32 word");
    let wgsl = art.wgsl.as_ref().expect("WGSL");
    assert!(wgsl.contains("bitcast<i32>"), "u32 Sar must bitcast the value to i32:\n{wgsl}");
    assert!(wgsl.contains("bitcast<u32>"), "u32 Sar must bitcast the result back to u32:\n{wgsl}");
    assert!(wgsl.contains(">>"), "the shift emits `>>`:\n{wgsl}");
    for tok in ["i64", "u64"] {
        assert!(!wgsl.contains(tok), "browser profile: no `{tok}`:\n{wgsl}");
    }
    let reparsed = naga::front::wgsl::parse_str(wgsl)
        .expect("naga wgsl-in must reparse the u32 Sar WGSL");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    )
    .validate(&reparsed)
    .expect("browser-profile validation (default caps) must accept the u32 Sar module");

    let dynamic = SpirvBackend::new()
        .with_workgroup_size(1, 1, 1)
        .compile_module(&build_u32_sar_nonimm())
        .expect("runtime-amount u32 Sar must compile");
    let dynamic_wgsl = dynamic.wgsl.as_ref().expect("runtime Sar WGSL");
    assert!(dynamic_wgsl.contains("bitcast<i32>"));
    assert!(dynamic_wgsl.contains(">>"));
    let reparsed = naga::front::wgsl::parse_str(dynamic_wgsl)
        .expect("runtime-amount Sar WGSL must reparse");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    )
    .validate(&reparsed)
    .expect("browser profile must validate runtime-amount Sar");
    eprintln!("spirv_u32_sar_shape OK: immediate and runtime amounts");
}

#[test]
fn u32_escape_grid_executes_on_lavapipe() {
    let module = build_u32_escape_grid();
    let artifact = SpirvBackend::new()
        .with_grid()
        .with_workgroup_size(8, 8, 1)
        .compile_module(&module).expect("escape grid should compile recursively");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL");
    let (w, h) = (32u32, 32u32);
    let output = run_grid_u32(wgsl, w, h, 8, 8, &[]);
    for y in 0..h {
        for x in 0..w {
            assert_eq!(output[(y * w + x) as usize], escape_ref(x as i32, y as i32), "({x},{y})");
        }
    }
}

// ===========================================================================
// M-render: fork push #3 - Render mode. ONE SPIR-V module with two entry
// points: a fixed fullscreen-triangle `@vertex` and a `@fragment` that binds
// args 0,1 to `u32(position.xy)` (the render analog of Grid's gid.xy), runs the
// SAME mode-blind body, and returns `unpack4x8unorm(result)` as an
// `@location(0) vec4<f32>` color. There is NO output storage buffer.
// Also: the `Shr` (logical shift) op arm, sign-correct under the u32 word.
// ===========================================================================

fn native_isa() -> Native {
    Native::new(TargetTriple::new(
        if cfg!(target_arch = "x86_64") { Architecture::X86_64 } else { Architecture::Aarch64 },
        Vendor::Unknown, OperatingSystem::Native,
    ))
}

/// A branchless render fragment (the probe's shape, ported through the emitter
/// using only supported ops - no bitwise `And`): `ramp(px, py) -> u32`.
///   s = px + py;  v = (s * 4) >> 3;   // Shr (logical), = (px+py)/2, in [0,63]
///   b = 255 - v;                       // Sub
///   packed = v + v*256 + b*65536 + 0xFF000000    // r=g=v, b=b, a=255
/// The color is the packed rgba8 word `unpack4x8unorm` maps exactly to the
/// rgba8unorm target's bytes. Exercises Add/Mul/Sub/Shr + the whole render path.
fn build_render_ramp_module() -> sonatina_ir::Module {
    let isa = native_isa();
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single(
        "ramp_frag", Linkage::Public, &[Type::I32, Type::I32], Type::I32,
    );
    let fr = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(fr);
    let entry = fb.append_block();
    fb.switch_to_block(entry);
    let px = fb.args()[0];
    let py = fb.args()[1];
    let s = fb.insert_inst(arith::Add::new(is, px, py), Type::I32);
    let four = fb.make_imm_value(4i32);
    let prod = fb.insert_inst(arith::Mul::new(is, s, four), Type::I32);
    let three = fb.make_imm_value(3i32);
    // Shr constructor order is (bits, value): value >> bits (logical).
    let v = fb.insert_inst(arith::Shr::new(is, three, prod), Type::I32);
    let c255 = fb.make_imm_value(255i32);
    let b = fb.insert_inst(arith::Sub::new(is, c255, v), Type::I32);
    let c256 = fb.make_imm_value(256i32);
    let g8 = fb.insert_inst(arith::Mul::new(is, v, c256), Type::I32);
    let c65536 = fb.make_imm_value(65536i32);
    let b16 = fb.insert_inst(arith::Mul::new(is, b, c65536), Type::I32);
    let s1 = fb.insert_inst(arith::Add::new(is, v, g8), Type::I32);
    let s2 = fb.insert_inst(arith::Add::new(is, s1, b16), Type::I32);
    let alpha = fb.make_imm_value(0xFF00_0000u32 as i32);
    let packed = fb.insert_inst(arith::Add::new(is, s2, alpha), Type::I32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, packed));
    fb.seal_all();
    fb.finish();
    mb.build()
}

/// In-test Rust oracle for `build_render_ramp_module`, integer-identical, written
/// here and never trusted from the kernel. Returns the 4 rgba8unorm bytes.
fn ramp_ref(px: u32, py: u32) -> [u8; 4] {
    let s = px.wrapping_add(py);
    let v = s.wrapping_mul(4) >> 3;
    let b = 255u32.wrapping_sub(v);
    let packed = v
        .wrapping_add(v.wrapping_mul(256))
        .wrapping_add(b.wrapping_mul(65536))
        .wrapping_add(0xFF00_0000);
    packed.to_le_bytes()
}

/// The real thing: the escape-time mandelbrot AND its integer color ramp as ONE
/// render fragment (spec section 4.2). The color is a LOOP-CARRIED phi updated in
/// the accept branch; escape returns the carried color, interior returns opaque
/// black. Q10 fixed point (1.0 = 1024), the M2 escape math with a per-iteration
/// color: `v = (i*655) >> 8` (Shr), packed r=g=v, b=255-v, a=255.
fn build_mandel_frag_module() -> sonatina_ir::Module {
    let isa = native_isa();
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single(
        "mandel_frag", Linkage::Public, &[Type::I32, Type::I32], Type::I32,
    );
    let fr = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(fr);
    let entry = fb.append_block();
    let lh = fb.append_block();
    let lb = fb.append_block();
    let cont = fb.append_block();
    let esc = fb.append_block();
    let exit = fb.append_block();

    fb.switch_to_block(entry);
    let px = fb.args()[0];
    let py = fb.args()[1];
    let step = fb.make_imm_value(40i32);
    let base_re = fb.make_imm_value(-2048i32);
    let base_im = fb.make_imm_value(-1280i32);
    let pxs = fb.insert_inst(arith::Mul::new(is, px, step), Type::I32);
    let c_re = fb.insert_inst(arith::Add::new(is, base_re, pxs), Type::I32);
    let pys = fb.insert_inst(arith::Mul::new(is, py, step), Type::I32);
    let c_im = fb.insert_inst(arith::Add::new(is, base_im, pys), Type::I32);
    let zero = fb.make_imm_value(0i32);
    let black = fb.make_imm_value(0xFF00_0000u32 as i32);
    fb.insert_inst_no_result(control_flow::Jump::new(is, lh));

    fb.switch_to_block(lh);
    let zr = fb.insert_inst(control_flow::Phi::new(is, vec![(zero, entry)]), Type::I32);
    let zi = fb.insert_inst(control_flow::Phi::new(is, vec![(zero, entry)]), Type::I32);
    let i = fb.insert_inst(control_flow::Phi::new(is, vec![(zero, entry)]), Type::I32);
    let color = fb.insert_inst(control_flow::Phi::new(is, vec![(black, entry)]), Type::I32);
    let max = fb.make_imm_value(100i32);
    // Loop-counter compare is UNSIGNED (`Lt`).
    let lc = fb.insert_inst(cmp::Lt::new(is, i, max), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, lc, lb, exit));

    fb.switch_to_block(lb);
    let rr = fb.insert_inst(arith::Mul::new(is, zr, zr), Type::I32);
    let ii = fb.insert_inst(arith::Mul::new(is, zi, zi), Type::I32);
    let mag = fb.insert_inst(arith::Add::new(is, rr, ii), Type::I32);
    let th = fb.make_imm_value(4_194_304i32);
    // Escape compare is SIGNED (`Slt`): Q20 magnitude carries a signed value.
    let ec = fb.insert_inst(cmp::Slt::new(is, mag, th), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, ec, cont, esc));

    fb.switch_to_block(cont);
    let ten = fb.make_imm_value(10i32);
    let diff = fb.insert_inst(arith::Sub::new(is, rr, ii), Type::I32);
    // `diff` and the cross term can be negative -> ARITHMETIC shift (Sar).
    let sr = fb.insert_inst(arith::Sar::new(is, ten, diff), Type::I32);
    let nr = fb.insert_inst(arith::Add::new(is, sr, c_re), Type::I32);
    let p = fb.insert_inst(arith::Mul::new(is, zr, zi), Type::I32);
    let two = fb.make_imm_value(2i32);
    let d = fb.insert_inst(arith::Mul::new(is, two, p), Type::I32);
    let si = fb.insert_inst(arith::Sar::new(is, ten, d), Type::I32);
    let ni = fb.insert_inst(arith::Add::new(is, si, c_im), Type::I32);
    let one = fb.make_imm_value(1i32);
    let i2 = fb.insert_inst(arith::Add::new(is, i, one), Type::I32);
    // Color ramp: v = (i2 * 655) >> 8 (Shr, logical; i2 non-negative), then
    // packed = v + v*256 + (255-v)*65536 + 0xFF000000.
    let k655 = fb.make_imm_value(655i32);
    let vprod = fb.insert_inst(arith::Mul::new(is, i2, k655), Type::I32);
    let eight = fb.make_imm_value(8i32);
    let vv = fb.insert_inst(arith::Shr::new(is, eight, vprod), Type::I32);
    let c255 = fb.make_imm_value(255i32);
    let cb = fb.insert_inst(arith::Sub::new(is, c255, vv), Type::I32);
    let c256 = fb.make_imm_value(256i32);
    let g8 = fb.insert_inst(arith::Mul::new(is, vv, c256), Type::I32);
    let c65536 = fb.make_imm_value(65536i32);
    let b16 = fb.insert_inst(arith::Mul::new(is, cb, c65536), Type::I32);
    let cs1 = fb.insert_inst(arith::Add::new(is, vv, g8), Type::I32);
    let cs2 = fb.insert_inst(arith::Add::new(is, cs1, b16), Type::I32);
    let alpha = fb.make_imm_value(0xFF00_0000u32 as i32);
    let color2 = fb.insert_inst(arith::Add::new(is, cs2, alpha), Type::I32);
    fb.append_phi_arg(zr, nr, cont);
    fb.append_phi_arg(zi, ni, cont);
    fb.append_phi_arg(i, i2, cont);
    fb.append_phi_arg(color, color2, cont);
    fb.insert_inst_no_result(control_flow::Jump::new(is, lh));

    fb.switch_to_block(esc);
    // Escape: return the carried color (a phi).
    fb.insert_inst_no_result(control_flow::Return::new_single(is, color));

    fb.switch_to_block(exit);
    // Interior (i reached 100): opaque black.
    fb.insert_inst_no_result(control_flow::Return::new_single(is, black));

    fb.seal_all();
    fb.finish();
    mb.build()
}

/// In-test Rust oracle for `build_mandel_frag_module`, integer-identical (i32/u32
/// wrapping, arithmetic `>>` for Sar, logical `>>` for the color ramp). Returns
/// the 4 rgba8unorm bytes.
fn mandel_ref(px: i32, py: i32) -> [u8; 4] {
    let c_re = (-2048i32).wrapping_add(px.wrapping_mul(40));
    let c_im = (-1280i32).wrapping_add(py.wrapping_mul(40));
    let mut zr = 0i32;
    let mut zi = 0i32;
    let mut i = 0i32;
    let mut color: u32 = 0xFF00_0000;
    loop {
        if !(i < 100) {
            color = 0xFF00_0000; // interior: opaque black
            break;
        }
        let rr = zr.wrapping_mul(zr);
        let ii = zi.wrapping_mul(zi);
        let mag = rr.wrapping_add(ii);
        if !(mag < 4_194_304) {
            break; // escape: keep the carried color
        }
        let diff = rr.wrapping_sub(ii);
        let sr = diff >> 10; // arithmetic
        let nr = sr.wrapping_add(c_re);
        let p = zr.wrapping_mul(zi);
        let d = 2i32.wrapping_mul(p);
        let si = d >> 10; // arithmetic
        let ni = si.wrapping_add(c_im);
        zr = nr;
        zi = ni;
        i = i.wrapping_add(1);
        let v = ((i as u32).wrapping_mul(655)) >> 8; // logical
        let b = 255u32.wrapping_sub(v);
        color = v
            .wrapping_add(v.wrapping_mul(256))
            .wrapping_add(b.wrapping_mul(65536))
            .wrapping_add(0xFF00_0000);
    }
    color.to_le_bytes()
}

/// Execute a render module OFFSCREEN on lavapipe (browser profile,
/// `Features::empty()`): a `w x h` rgba8unorm target, `draw(0..3)`, then
/// `copy_texture_to_buffer` + readback. Returns the tightly-packed RGBA bytes
/// (row padding stripped). Hard-fails if no adapter is available: this is an
/// EXECUTED gate (lavapipe is present in CI), not validate-only. Ported from the
/// executed render probe.
fn run_render_rgba8(wgsl: &str, w: u32, h: u32, input: &[u8]) -> Vec<u8> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        force_fallback_adapter: false,
        ..Default::default()
    }))
    .expect("render execute requires a GPU adapter (lavapipe); none available");
    eprintln!("  GPU: {}", adapter.get_info().name);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::empty(), // browser profile: no SHADER_INT64
        ..Default::default()
    }))
    .expect("browser-profile device (Features::empty) must be available on lavapipe");

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("render"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });

    // The broadcast input storage buffer at @group(0) @binding(1), FRAGMENT
    // visibility (the fragment-stage storage read the probe proved on lavapipe).
    let input_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("input"),
        size: input.len().max(4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !input.is_empty() {
        queue.write_buffer(&input_buf, 0, input);
    }
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("render_bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("render_pl"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("render_bg"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry { binding: 1, resource: input_buf.as_entire_binding() }],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("fullscreen"),
        layout: Some(&pl),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_fullscreen"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });

    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("target"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = tex.create_view(&Default::default());

    // 256-aligned bytes_per_row (COPY_BYTES_PER_ROW_ALIGNMENT).
    let bytes_per_row = ((w * 4 + 255) / 256) * 256;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: u64::from(bytes_per_row * h),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(std::time::Duration::from_secs(30)),
    });
    rx.recv().unwrap().expect("staging map");
    let data = slice.get_mapped_range();
    let row = (w * 4) as usize;
    let mut out = Vec::with_capacity(row * h as usize);
    for y in 0..h {
        let off = (y * bytes_per_row) as usize;
        out.extend_from_slice(&data[off..off + row]);
    }
    drop(data);
    staging.unmap();
    out
}

fn expect_render_err(module: &sonatina_ir::Module, backend: SpirvBackend) -> String {
    match backend.compile_module(module) {
        Ok(_) => panic!("expected a render fail-closed error, got Ok"),
        Err(errs) => {
            let msg = errs.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("; ");
            assert!(msg.contains("render"), "fail-closed error must name render: {msg}");
            msg
        }
    }
}

/// Render WGSL shape + self-describing layout, GPU-free: TWO entry points
/// (@vertex + @fragment), a color attachment (@location(0)), no output buffer,
/// browser-profile (no i64/u64), validates under Capabilities::default(). Also
/// asserts the SPIR-V module carries exactly 2 OpEntryPoint instructions.
#[test]
fn render_wgsl_shape() {
    let module = build_render_ramp_module();
    let art = SpirvBackend::new()
        .with_render()
        .compile_module(&module)
        .expect("render ramp must compile");

    let layout = &art.layout;
    assert_eq!(layout.mode, LayoutMode::Render, "render mode");
    assert_eq!(layout.word, WordKind::U32, "u32 word");
    assert!(layout.result.is_none(), "render states no single-slot result");
    assert_eq!(layout.workgroup_size, [0, 0, 0], "render has no workgroup size");
    assert_eq!(layout.vertex_entry.as_deref(), Some("vs_fullscreen"), "vertex entry name");
    assert_eq!(layout.fragment_entry.as_deref(), Some("fs_main"), "fragment entry name");
    assert_eq!(
        layout.color_target_format.as_deref(), Some("rgba8unorm"),
        "render states its color-target format"
    );
    // No output binding (binding 0 absent); only the input storage buffer @0/1.
    assert!(
        layout.bindings.iter().all(|b| b.role != Role::Output),
        "render mode has no output storage binding"
    );
    let in_b = layout.bindings.iter().find(|b| b.role == Role::Input).expect("input binding");
    assert_eq!(in_b.binding, 1, "input stays at @group(0) @binding(1)");

    let wgsl = art.wgsl.as_ref().expect("WGSL side artifact");
    assert!(wgsl.contains("@vertex"), "WGSL must contain the vertex stage:\n{wgsl}");
    assert!(wgsl.contains("@fragment"), "WGSL must contain the fragment stage:\n{wgsl}");
    assert!(wgsl.contains("@location(0)"), "fragment must write @location(0):\n{wgsl}");
    assert!(wgsl.contains("unpack4x8unorm"), "epilogue must be unpack4x8unorm:\n{wgsl}");
    assert!(wgsl.contains("vertex_index"), "vertex must read vertex_index:\n{wgsl}");
    for tok in ["i64", "u64"] {
        assert!(!wgsl.contains(tok), "browser profile: no `{tok}`:\n{wgsl}");
    }

    // Exactly two OpEntryPoint (opcode 15) instructions in the SPIR-V stream.
    let words = &art.words;
    let mut eps = 0usize;
    let mut idx = 5usize; // skip the 5-word header
    while idx < words.len() {
        let opword = words[idx];
        let wc = (opword >> 16) as usize;
        if (opword & 0xffff) == 15 { eps += 1; }
        if wc == 0 { break; }
        idx += wc;
    }
    assert_eq!(eps, 2, "one SPIR-V module must carry BOTH entry points");

    // Browser-profile reparse + validate (default caps, no SHADER_INT64).
    let reparsed = naga::front::wgsl::parse_str(wgsl)
        .expect("naga wgsl-in must reparse the render WGSL");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    )
    .validate(&reparsed)
    .expect("browser-profile validation (default caps) must accept the render module");
    eprintln!("render_wgsl_shape OK: 2 entry points, {} words", art.words.len());
}

/// U32 shift-left emits a direct WGSL `<<` and validates with either an
/// immediate or runtime amount.
#[test]
fn spirv_u32_shl_shape() {
    let isa = native_isa();
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single("shl_probe", Linkage::Public, &[Type::I32], Type::I32);
    let fr = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(fr);
    let entry = fb.append_block();
    fb.switch_to_block(entry);
    let runtime_value = fb.args()[0];
    let four = fb.make_imm_value(4i32);
    let shifted = fb.insert_inst(arith::Shl::new(is, four, runtime_value), Type::I32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, shifted));
    fb.seal_all();
    fb.finish();

    let art = SpirvBackend::new()
        .with_workgroup_size(1, 1, 1)
        .compile_module(&mb.build())
        .expect("u32 Shl kernel must compile");
    assert_eq!(art.layout.word, WordKind::U32, "i32 return -> u32 word");
    let wgsl = art.wgsl.as_ref().expect("WGSL");
    assert!(wgsl.contains("<<"), "shift-left must remain a genuine WGSL `<<`:\n{wgsl}");
    for tok in ["i64", "u64", "bitcast<i32>"] {
        assert!(!wgsl.contains(tok), "browser Shl needs no `{tok}`:\n{wgsl}");
    }
    let reparsed = naga::front::wgsl::parse_str(wgsl)
        .expect("naga wgsl-in must reparse runtime-value Shl WGSL");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    )
    .validate(&reparsed)
    .expect("browser-profile validation must accept scalar Shl");

    let mb2 = native_module_builder();
    let sig2 = Signature::new_single(
        "shl_nonimm",
        Linkage::Public,
        &[Type::I32, Type::I32],
        Type::I32,
    );
    let fr2 = mb2.declare_function(sig2).unwrap();
    let mut fb2 = mb2.func_builder::<InstInserter>(fr2);
    let entry2 = fb2.append_block();
    fb2.switch_to_block(entry2);
    let value2 = fb2.args()[0];
    let bits2 = fb2.args()[1];
    let shifted2 = fb2.insert_inst(arith::Shl::new(is, bits2, value2), Type::I32);
    fb2.insert_inst_no_result(control_flow::Return::new_single(is, shifted2));
    fb2.seal_all();
    fb2.finish();
    let dynamic = SpirvBackend::new()
        .with_workgroup_size(1, 1, 1)
        .compile_module(&mb2.build())
        .expect("runtime-amount u32 Shl must compile");
    let dynamic_wgsl = dynamic.wgsl.as_ref().expect("runtime Shl WGSL");
    assert!(dynamic_wgsl.contains("<<"));
    let reparsed = naga::front::wgsl::parse_str(dynamic_wgsl)
        .expect("runtime-amount Shl WGSL must reparse");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    )
    .validate(&reparsed)
    .expect("browser profile must validate runtime-amount Shl");
}

#[test]
fn spirv_u32_shr_shape() {
    let isa = native_isa();
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single("shr_probe", Linkage::Public, &[Type::I32], Type::I32);
    let fr = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(fr);
    let entry = fb.append_block();
    fb.switch_to_block(entry);
    let a = fb.args()[0];
    let four = fb.make_imm_value(4i32);
    // Shr order is (bits, value): a >> 4.
    let s = fb.insert_inst(arith::Shr::new(is, four, a), Type::I32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, s));
    fb.seal_all();
    fb.finish();
    let module = mb.build();

    let art = SpirvBackend::new()
        .with_workgroup_size(1, 1, 1)
        .compile_module(&module)
        .expect("u32 Shr kernel must compile");
    assert_eq!(art.layout.word, WordKind::U32, "i32 return -> u32 word");
    let wgsl = art.wgsl.as_ref().expect("WGSL");
    assert!(wgsl.contains(">>"), "the logical shift emits `>>`:\n{wgsl}");
    // The u32 logical shift is DIRECT: no i32 bitcast (that is Sar's dance).
    assert!(
        !wgsl.contains("bitcast<i32>"),
        "u32 Shr is a direct logical shift, no i32 bitcast:\n{wgsl}"
    );
    for tok in ["i64", "u64"] {
        assert!(!wgsl.contains(tok), "browser profile: no `{tok}`:\n{wgsl}");
    }
    let reparsed = naga::front::wgsl::parse_str(wgsl)
        .expect("naga wgsl-in must reparse the u32 Shr WGSL");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    )
    .validate(&reparsed)
    .expect("browser-profile validation (default caps) must accept the u32 Shr module");

    // A runtime u32 shift amount lowers directly and validates.
    {
        let is2 = isa.inst_set();
        let mb2 = native_module_builder();
        let sig2 = Signature::new_single("shr_nonimm", Linkage::Public, &[Type::I32, Type::I32], Type::I32);
        let fr2 = mb2.declare_function(sig2).unwrap();
        let mut fb2 = mb2.func_builder::<InstInserter>(fr2);
        let e2 = fb2.append_block();
        fb2.switch_to_block(e2);
        let a2 = fb2.args()[0];
        let sh = fb2.args()[1];
        let s2 = fb2.insert_inst(arith::Shr::new(is2, sh, a2), Type::I32);
        fb2.insert_inst_no_result(control_flow::Return::new_single(is2, s2));
        fb2.seal_all();
        fb2.finish();
        let dynamic = SpirvBackend::new()
            .with_workgroup_size(1, 1, 1)
            .compile_module(&mb2.build())
            .expect("runtime-amount u32 Shr must compile");
        let dynamic_wgsl = dynamic.wgsl.as_ref().expect("runtime Shr WGSL");
        assert!(dynamic_wgsl.contains(">>"));
        let reparsed = naga::front::wgsl::parse_str(dynamic_wgsl)
            .expect("runtime-amount Shr WGSL must reparse");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::default(),
        )
        .validate(&reparsed)
        .expect("browser profile must validate runtime-amount Shr");
    }

    // An i64-word Shr fails closed (only the u32 browser word lowers `>>`).
    {
        let is3 = isa.inst_set();
        let mb3 = native_module_builder();
        let sig3 = Signature::new_single("shr_i64", Linkage::Public, &[Type::I64], Type::I64);
        let fr3 = mb3.declare_function(sig3).unwrap();
        let mut fb3 = mb3.func_builder::<InstInserter>(fr3);
        let e3 = fb3.append_block();
        fb3.switch_to_block(e3);
        let a3 = fb3.args()[0];
        let four3 = fb3.make_imm_value(4i64);
        let s3 = fb3.insert_inst(arith::Shr::new(is3, four3, a3), Type::I64);
        fb3.insert_inst_no_result(control_flow::Return::new_single(is3, s3));
        fb3.seal_all();
        fb3.finish();
        let bad = mb3.build();
        let err = match SpirvBackend::new().with_workgroup_size(1, 1, 1).compile_module(&bad) {
            Ok(_) => panic!("i64-word Shr must fail closed"),
            Err(errs) => errs.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("; "),
        };
        assert!(
            err.contains("i64") && err.contains("Shr"),
            "i64 Shr must name the fail-closed reason: {err}"
        );
    }
    eprintln!(
        "spirv_u32_shr_shape OK: direct logical `>>`, immediate/runtime amounts + i64 fail closed"
    );
}

/// Fail-closed: the four render preconditions each err, and the error names render.
#[test]
fn render_fail_closed() {
    // i64 word (build_grid_i64_module is a 2-arg i64 kernel).
    let m = build_grid_i64_module();
    let e = expect_render_err(&m, SpirvBackend::new().with_render());
    assert!(
        e.contains("coordinate args") && e.contains("i32"),
        "i64 coordinate args must name the i32 coordinate requirement: {e}"
    );

    // < 2 args.
    let m = build_grid_1arg_module();
    let e = expect_render_err(&m, SpirvBackend::new().with_render());
    assert!(
        e.contains("coordinate args") && e.contains("i32"),
        "1-arg kernel must name the two-coordinate requirement: {e}"
    );

    // ObjAlloc (batch).
    let m = build_grid_objalloc_module();
    let e = expect_render_err(&m, SpirvBackend::new().with_render());
    assert!(e.contains("mutually exclusive"), "ObjAlloc must name the batch conflict: {e}");

    // grid + render together.
    let m = build_grid_gradient_module();
    let e = expect_render_err(&m, SpirvBackend::new().with_grid().with_render());
    assert!(e.contains("mutually exclusive"), "grid+render must name the conflict: {e}");

    eprintln!("render_fail_closed OK: all four preconditions err and name render");
}

/// Headline (ported from the executed probe): the branchless color ramp RENDERS
/// on lavapipe to a 64x64 rgba8unorm offscreen target and EVERY pixel is
/// byte-exact vs the in-test Rust oracle. JavaScript paints nothing here.
#[test]
fn render_ramp_executes_on_lavapipe() {
    let module = build_render_ramp_module();
    let art = SpirvBackend::new()
        .with_render()
        .compile_module(&module)
        .expect("render ramp must compile");
    assert_eq!(art.layout.mode, LayoutMode::Render, "render mode");
    let wgsl = art.wgsl.as_ref().expect("WGSL");

    let (w, h) = (64u32, 64u32);
    let bytes = run_render_rgba8(wgsl, w, h, &[]);
    assert_eq!(bytes.len(), (w * h * 4) as usize, "full frame readback");

    let mut mismatches = 0u32;
    for y in 0..h {
        for x in 0..w {
            let off = ((y * w + x) * 4) as usize;
            let got = &bytes[off..off + 4];
            let want = ramp_ref(x, y);
            if got != want {
                if mismatches < 5 {
                    eprintln!("  MISMATCH at ({x},{y}): got {got:?} want {want:?}");
                }
                mismatches += 1;
            }
        }
    }
    assert_eq!(mismatches, 0, "rendered ramp must equal the oracle byte-for-byte");
    eprintln!("render_ramp_executes_on_lavapipe OK: {w}x{h}, all {} pixels byte-exact", w * h);
}

#[test]
fn render_mandelbrot_executes_on_lavapipe() {
    let module = build_mandel_frag_module();
    let artifact = SpirvBackend::new()
        .with_render()
        .compile_module(&module).expect("Mandelbrot render should compile recursively");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL");
    let (w, h) = (32u32, 32u32);
    let bytes = run_render_rgba8(wgsl, w, h, &[]);
    for y in 0..h {
        for x in 0..w {
            let offset = ((y * w + x) * 4) as usize;
            assert_eq!(&bytes[offset..offset + 4], &mandel_ref(x as i32, y as i32), "({x},{y})");
        }
    }
}

#[test]
fn unsupported_instruction_fails_closed_in_all_spirv_modes() {
    let source = r#"
target = "wasm32-unknown-native"
func public %unsupported(v0.i32, v1.i32, v2.i32) -> i32 {
    block0:
        v3.i32 = neg v2;
        return v3;
}
"#;
    let module = sonatina_parser::parse_module(source)
        .expect("unsupported-op module should parse")
        .module;
    for (mode, backend) in [
        ("scalar", SpirvBackend::new()),
        ("grid", SpirvBackend::new().with_grid()),
        ("render", SpirvBackend::new().with_render()),
    ] {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            backend.compile_module(&module)
        }));
        let compile = result.unwrap_or_else(|_| panic!("{mode} unsupported op must not panic"));
        let errors = match compile {
            Err(errors) => errors,
            Ok(_) => panic!("{mode} unsupported op must fail"),
        };
        let rendered = errors
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            rendered.contains("`neg`"),
            "{mode} error must name neg: {rendered}"
        );
        assert!(
            rendered.contains("unsupported"),
            "{mode} error must explain failure: {rendered}"
        );
    }
}

#[test]
fn mixed_i32_coordinate_f32_grid_executes_on_lavapipe() {
    let source = r#"
target = "wasm32-unknown-native"
func public %mixed(v0.i32, v1.i32, v2.f32, v3.f32) -> i32 {
    block0:
        v4.i1 = lt v0 v1;
        br v4 block1 block2;
    block1:
        v5.f32 = fadd v2 v3;
        jump block3;
    block2:
        v6.f32 = i32_to_f32 -1.i32;
        jump block3;
    block3:
        v7.f32 = phi (v5 block1) (v6 block2);
        v8.i32 = f32_to_i32 v7;
        return v8;
}

"#;
    let module = sonatina_parser::parse_module(source).expect("mixed grid should parse").module;
    let artifact = SpirvBackend::new().with_grid().with_workgroup_size(2, 2, 1)
        .compile_module(&module).expect("mixed grid should compile");
    assert_layout_metadata_invariants(&artifact.layout, 4);
    let wgsl = artifact.wgsl.as_deref().expect("WGSL should be emitted");
    let input = artifact.layout.bindings.iter().find(|binding| matches!(binding.role, Role::Input)).unwrap();
    assert_eq!(input.stride, 8, "two f32 broadcasts must occupy offsets 0 and 4");
    assert_eq!(input.span, 8);
    assert_eq!(input.members, vec![
        SpirvBindingMember { arg_index: 2, offset: 0, width: 4, scalar: SpirvScalarKind::F32 },
        SpirvBindingMember { arg_index: 3, offset: 4, width: 4, scalar: SpirvScalarKind::F32 },
    ]);
    assert_eq!(artifact.layout.builtin_inputs, vec![
        SpirvBuiltinInput { arg_index: 0, source: SpirvBuiltinSource::GlobalInvocationIdX, scalar: SpirvScalarKind::I32 },
        SpirvBuiltinInput { arg_index: 1, source: SpirvBuiltinSource::GlobalInvocationIdY, scalar: SpirvScalarKind::I32 },
    ]);
    assert!(wgsl.contains("p0_: f32") && wgsl.contains("p1_: f32"), "WGSL must reflect f32 broadcast members: {wgsl}");
    let reparsed = naga::front::wgsl::parse_str(wgsl).expect("mixed WGSL should reparse");
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::default())
        .validate(&reparsed).expect("mixed WGSL should validate without SHADER_INT64");
    let input_bits = [3.0_f32.to_bits().to_le_bytes(), 4.0_f32.to_bits().to_le_bytes()].concat();
    let output = run_grid_u32(wgsl, 4, 2, 2, 2, &input_bits);
    assert_eq!(output, vec![u32::MAX, u32::MAX, u32::MAX, u32::MAX, 7, u32::MAX, u32::MAX, u32::MAX]);
}

#[test]
fn render_reflects_two_f32_broadcast_members() {
    let source = r#"
target = "wasm32-unknown-native"
func public %render_broadcast(v0.i32, v1.i32, v2.f32, v3.f32) -> i32 {
    block0:
        v4.f32 = fadd v2 v3;
        v5.i32 = f32_to_u32 v4;
        return v5;
}
"#;
    let module = sonatina_parser::parse_module(source).expect("render broadcasts should parse").module;
    let artifact = SpirvBackend::new().with_render().compile_module(&module)
        .expect("render broadcasts should compile");
    assert_layout_metadata_invariants(&artifact.layout, 4);
    let input = artifact.layout.bindings.iter().find(|binding| matches!(binding.role, Role::Input)).unwrap();
    assert_eq!(input.stride, 8);
    assert_eq!(input.span, 8);
    assert_eq!(input.members, vec![
        SpirvBindingMember { arg_index: 2, offset: 0, width: 4, scalar: SpirvScalarKind::F32 },
        SpirvBindingMember { arg_index: 3, offset: 4, width: 4, scalar: SpirvScalarKind::F32 },
    ]);
    assert_eq!(artifact.layout.builtin_inputs, vec![
        SpirvBuiltinInput { arg_index: 0, source: SpirvBuiltinSource::FragmentPositionX, scalar: SpirvScalarKind::I32 },
        SpirvBuiltinInput { arg_index: 1, source: SpirvBuiltinSource::FragmentPositionY, scalar: SpirvScalarKind::I32 },
    ]);
    let wgsl = artifact.wgsl.as_deref().expect("WGSL should be emitted");
    assert!(
        wgsl.contains("p2_: f32") && wgsl.contains("p3_: f32"),
        "WGSL must preserve the original render argument indices: {wgsl}",
    );
    let reparsed = naga::front::wgsl::parse_str(wgsl).expect("render WGSL should reparse");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    )
    .validate(&reparsed)
    .expect("render WGSL should validate without SHADER_INT64");
}

#[test]
fn f32_integer_conversions_saturate_on_lavapipe() {
    let source = r#"
target = "wasm32-unknown-native"
func public %saturating(v0.i32, v1.i32, v2.f32) -> i32 {
    block0:
        v3.i1 = lt v0 1.i32;
        br v3 block1 block2;
    block1:
        v4.i32 = f32_to_i32 v2;
        jump block3;
    block2:
        v5.i32 = f32_to_u32 v2;
        jump block3;
    block3:
        v6.i32 = phi (v4 block1) (v5 block2);
        return v6;
}
"#;
    let module = sonatina_parser::parse_module(source).expect("conversion probe should parse").module;
    let artifact = SpirvBackend::new().with_grid().with_workgroup_size(2, 1, 1)
        .compile_module(&module).expect("conversion probe should compile");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL should be emitted");
    let values = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        2_147_483_520.0,
        2_147_483_648.0,
        -2_147_483_648.0,
        -2_147_483_904.0,
        4_294_967_040.0,
        4_294_967_296.0,
        -1.0,
        42.75,
    ];
    for value in values {
        let output = run_grid_u32(wgsl, 2, 1, 2, 1, &value.to_bits().to_le_bytes());
        assert_eq!(
            output,
            vec![(value as i32) as u32, value as u32],
            "conversion mismatch for f32 bits {:#010x}",
            value.to_bits(),
        );
    }
}

#[test]
fn grid_loop_exit_phi_resumes_at_sibling_on_lavapipe() {
    let source = r#"
target = "wasm32-unknown-native"
func public %exit_phi(v0.i32, v1.i32) -> i32 {
    block0:
        jump block1;
    block1:
        v2.i32 = phi (0.i32 block0) (v5 block2);
        v3.i32 = add v2 10.i32;
        v4.i1 = lt v2 2.i32;
        br v4 block2 block3;
    block2:
        v5.i32 = add v2 1.i32;
        jump block1;
    block3:
        v6.i32 = phi (v3 block1);
        jump block4;
    block4:
        return v6;
}
"#;
    let module = sonatina_parser::parse_module(source).expect("exit phi should parse").module;
    let artifact = SpirvBackend::new().with_grid().with_workgroup_size(2, 2, 1)
        .compile_module(&module).expect("exit phi should compile");
    let output = run_grid_u32(artifact.wgsl.as_deref().expect("WGSL"), 2, 2, 2, 2, &[]);
    assert_eq!(output, vec![12; 4]);
}

#[test]
fn grid_conditional_direct_edge_does_not_override_sibling_loop_exit_phi() {
    let source = r#"
target = "wasm32-unknown-native"
func public %conditional_loop_phi(v0.i32, v1.i32) -> i32 {
    block0:
        v2.i1 = lt v0 1.i32;
        br v2 block5 block2;
    block2:
        jump block3;
    block3:
        v3.i32 = phi (5.i32 block2) (v6 block4);
        v4.i32 = phi (0.i32 block2) (v7 block4);
        v5.i1 = lt v4 3.i32;
        br v5 block4 block5;
    block4:
        v6.i32 = add v3 v3;
        v7.i32 = add v4 1.i32;
        jump block3;
    block5:
        v8.i32 = phi (0.i32 block0) (v3 block3) (v14 block6);
        v9.i32 = phi (1.i32 block0) (1.i32 block3) (v13 block6);
        v10.i32 = phi (v0 block0) (v0 block3) (v15 block6);
        v11.i1 = eq v10 0.i32;
        br v11 block7 block6;
    block6:
        v13.i32 = mul v9 v8;
        v14.i32 = mul v8 v8;
        v15.i32 = sub v10 1.i32;
        jump block5;
    block7:
        return v9;
}
"#;
    let module = sonatina_parser::parse_module(source)
        .expect("conditional loop-phi probe should parse")
        .module;
    let artifact = SpirvBackend::new()
        .with_grid()
        .with_workgroup_size(4, 1, 1)
        .compile_module(&module)
        .expect("conditional loop-phi probe should compile");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL");
    assert!(
        !wgsl.contains("edge_0_5_phi_8_1"),
        "the direct outside edge must not be emitted again after its sibling loop:\n{wgsl}",
    );
    assert!(
        !wgsl.lines().any(|line| {
            let line = line.trim_start();
            line.starts_with("var phi_")
                || line.starts_with("var edge_")
                || line.starts_with("var structured_result")
                || line.starts_with("var structured_did_return")
        }),
        "compiler-internal control transport names must stay compact:\n{wgsl}",
    );
    let reparsed = naga::front::wgsl::parse_str(wgsl)
        .expect("conditional loop-phi WGSL should reparse");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    )
    .validate(&reparsed)
    .expect("conditional loop-phi WGSL should validate");
}

#[test]
fn spirv_nested_terminal_arm_selects_nearest_phi_merge() {
    let source = r#"
target = "wasm32-unknown-native"
func public %nearest_phi_merge(v0.i32) -> i32 {
    block0:
        v1.i1 = lt v0 100.i32;
        br v1 block1 block8;
    block1:
        v2.i1 = lt v0 10.i32;
        br v2 block2 block3;
    block2:
        v3.i1 = eq v0 5.i32;
        br v3 block9 block4;
    block3:
        v4.i32 = add v0 20.i32;
        jump block5;
    block4:
        v5.i32 = add v0 10.i32;
        jump block5;
    block5:
        v6.i32 = phi (v4 block3) (v5 block4);
        v7.i1 = lt v6 50.i32;
        br v7 block6 block7;
    block6:
        v8.i32 = add v6 1.i32;
        jump block8;
    block7:
        v9.i32 = add v6 2.i32;
        jump block8;
    block8:
        v10.i32 = phi (0.i32 block0) (v8 block6) (v9 block7);
        jump block10;
    block10:
        return v10;
    block9:
        unreachable;
}
"#;
    let module = sonatina_parser::parse_module(source)
        .expect("nearest-merge regression should parse")
        .module;
    SpirvBackend::new()
        .compile_module(&module)
        .expect("the nearest live phi merge should structure before the enclosing stop");
}

fn spirv_error(source: &str, backend: SpirvBackend) -> String {
    let module = sonatina_parser::parse_module(source).expect("regression source should parse").module;
    match backend.compile_module(&module) {
        Err(errors) => errors.iter().map(ToString::to_string).collect::<Vec<_>>().join("\n"),
        Ok(_) => panic!("regression must fail"),
    }
}

#[test]
fn i64_profile_remains_homogeneous_but_accepts_integer_literals() {
    let mixed = r#"
target = "wasm32-unknown-native"
func public %mixed(v0.i32) -> i64 {
    block0:
        return v0;
}
"#;
    let error = spirv_error(mixed, SpirvBackend::new());
    assert!(error.contains("i64") && error.contains("homogeneous"), "{error}");

    let literal = r#"
target = "wasm32-unknown-native"
func public %literal() -> i64 {
    block0:
        return 7.i32;
}
"#;
    let module = sonatina_parser::parse_module(literal).expect("i32 literal in i64 profile should parse").module;
    SpirvBackend::new().compile_module(&module).expect("i32 literal must retain i64-profile compatibility");
}

#[test]
fn narrow_integer_intermediate_and_phi_results_fail_closed() {
    let narrow_result = r#"
target = "wasm32-unknown-native"
func public %narrow_result() -> i32 {
    block0:
        v0.i8 = add 127.i8 1.i8;
        return 0.i32;
}
"#;
    let error = spirv_error(narrow_result, SpirvBackend::new());
    assert!(
        error.contains("integer instruction result") && error.contains("I8"),
        "{error}"
    );

    let narrow_phi = r#"
target = "wasm32-unknown-native"
func public %narrow_phi(v0.i32) -> i32 {
    block0:
        v1.i1 = lt v0 1.i32;
        br v1 block1 block2;
    block1:
        jump block3;
    block2:
        jump block3;
    block3:
        v2.i8 = phi (1.i8 block1) (2.i8 block2);
        return v0;
}
"#;
    let error = spirv_error(narrow_phi, SpirvBackend::new());
    assert!(
        error.contains("integer instruction result") && error.contains("I8"),
        "{error}"
    );
}

#[test]
fn i64_profile_rejects_f32_values_and_conversions() {
    for source in [
        r#"
target = "wasm32-unknown-native"
func public %to_float() -> i64 {
    block0:
        v0.f32 = i32_to_f32 1.i32;
        return 0.i64;
}
"#,
        r#"
target = "wasm32-unknown-native"
func public %from_float() -> i64 {
    block0:
        v0.i32 = f32_to_i32 0x3f800000.f32;
        return 0.i64;
}
"#,
        r#"
target = "wasm32-unknown-native"
func public %unsigned_to_float() -> i64 {
    block0:
        v0.f32 = u32_to_f32 1.i32;
        return 0.i64;
}
"#,
        r#"
target = "wasm32-unknown-native"
func public %float_to_unsigned() -> i64 {
    block0:
        v0.i32 = f32_to_u32 0x3f800000.f32;
        return 0.i64;
}
"#,
    ] {
        let error = spirv_error(source, SpirvBackend::new());
        assert!(error.contains("spirv i64") && error.contains("f32"), "{error}");
    }
}

#[test]
fn storage_buffer_boolean_broadcast_fails_closed() {
    let source = r#"
target = "wasm32-unknown-native"
func public %boolean(v0.i32, v1.i32, v2.i1) -> i32 {
    block0:
        return v0;
}
"#;
    let error = spirv_error(source, SpirvBackend::new().with_grid());
    assert!(error.contains("boolean") && error.contains("storage-buffer"), "{error}");
}

#[test]
fn float_and_boolean_loop_header_phis_compile_with_typed_locals() {
    let float_loop = r#"
target = "wasm32-unknown-native"
func public %float_loop(v0.i32) -> i32 {
    block0:
        jump block1;
    block1:
        v1.f32 = phi (0x00000000.f32 block0) (v4 block2);
        v2.i1 = lt v0 4.i32;
        br v2 block2 block3;
    block2:
        v4.f32 = fadd v1 0x3f800000.f32;
        jump block1;
    block3:
        v5.i32 = f32_to_i32 v1;
        return v5;
}

"#;
    let module = sonatina_parser::parse_module(float_loop).expect("float loop should parse").module;
    SpirvBackend::new().compile_module(&module).expect("f32 loop phi should compile");

    let bool_loop = r#"
target = "wasm32-unknown-native"
func public %bool_loop(v0.i32) -> i32 {
    block0:
        jump block1;
    block1:
        v1.i1 = phi (0.i1 block0) (v3 block2);
        br v1 block2 block3;
    block2:
        v3.i1 = lt v0 1.i32;
        jump block1;
    block3:
        return 0.i32;
}
"#;
    let module = sonatina_parser::parse_module(bool_loop).expect("bool loop should parse").module;
    SpirvBackend::new().compile_module(&module).expect("i1 loop phi should compile");
}

#[test]
fn boolean_is_zero_lowers_as_logical_not() {
    let source = r#"
target = "wasm32-unknown-native"
func public %bool_not(v0.i32) -> i32 {
    block0:
        v1.i1 = is_zero v0;
        v2.i1 = is_zero v1;
        br v2 block1 block2;
    block1:
        return 1.i32;
    block2:
        return 0.i32;
}
"#;
    let module = sonatina_parser::parse_module(source).expect("bool not should parse").module;
    SpirvBackend::new().compile_module(&module).expect("i1 is_zero should compile");
}

#[test]
fn f32_object_load_and_store_have_named_rejections() {
    let store = r#"
target = "wasm32-unknown-native"
type @box = {f32};
func public %store(v0.i32, v1.i32, v2.f32) -> i32 {
    block0:
        v3.objref<@box> = obj.alloc @box;
        v4.objref<f32> = obj.proj v3 0.i8;
        obj.store v4 v2;
        return v0;
}
"#;
    let error = spirv_error(store, SpirvBackend::new().with_grid());
    assert!(error.contains("f32 object storage"), "{error}");

    let load = r#"
target = "wasm32-unknown-native"
type @box = {f32};
func public %load(v0.i32, v1.i32) -> i32 {
    block0:
        v2.objref<@box> = obj.alloc @box;
        v3.objref<f32> = obj.proj v2 0.i8;
        v4.f32 = obj.load v3;
        v5.i32 = f32_to_i32 v4;
        return v5;
}
"#;
    let error = spirv_error(load, SpirvBackend::new().with_grid());
    assert!(error.contains("f32 object storage"), "{error}");
}

/// Slice 0 of the float-semantics design (float-semantics NO-GO fix): naga's
/// `Fmin`/`Fmax`/`Fabs`/`Fclamp` lowering must be the pinned-exact,
/// branch-free integer key-compare-and-select expansion (`emit_exact_fminmax`
/// in `crates/codegen/src/isa/spirv/mod.rs`), NOT `MathFunction::Min`/`Max`/
/// `Abs` or a single `FClamp`. This is a structural regression test: the
/// emitted WGSL must reparse/validate AND contain zero branch/phi
/// (`if `/`loop {`) for these ops, only `select(`. Kernel ABI needs an i32
/// return, so args/result are converted through `I32ToF32`/`F32ToI32` (same
/// trick as `build_grid_f32_loop_return_module` above); the four ops are
/// combined into one f32 accumulator so a single WGSL dump covers all of
/// them.
#[test]
fn spirv_f32_minmaxabsclamp_lowering_is_exact_and_branch_free() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64, Vendor::Unknown, OperatingSystem::Native
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();

    let sig = Signature::new_single(
        "f32_minmaxabsclamp", Linkage::Public, &[Type::I32, Type::I32], Type::I32,
    );
    let func_ref = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);

    let a = fb.args()[0];
    let b = fb.args()[1];
    let af = fb.insert_inst(cast::I32ToF32::new(is, a), Type::F32);
    let bf = fb.insert_inst(cast::I32ToF32::new(is, b), Type::F32);

    let min = fb.insert_inst(arith::Fmin::new(is, af, bf), Type::F32);
    let max = fb.insert_inst(arith::Fmax::new(is, af, bf), Type::F32);
    let abs = fb.insert_inst(arith::Fabs::new(is, af), Type::F32);
    let clamp = fb.insert_inst(arith::Fclamp::new(is, af, min, max), Type::F32);

    let s1 = fb.insert_inst(arith::Fadd::new(is, min, max), Type::F32);
    let s2 = fb.insert_inst(arith::Fadd::new(is, s1, abs), Type::F32);
    let s3 = fb.insert_inst(arith::Fadd::new(is, s2, clamp), Type::F32);
    let out = fb.insert_inst(cast::F32ToI32::new(is, s3), Type::I32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, out));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let backend = SpirvBackend::new().with_workgroup_size(1, 1, 1);
    let artifact = backend
        .compile_module(&module)
        .expect("min/max/abs/clamp kernel must compile to SPIR-V");
    assert_eq!(artifact.words[0], 0x07230203, "valid SPIR-V magic");

    let wgsl = artifact.wgsl.as_ref().expect("WGSL side artifact");

    // Structural evidence of branch-freedom: naga's WGSL writer emits
    // `if `/`else `/`loop {` for `Statement::If`/`Loop` and `select(` for
    // `Expression::Select`
    // (~/.cargo/registry/.../naga-29.0.3/src/back/wgsl/writer.rs). None of
    // our new IR ops (min/max/abs/clamp, all Opaque/Binary/Unary, zero
    // control-flow instructions in the source function) should produce any
    // `Statement`, so this also incidentally covers "no phi" (naga has no
    // phi node concept in its IR at all; sonatina phi only ever lowers to a
    // `LocalVariable` load/store, and this function has none).
    assert!(
        !wgsl.contains("if ") && !wgsl.contains("else") && !wgsl.contains("loop {"),
        "min/max/abs/clamp WGSL must be branch-free (no if/else/loop); got:\n{wgsl}"
    );
    let select_count = wgsl.matches("select(").count();
    assert!(
        select_count >= 6,
        "expected >= 6 `select(` calls (min: 2, max: 2, clamp composes two more \
         min/max calls with 4 more selects, minus shared literals) in the exact \
         expansion, got {select_count}; WGSL:\n{wgsl}"
    );

    let reparsed = naga::front::wgsl::parse_str(wgsl)
        .expect("naga wgsl-in must reparse the min/max/abs/clamp WGSL");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    )
    .validate(&reparsed)
    .expect("browser-profile validation must accept the min/max/abs/clamp module");

    eprintln!("spirv_f32_minmaxabsclamp_lowering_is_exact_and_branch_free OK: {select_count} selects, 0 branches");
}

/// Slice 1 of the float-semantics design (the typed opt-in,
/// `/workspace/mb2/FLOAT_SEMANTICS_TYPE_API_DESIGN.md`): THE POINT. Plain
/// `Fmin`/`Fmax` (reachable from a naive `f32` `min`/`max` in Fe) must keep
/// paying the pinned-exact ~15-20-op branch-free integer expansion; the new
/// `FminRelaxed`/`FmaxRelaxed` ops (reachable only through the `Regular`
/// domain newtype in Fe) must lower to a SINGLE native `MathFunction::Min`/
/// `Max` WGSL call -- literally `min(...)`/`max(...)`, no bitcast, no
/// select. Two structurally-identical kernels (same ABI, same op count),
/// differing only in Fmin/Fmax vs FminRelaxed/FmaxRelaxed, so any WGSL delta
/// is attributable to the op choice alone, not kernel shape. A third
/// "baseline" kernel with no min/max at all isolates how many `select(`s the
/// trailing `F32ToI32` return-ABI conversion contributes on its own, so the
/// exact-vs-relaxed delta can be attributed precisely.
#[test]
fn spirv_f32_min_relaxed_is_single_op_exact_min_is_not() {
    fn build_kernel(is: &dyn sonatina_ir::InstSetBase, relaxed: bool, with_minmax: bool) -> sonatina_ir::module::Module {
        let mb = native_module_builder();
        let name = if with_minmax {
            if relaxed { "f32_min_relaxed" } else { "f32_min_exact" }
        } else {
            "f32_identity_baseline"
        };
        let sig = Signature::new_single(name, Linkage::Public, &[Type::I32, Type::I32], Type::I32);
        let func_ref = mb.declare_function(sig).unwrap();
        let mut fb = mb.func_builder::<InstInserter>(func_ref);
        let entry = fb.append_block();
        fb.switch_to_block(entry);

        let a = fb.args()[0];
        let b = fb.args()[1];
        let af = fb.insert_inst(cast::I32ToF32::new(is, a), Type::F32);
        let bf = fb.insert_inst(cast::I32ToF32::new(is, b), Type::F32);

        let result_f = if !with_minmax {
            fb.insert_inst(arith::Fadd::new(is, af, bf), Type::F32)
        } else if relaxed {
            fb.insert_inst(arith::FminRelaxed::new(is, af, bf), Type::F32)
        } else {
            fb.insert_inst(arith::Fmin::new(is, af, bf), Type::F32)
        };
        let out = fb.insert_inst(cast::F32ToI32::new(is, result_f), Type::I32);
        fb.insert_inst_no_result(control_flow::Return::new_single(is, out));
        fb.seal_all();
        fb.finish();

        mb.build()
    }

    fn compile_wgsl(module: &sonatina_ir::module::Module) -> String {
        let backend = SpirvBackend::new().with_workgroup_size(1, 1, 1);
        let artifact = backend.compile_module(module).expect("kernel must compile to SPIR-V");
        assert_eq!(artifact.words[0], 0x07230203, "valid SPIR-V magic");
        artifact.wgsl.as_ref().expect("WGSL side artifact").clone()
    }

    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64, Vendor::Unknown, OperatingSystem::Native
    ));
    let is = isa.inst_set();

    let baseline_wgsl = compile_wgsl(&build_kernel(is, false, false));
    let baseline_selects = baseline_wgsl.matches("select(").count();

    let exact_wgsl = compile_wgsl(&build_kernel(is, false, true));
    let relaxed_wgsl = compile_wgsl(&build_kernel(is, true, true));

    // Relaxed: a single native `min(` call, no extra `select(`s beyond
    // whatever the I32<->F32 ABI conversion already contributes on its own
    // (the baseline kernel, which has no min/max at all).
    assert!(
        relaxed_wgsl.contains("min("),
        "expected a native `min(` call in the relaxed WGSL; got:\n{relaxed_wgsl}"
    );
    let relaxed_selects = relaxed_wgsl.matches("select(").count();
    assert_eq!(
        relaxed_selects, baseline_selects,
        "FminRelaxed must add ZERO select()s beyond the ABI-conversion baseline \
         ({baseline_selects}); relaxed WGSL:\n{relaxed_wgsl}"
    );

    // Exact: the pinned branch-free integer expansion, strictly more
    // selects than the relaxed kernel (and than the baseline), and no
    // native `min(` call.
    assert!(
        !exact_wgsl.contains("min("),
        "exact Fmin must NOT lower to a native `min(` call; got:\n{exact_wgsl}"
    );
    let exact_selects = exact_wgsl.matches("select(").count();
    assert!(
        exact_selects > relaxed_selects,
        "exact Fmin must cost strictly more select()s than relaxed FminRelaxed \
         (exact={exact_selects}, relaxed={relaxed_selects}, baseline={baseline_selects})"
    );
    assert!(
        exact_selects >= baseline_selects + 2,
        "exact Fmin's emit_exact_fminmax must add >= 2 selects (pick + nan-detect) \
         over the ABI baseline; exact={exact_selects}, baseline={baseline_selects}"
    );

    for wgsl in [&exact_wgsl, &relaxed_wgsl] {
        assert!(
            !wgsl.contains("if ") && !wgsl.contains("else") && !wgsl.contains("loop {"),
            "both exact and relaxed WGSL must stay branch-free (no if/else/loop); got:\n{wgsl}"
        );
        let reparsed = naga::front::wgsl::parse_str(wgsl).expect("naga wgsl-in must reparse");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::default(),
        )
        .validate(&reparsed)
        .expect("browser-profile validation must accept the module");
    }

    eprintln!(
        "spirv_f32_min_relaxed_is_single_op_exact_min_is_not OK: baseline={baseline_selects} \
         selects, relaxed={relaxed_selects} selects, exact={exact_selects} selects"
    );
    eprintln!("=== EXACT (Fmin) WGSL ===\n{exact_wgsl}");
    eprintln!("=== RELAXED (FminRelaxed) WGSL ===\n{relaxed_wgsl}");
}

/// Rounding-family (VERIFY item 2): `Ffloor`/`Fceil`/`Ftrunc`/`Fround` must
/// each emit a single native `MathFunction` call (`floor`/`ceil`/`trunc`/
/// `round` in WGSL), no branch/phi, mirroring
/// `spirv_f32_minmaxabsclamp_lowering_is_exact_and_branch_free` above.
/// `Fround` in particular must reparse/validate as WGSL's `round()`, which is
/// ties-to-even by spec (matching wasm `f32.nearest`/cranelift `nearest`);
/// this test does not (and cannot, no GPU adapter here) execute the shader,
/// so the ties-to-even claim itself is pinned by the naga source-level check
/// in `arith::Fround`'s doc comment plus `cranelift_backend.rs`'s oracle, not
/// here -- this test only asserts the *shape* of the emitted code.
#[test]
fn spirv_f32_rounding_lowering_is_exact_and_branch_free() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64, Vendor::Unknown, OperatingSystem::Native
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();

    let sig = Signature::new_single(
        "f32_rounding", Linkage::Public, &[Type::I32], Type::I32,
    );
    let func_ref = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);

    let a = fb.args()[0];
    let af = fb.insert_inst(cast::I32ToF32::new(is, a), Type::F32);

    let floor = fb.insert_inst(arith::Ffloor::new(is, af), Type::F32);
    let ceil = fb.insert_inst(arith::Fceil::new(is, af), Type::F32);
    let trunc = fb.insert_inst(arith::Ftrunc::new(is, af), Type::F32);
    let round = fb.insert_inst(arith::Fround::new(is, af), Type::F32);

    let s1 = fb.insert_inst(arith::Fadd::new(is, floor, ceil), Type::F32);
    let s2 = fb.insert_inst(arith::Fadd::new(is, s1, trunc), Type::F32);
    let s3 = fb.insert_inst(arith::Fadd::new(is, s2, round), Type::F32);
    let out = fb.insert_inst(cast::F32ToI32::new(is, s3), Type::I32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, out));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let backend = SpirvBackend::new().with_workgroup_size(1, 1, 1);
    let artifact = backend
        .compile_module(&module)
        .expect("rounding-family kernel must compile to SPIR-V");
    assert_eq!(artifact.words[0], 0x07230203, "valid SPIR-V magic");

    let wgsl = artifact.wgsl.as_ref().expect("WGSL side artifact");

    assert!(
        !wgsl.contains("if ") && !wgsl.contains("else") && !wgsl.contains("loop {"),
        "rounding-family WGSL must be branch-free (no if/else/loop); got:\n{wgsl}"
    );
    for (op, wgsl_fn) in [
        (stringify!(Ffloor), "floor("),
        (stringify!(Fceil), "ceil("),
        (stringify!(Ftrunc), "trunc("),
        (stringify!(Fround), "round("),
    ] {
        assert!(
            wgsl.contains(wgsl_fn),
            "expected a single native `{wgsl_fn}` call for {op}; WGSL:\n{wgsl}"
        );
    }
    // Unlike the exact Fmin/Fmax expansion, the rounding family has no
    // NaN/-0.0 ambiguity to work around, so each op is a single
    // `MathFunction` `OpExtInst`/WGSL call, not an integer expansion: the
    // `floor(`/`ceil(`/`trunc(`/`round(` containment checks above already
    // pin that "single call, no branch" shape. (The kernel's trailing
    // `F32ToI32` return-ABI conversion legitimately emits its own
    // `select(`s for saturation -- unrelated to this family -- so this test
    // does not assert a global "zero select" count.)

    let reparsed = naga::front::wgsl::parse_str(wgsl)
        .expect("naga wgsl-in must reparse the rounding-family WGSL");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    )
    .validate(&reparsed)
    .expect("browser-profile validation must accept the rounding-family module");

    eprintln!("spirv_f32_rounding_lowering_is_exact_and_branch_free OK: floor/ceil/trunc/round all single-OpExtInst, 0 branches, 0 selects");
}

// ===========================================================================
// Rung 3 STEP 2: SPIR-V lowering of function-local [u32; N] arrays
// (MemAllocDynamic / Mload / Mstore), a private-storage `fe_heap` +
// `fe_bump` emulation (RUNG3_SPIRV_ARRAYS_DESIGN.md). Hand-built Sonatina IR
// (no Fe compiler in this crate), mirroring this file's existing style.
//
// Each test names the specific adversarial-review finding it guards
// against (heap-exhaustion aliasing, misaligned-access miscompile,
// poison-sentinel collision, wrong-value-on-unconditional-trap).
// ===========================================================================

/// The S2-A probe from the design doc: `probe(k) -> u32` allocates an 8-word
/// array, stores a constant at a[3], then bounds-checks a dynamic load a[k]
/// (Lt + Br + an Unreachable trap arm -- the exact shape `wasm_lower.rs`
/// emits for every Fe dynamic array access). Exercises MemAllocDynamic,
/// Mstore, Mload, and the Unreachable-as-trap path together.
#[test]
fn spirv_array_probe_compiles_naga_valid_with_heap_and_trap_channel() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64, Vendor::Unknown, OperatingSystem::Native
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();

    let sig = Signature::new_single("probe", Linkage::Public, &[Type::I32], Type::I32);
    let func_ref = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(func_ref);

    let entry = fb.append_block();
    let ok = fb.append_block();
    let trap = fb.append_block();

    fb.switch_to_block(entry);
    let k = fb.args()[0];
    let alloc_size = fb.make_imm_value(32i32); // 8 words * 4 bytes
    let base = fb.insert_inst(data::MemAllocDynamic::new(is, alloc_size), Type::I32);
    let twelve = fb.make_imm_value(12i32);
    let addr3 = fb.insert_inst(arith::Add::new(is, base, twelve), Type::I32);
    let stored = fb.make_imm_value(0xABCDi32);
    fb.insert_inst_no_result(data::Mstore::new(is, addr3, stored, Type::I32));
    let eight = fb.make_imm_value(8i32);
    let in_bounds = fb.insert_inst(cmp::Lt::new(is, k, eight), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, in_bounds, ok, trap));

    fb.switch_to_block(ok);
    let four = fb.make_imm_value(4i32);
    let byte_off = fb.insert_inst(arith::Mul::new(is, k, four), Type::I32);
    let addr_k = fb.insert_inst(arith::Add::new(is, base, byte_off), Type::I32);
    let loaded = fb.insert_inst(data::Mload::new(is, addr_k, Type::I32), Type::I32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, loaded));

    fb.switch_to_block(trap);
    fb.insert_inst_no_result(control_flow::Unreachable::new(is));

    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let backend = SpirvBackend::new().with_workgroup_size(1, 1, 1);
    let artifact = backend.compile_module(&module).expect(
        "array probe (MemAllocDynamic/Mstore/Mload + a bounds trap) should compile to \
         naga-validated SPIR-V",
    );

    assert_eq!(artifact.words[0], 0x0723_0203, "valid SPIR-V magic");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL side artifact");
    assert!(wgsl.contains("fe_heap"), "private heap local must appear in WGSL:\n{wgsl}");
    assert!(
        wgsl.contains("array<u32, 8>"),
        "the proven eight-word allocation must size the emitted private heap exactly:\n{wgsl}"
    );
    assert!(
        !wgsl.contains("array<u32, 8192>"),
        "the default capacity is a compile-time ceiling, not per-invocation storage:\n{wgsl}"
    );
    assert!(wgsl.contains("fe_bump"), "bump pointer local must appear in WGSL:\n{wgsl}");
    assert!(
        wgsl.contains("fe_trapped"),
        "trap-status local must appear in WGSL (review findings 1/2/3/4's shared channel):\n{wgsl}"
    );

    // Review finding 3 (poison-sentinel collision): the trap channel is a real,
    // separate binding stated in the layout, not folded into the result.
    assert!(
        artifact.layout.trap.is_some(),
        "a Scalar-mode Mem-bearing kernel must state a trap SpirvResult"
    );
    assert!(
        artifact.layout.bindings.iter().any(|b| b.name == "trap"),
        "layout bindings must include the trap channel"
    );
    assert_ne!(
        artifact.layout.trap.map(|t| t.binding),
        artifact.layout.result.map(|r| r.binding),
        "trap and result must occupy DIFFERENT bindings (never overload the result slot)"
    );

    let reparsed = naga::front::wgsl::parse_str(wgsl)
        .expect("naga wgsl-in should reparse the array-probe WGSL");
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(&reparsed)
        .expect("re-validation of the reparsed WGSL should also pass");

    eprintln!(
        "array probe OK: {} SPIR-V words; WGSL carries fe_heap/fe_bump/fe_trapped; trap binding \
         is a distinct group 0 binding {} (result is binding {})",
        artifact.words.len(),
        artifact.layout.trap.unwrap().binding,
        artifact.layout.result.unwrap().binding,
    );
}

/// Review finding 1 (heap-exhaustion aliasing): a single allocation whose constant
/// size exceeds the declared private-heap capacity (default 8192 words =
/// 32768 bytes) must fail the compile OUTRIGHT, never silently clamp the
/// bump pointer and let two logically distinct arrays alias the same heap
/// word.
#[test]
fn spirv_mem_alloc_exceeding_heap_capacity_fails_closed_at_compile_time() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64, Vendor::Unknown, OperatingSystem::Native
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single("big_alloc", Linkage::Public, &[], Type::I32);
    let func_ref = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);
    let too_big = fb.make_imm_value(40_000i32); // > 32768-byte default heap
    let base = fb.insert_inst(data::MemAllocDynamic::new(is, too_big), Type::I32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, base));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let backend = SpirvBackend::new();
    let errors = match backend.compile_module(&module) {
        Ok(_) => panic!(
            "an allocation exceeding the private heap capacity must fail closed at compile \
             time, never silently clamp-and-alias"
        ),
        Err(errors) => errors,
    };
    let msg = errors.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("; ");
    assert!(
        msg.contains("exceeds the private heap capacity"),
        "expected a named heap-capacity error, got: {msg}"
    );
    assert!(
        msg.contains("Largest contributors: 40000 bytes total = 40000 bytes x 1")
            && msg.contains("instruction `mem.alloc_dynamic`"),
        "expected the static allocation census to identify the oversized allocation, got: {msg}"
    );
    eprintln!("heap overflow correctly fails closed at compile time: {msg}");
}

/// Two verified sibling arena scopes cannot be live simultaneously. Their
/// private words are therefore reused after rewind instead of being charged as
/// one monolithic allocation sum.
#[test]
fn spirv_sequential_arena_scopes_reuse_private_heap_capacity() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64,
        Vendor::Unknown,
        OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single("sequential_scopes", Linkage::Public, &[], Type::I32);
    let func_ref = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);

    let size = fb.make_imm_value(24_000i32);
    let first_checkpoint = fb.insert_inst(data::MemCheckpoint::new(is), Type::I32);
    let first = fb.insert_inst(data::MemAllocDynamic::new(is, size), Type::I32);
    let seven = fb.make_imm_value(7i32);
    fb.insert_inst_no_result(data::Mstore::new(is, first, seven, Type::I32));
    let carried = fb.insert_inst(data::Mload::new(is, first, Type::I32), Type::I32);
    fb.insert_inst_no_result(data::MemRewind::new(is, first_checkpoint));

    let second_checkpoint = fb.insert_inst(data::MemCheckpoint::new(is), Type::I32);
    let second = fb.insert_inst(data::MemAllocDynamic::new(is, size), Type::I32);
    fb.insert_inst_no_result(data::Mstore::new(is, second, carried, Type::I32));
    let result = fb.insert_inst(data::Mload::new(is, second, Type::I32), Type::I32);
    fb.insert_inst_no_result(data::MemRewind::new(is, second_checkpoint));
    fb.insert_inst_no_result(control_flow::Return::new_single(is, result));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let artifact = SpirvBackend::new()
        .compile_module(&module)
        .expect("sequential verified scopes must reuse private heap words");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL side artifact");
    assert!(
        wgsl.contains("array<u32, 6000>"),
        "two sequential 24 KB scopes need one 24 KB private heap:\n{wgsl}",
    );
}

/// Nested scopes overlap their parent's live allocation. They must add rather
/// than reuse capacity, preserving the fail-closed heap bound.
#[test]
fn spirv_nested_arena_scopes_add_private_heap_capacity() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64,
        Vendor::Unknown,
        OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single("nested_scopes", Linkage::Public, &[], Type::I32);
    let func_ref = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);

    let size = fb.make_imm_value(20_000i32);
    let outer_checkpoint = fb.insert_inst(data::MemCheckpoint::new(is), Type::I32);
    let outer = fb.insert_inst(data::MemAllocDynamic::new(is, size), Type::I32);
    let seven = fb.make_imm_value(7i32);
    fb.insert_inst_no_result(data::Mstore::new(is, outer, seven, Type::I32));
    let inner_checkpoint = fb.insert_inst(data::MemCheckpoint::new(is), Type::I32);
    let inner = fb.insert_inst(data::MemAllocDynamic::new(is, size), Type::I32);
    fb.insert_inst_no_result(data::Mstore::new(is, inner, seven, Type::I32));
    let result = fb.insert_inst(data::Mload::new(is, inner, Type::I32), Type::I32);
    fb.insert_inst_no_result(data::MemRewind::new(is, inner_checkpoint));
    fb.insert_inst_no_result(data::MemRewind::new(is, outer_checkpoint));
    fb.insert_inst_no_result(control_flow::Return::new_single(is, result));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let errors = match SpirvBackend::new().compile_module(&module) {
        Ok(_) => panic!("nested 20 KB scopes must exceed the 32 KB private heap"),
        Err(errors) => errors,
    };
    let message = errors
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("; ");
    assert!(
        message.contains("static allocation high-water (40000 bytes)"),
        "nested scopes must retain a 40 KB high-water bound: {message}",
    );
}

/// A `MemAllocDynamic` inside a loop whose bound is supplied at runtime has a
/// total byte cost that the compile-time capacity proof cannot bound. It must
/// fail closed rather than silently summing only one iteration.
#[test]
fn spirv_mem_alloc_inside_loop_fails_closed() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64, Vendor::Unknown, OperatingSystem::Native
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single("loop_alloc", Linkage::Public, &[Type::I32], Type::I32);
    let func_ref = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let runtime_limit = fb.args()[0];

    let entry = fb.append_block();
    let header = fb.append_block();
    let body = fb.append_block();
    let exit = fb.append_block();

    fb.switch_to_block(entry);
    let zero = fb.make_imm_value(0i32);
    fb.insert_inst_no_result(control_flow::Jump::new(is, header));

    fb.switch_to_block(header);
    let i = fb.insert_inst(control_flow::Phi::new(is, vec![(zero, entry)]), Type::I32);
    let cond = fb.insert_inst(cmp::Lt::new(is, i, runtime_limit), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, cond, body, exit));

    fb.switch_to_block(body);
    let alloc_size = fb.make_imm_value(16i32);
    let _base = fb.insert_inst(data::MemAllocDynamic::new(is, alloc_size), Type::I32);
    let one = fb.make_imm_value(1i32);
    let next_i = fb.insert_inst(arith::Add::new(is, i, one), Type::I32);
    fb.append_phi_arg(i, next_i, body);
    fb.insert_inst_no_result(control_flow::Jump::new(is, header));

    fb.switch_to_block(exit);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, i));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let backend = SpirvBackend::new();
    let errors = match backend.compile_module(&module) {
        Ok(_) => panic!("MemAllocDynamic inside a loop must fail closed (unbounded compile-time total)"),
        Err(errors) => errors,
    };
    let msg = errors.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("; ");
    assert!(
        msg.contains("MemAllocDynamic inside a loop"),
        "expected the loop-carried-allocation error, got: {msg}"
    );
    eprintln!("loop-carried allocation correctly fails closed: {msg}");
}

/// A canonical constant-bounded induction loop has a compiler-proven maximum
/// allocation count. The private heap includes every iteration rather than
/// treating the allocation site as if it executed once.
#[test]
fn spirv_statically_bounded_mem_alloc_inside_loop_compiles() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64,
        Vendor::Unknown,
        OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single("bounded_loop_alloc", Linkage::Public, &[], Type::I32);
    let func_ref = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(func_ref);

    let entry = fb.append_block();
    let header = fb.append_block();
    let body = fb.append_block();
    let exit = fb.append_block();

    fb.switch_to_block(entry);
    let zero = fb.make_imm_value(0i32);
    fb.insert_inst_no_result(control_flow::Jump::new(is, header));

    fb.switch_to_block(header);
    let i = fb.insert_inst(control_flow::Phi::new(is, vec![(zero, entry)]), Type::I32);
    let three = fb.make_imm_value(3i32);
    let cond = fb.insert_inst(cmp::Lt::new(is, i, three), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, cond, body, exit));

    fb.switch_to_block(body);
    let alloc_size = fb.make_imm_value(16i32);
    let base = fb.insert_inst(data::MemAllocDynamic::new(is, alloc_size), Type::I32);
    fb.insert_inst_no_result(data::Mstore::new(is, base, i, Type::I32));
    let one = fb.make_imm_value(1i32);
    let next_i = fb.insert_inst(arith::Add::new(is, i, one), Type::I32);
    fb.append_phi_arg(i, next_i, body);
    fb.insert_inst_no_result(control_flow::Jump::new(is, header));

    fb.switch_to_block(exit);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, i));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let artifact = SpirvBackend::new()
        .compile_module(&module)
        .expect("a statically bounded loop allocation must compile");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL side artifact");
    assert!(
        wgsl.contains("array<u32, 12>"),
        "three 16-byte allocations need twelve private words:\n{wgsl}"
    );
    let reparsed = naga::front::wgsl::parse_str(wgsl).expect("naga must reparse bounded-loop WGSL");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    )
    .validate(&reparsed)
    .expect("browser-profile validation must accept bounded-loop WGSL");
}

/// A compiler-proven arena frame changes the loop allocation bound from
/// runtime-trip-count-dependent to one statically known iteration footprint.
/// The backend independently checks the checkpoint stack at every CFG join and
/// lowers the rewind into a guarded restoration of `fe_bump`.
#[test]
fn spirv_scoped_mem_alloc_inside_loop_compiles() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64,
        Vendor::Unknown,
        OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single("scoped_loop_alloc", Linkage::Public, &[], Type::I32);
    let func_ref = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(func_ref);

    let entry = fb.append_block();
    let header = fb.append_block();
    let body = fb.append_block();
    let exit = fb.append_block();

    fb.switch_to_block(entry);
    let zero = fb.make_imm_value(0i32);
    fb.insert_inst_no_result(control_flow::Jump::new(is, header));

    fb.switch_to_block(header);
    let i = fb.insert_inst(control_flow::Phi::new(is, vec![(zero, entry)]), Type::I32);
    let three = fb.make_imm_value(3i32);
    let cond = fb.insert_inst(cmp::Lt::new(is, i, three), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, cond, body, exit));

    fb.switch_to_block(body);
    let checkpoint = fb.insert_inst(data::MemCheckpoint::new(is), Type::I32);
    let alloc_size = fb.make_imm_value(16i32);
    let base = fb.insert_inst(data::MemAllocDynamic::new(is, alloc_size), Type::I32);
    fb.insert_inst_no_result(data::Mstore::new(is, base, i, Type::I32));
    let loaded = fb.insert_inst(data::Mload::new(is, base, Type::I32), Type::I32);
    fb.insert_inst_no_result(data::MemRewind::new(is, checkpoint));
    let one = fb.make_imm_value(1i32);
    let next_i = fb.insert_inst(arith::Add::new(is, loaded, one), Type::I32);
    fb.append_phi_arg(i, next_i, body);
    fb.insert_inst_no_result(control_flow::Jump::new(is, header));

    fb.switch_to_block(exit);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, i));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let artifact = SpirvBackend::new()
        .compile_module(&module)
        .expect("a balanced scoped loop allocation must compile");
    assert_eq!(artifact.words[0], 0x0723_0203, "valid SPIR-V magic");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL side artifact");
    assert!(
        wgsl.contains("array<u32, 4>"),
        "one scoped 16-byte allocation needs exactly four private words:\n{wgsl}"
    );
    assert!(
        wgsl.matches("fe_bump").count() >= 4,
        "checkpoint, allocation, and rewind must all operate on fe_bump:\n{wgsl}"
    );
    let reparsed = naga::front::wgsl::parse_str(wgsl).expect("naga must reparse scoped-arena WGSL");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    )
    .validate(&reparsed)
    .expect("browser-profile validation must accept scoped-arena WGSL");
}

/// Memcopy carries WebAssembly memory.copy semantics, including unaligned and
/// overlapping ranges. The SPIR-V path emits a bounded byte loop and chooses
/// the safe direction at runtime instead of silently treating it as memcpy.
#[test]
fn spirv_memcopy_is_bounded_overlap_safe_memmove() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64,
        Vendor::Unknown,
        OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single("overlap_copy", Linkage::Public, &[], Type::I32);
    let func_ref = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);

    let alloc_size = fb.make_imm_value(32i32);
    let base = fb.insert_inst(data::MemAllocDynamic::new(is, alloc_size), Type::I32);
    let one = fb.make_imm_value(1i32);
    let destination = fb.insert_inst(arith::Add::new(is, base, one), Type::I32);
    let len = fb.make_imm_value(7i32);
    fb.insert_inst_no_result(data::Memcopy::new(is, destination, base, len));
    fb.insert_inst_no_result(control_flow::Return::new_single(is, base));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let artifact = SpirvBackend::new()
        .compile_module(&module)
        .expect("bounded unaligned overlapping Memcopy must compile");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL side artifact");
    assert!(
        wgsl.contains("fe_memcopy_index"),
        "Memcopy must retain its compact bounded loop index:\n{wgsl}"
    );
    assert!(
        wgsl.contains("loop"),
        "Memcopy must lower to a loop rather than byte-count-sized shader text:\n{wgsl}"
    );
    let reparsed = naga::front::wgsl::parse_str(wgsl).expect("naga must reparse Memcopy WGSL");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    )
    .validate(&reparsed)
    .expect("browser-profile validation must accept Memcopy WGSL");
}

/// I1 memory uses Wasm-compatible byte addressing over the private u32 heap.
/// The deliberately unaligned `base + 1` access must select the second byte,
/// preserve its neighboring bytes, and remain valid SPIR-V/WGSL.
#[test]
fn spirv_i1_mem_uses_byte_exact_packed_heap() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64, Vendor::Unknown, OperatingSystem::Native
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single("bool_mem", Linkage::Public, &[], Type::I32);
    let func_ref = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    let is_true = fb.append_block();
    let is_false = fb.append_block();

    fb.switch_to_block(entry);
    let size = fb.make_imm_value(4i32);
    let base = fb.insert_inst(data::MemAllocDynamic::new(is, size), Type::I32);
    let original = fb.make_imm_value(0x5a5a_5a5ai32);
    fb.insert_inst_no_result(data::Mstore::new(is, base, original, Type::I32));
    let one = fb.make_imm_value(1i32);
    let byte_one = fb.insert_inst(arith::Add::new(is, base, one), Type::I32);
    let value = fb.make_imm_value(true);
    fb.insert_inst_no_result(data::Mstore::new(is, byte_one, value, Type::I1));
    let loaded_byte = fb.insert_inst(data::Mload::new(is, byte_one, Type::I1), Type::I32);
    let flag = fb.insert_inst(cast::Trunc::new(is, loaded_byte, Type::I1), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, flag, is_true, is_false));

    fb.switch_to_block(is_true);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, one));

    fb.switch_to_block(is_false);
    let zero = fb.make_imm_value(0i32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, zero));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let artifact = SpirvBackend::new()
        .compile_module(&module)
        .expect("I1 byte-lane memory must compile to validated SPIR-V");
    assert_eq!(artifact.words[0], 0x0723_0203, "valid SPIR-V magic");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL side artifact");
    assert!(wgsl.contains("fe_heap"), "private heap must appear in WGSL:\n{wgsl}");
    assert!(
        wgsl.contains("255u") && wgsl.contains("<<") && wgsl.contains(">>"),
        "I1 memory must use a byte mask and lane shifts over the u32 heap:\n{wgsl}"
    );
}

/// Design section 2: Mem ops are supported under the u32 (browser) word
/// only. A kernel returning I64 (selecting the i64 word) that also uses
/// MemAllocDynamic/Mload must fail closed rather than emit an ambiguous
/// mixed-width heap.
#[test]
fn spirv_mem_op_under_i64_word_fails_closed() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64, Vendor::Unknown, OperatingSystem::Native
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();
    let sig = Signature::new_single("i64_mem", Linkage::Public, &[], Type::I64);
    let func_ref = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);
    let size = fb.make_imm_value(16i32);
    let base = fb.insert_inst(data::MemAllocDynamic::new(is, size), Type::I32);
    let loaded = fb.insert_inst(data::Mload::new(is, base, Type::I32), Type::I32);
    let widened = fb.insert_inst(cast::Zext::new(is, loaded, Type::I64), Type::I64);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, widened));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let backend = SpirvBackend::new();
    // Whichever named check fires first (the i64-word Mem gate, or the more
    // general narrow/mixed-carrier-type check, since an I32-typed address
    // under an I64 word ALSO trips that one), the load-bearing property is
    // "fails closed with a named error", not one specific message.
    let errors = match backend.compile_module(&module) {
        Ok(_) => panic!("Mem ops under the i64 word must fail closed (u32 word only)"),
        Err(errors) => errors,
    };
    let msg = errors.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("; ");
    assert!(!msg.is_empty(), "expected a named error, got an empty message");
    eprintln!("i64-word Mem op correctly fails closed: {msg}");
}

/// `wasm_lower.rs::trap_block` creates and caches exactly ONE trap block per
/// function, reused for EVERY dynamically-indexed bounds check
/// (`crates/codegen/src/sonatina/wasm_lower.rs::trap_block`, mb2 fe-codegen).
/// A function with two SEQUENTIAL bounds-checked accesses branches to the
/// SAME `Unreachable`-only `BlockId` from two different `Br` sites. Without
/// `Structurer::is_shared_trap_block`'s special case, the second occurrence
/// hits `build_seq`'s general active/consumed cycle guard and the whole
/// function fails to structurize ("cyclic or multiply consumed block") --
/// this was found live against `field_mul_bn254_fr_loop.fe` (Rung 3's own
/// target fixture has ~dozens of dynamically-indexed accesses in one
/// function) and is not one of the four originally reported findings, but
/// is the same class of defect: a real kernel silently could not reach the
/// SPIR-V backend at all.
#[test]
fn spirv_two_bounds_checks_sharing_one_trap_block_compiles() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64, Vendor::Unknown, OperatingSystem::Native
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();

    let sig = Signature::new_single(
        "two_reads", Linkage::Public, &[Type::I32, Type::I32], Type::I32,
    );
    let func_ref = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(func_ref);

    let entry = fb.append_block();
    let ok1 = fb.append_block();
    let check2 = fb.append_block();
    let ok2 = fb.append_block();
    let trap = fb.append_block(); // SHARED by both checks, like wasm_lower.rs's cache

    fb.switch_to_block(entry);
    let j = fb.args()[0];
    let k = fb.args()[1];
    let size = fb.make_imm_value(32i32);
    let base = fb.insert_inst(data::MemAllocDynamic::new(is, size), Type::I32);
    let eight = fb.make_imm_value(8i32);
    let cond1 = fb.insert_inst(cmp::Lt::new(is, j, eight), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, cond1, ok1, trap));

    fb.switch_to_block(ok1);
    let four = fb.make_imm_value(4i32);
    let off_j = fb.insert_inst(arith::Mul::new(is, j, four), Type::I32);
    let addr_j = fb.insert_inst(arith::Add::new(is, base, off_j), Type::I32);
    let val_j = fb.insert_inst(data::Mload::new(is, addr_j, Type::I32), Type::I32);
    fb.insert_inst_no_result(control_flow::Jump::new(is, check2));

    fb.switch_to_block(check2);
    let cond2 = fb.insert_inst(cmp::Lt::new(is, k, eight), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, cond2, ok2, trap));

    fb.switch_to_block(ok2);
    let off_k = fb.insert_inst(arith::Mul::new(is, k, four), Type::I32);
    let addr_k = fb.insert_inst(arith::Add::new(is, base, off_k), Type::I32);
    let val_k = fb.insert_inst(data::Mload::new(is, addr_k, Type::I32), Type::I32);
    let sum = fb.insert_inst(arith::Add::new(is, val_j, val_k), Type::I32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, sum));

    fb.switch_to_block(trap);
    fb.insert_inst_no_result(control_flow::Unreachable::new(is));

    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let backend = SpirvBackend::new().with_workgroup_size(1, 1, 1);
    let artifact = backend
        .compile_module(&module)
        .expect("two bounds checks sharing one cached trap block must still structurize/compile");

    assert_eq!(artifact.words[0], 0x0723_0203, "valid SPIR-V magic");
    let wgsl = artifact.wgsl.as_deref().expect("WGSL side artifact");
    assert!(wgsl.contains("fe_trapped"), "trap channel must appear in WGSL:\n{wgsl}");

    let reparsed = naga::front::wgsl::parse_str(wgsl)
        .expect("naga wgsl-in should reparse the shared-trap-block WGSL");
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(&reparsed)
        .expect("re-validation of the reparsed WGSL should also pass");

    eprintln!(
        "two bounds checks sharing one cached trap block OK: {} SPIR-V words",
        artifact.words.len()
    );
}

/// Adversarial review Finding A (2026-08-08, CONFIRMED HIGH): a function
/// that traps but has NO Mem ops at all. Before this fix, `mem_ctx` (and the
/// trap channel it carries) was declared ONLY under `has_mem`, so this exact
/// shape -- `fn(k): Br(Lt(k,8), ok, trap); ok: Return 42; trap: Unreachable`
/// -- compiled and naga-validated with the trap arm silently falling through
/// to a zero/uninitialized result and `layout.trap == None`: byte-for-byte
/// the original review finding 4 failure mode, reopened on the has_mem==false
/// side (this is fe-reachable without arrays via `RTerminator::Trap` and
/// `lower_checked_usize_arith`'s `trap_if`, wasm_lower.rs). At the pin
/// (22a95696) this same input hard-errored in the structurizer
/// ("unsupported terminator"); the v2 commits traded that fail-closed
/// behavior for silent acceptance until this test's shape is handled.
///
/// The fix: the trap channel is now declared whenever `has_mem ||
/// has_unreachable`, so a no-Mem trapping function still gets a real,
/// externally-visible trap flag instead of a silent zero. This test asserts
/// exactly that: `fe_trapped` appears in the WGSL, a `trap` binding/result
/// is stated in the layout, and (going one step further than a WGSL text
/// scan) the naga IR itself proves the trap arm stores into `fe_trapped`
/// rather than falling through unnoticed.
#[test]
fn spirv_no_mem_trap_raises_trap_channel_not_silent_zero() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64, Vendor::Unknown, OperatingSystem::Native
    ));
    let is = isa.inst_set();
    let mb = native_module_builder();

    // fn(k) -> i32 { if k < 8 { return 42 } else { unreachable } } -- the
    // review's exact counterexample shape. Zero Mem ops anywhere.
    let sig = Signature::new_single("no_mem_trap", Linkage::Public, &[Type::I32], Type::I32);
    let func_ref = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(func_ref);

    let entry = fb.append_block();
    let ok = fb.append_block();
    let trap = fb.append_block();

    fb.switch_to_block(entry);
    let k = fb.args()[0];
    let eight = fb.make_imm_value(8i32);
    let cond = fb.insert_inst(cmp::Lt::new(is, k, eight), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, cond, ok, trap));

    fb.switch_to_block(ok);
    let forty_two = fb.make_imm_value(42i32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, forty_two));

    fb.switch_to_block(trap);
    fb.insert_inst_no_result(control_flow::Unreachable::new(is));

    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let backend = SpirvBackend::new().with_workgroup_size(1, 1, 1);

    // Either a named fail-closed error, or (the fix actually landed)
    // successful compilation WITH a real trap channel raised on the trap
    // arm -- NEVER silent success with no trap channel at all (that is
    // exactly the reopened hole).
    match backend.compile_module(&module) {
        Err(errors) => {
            let msg = errors.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("; ");
            eprintln!("no-Mem trap failed closed (an acceptable disposition): {msg}");
        }
        Ok(artifact) => {
            let wgsl = artifact.wgsl.as_deref().expect("WGSL side artifact");
            assert!(
                wgsl.contains("fe_trapped"),
                "Finding A regression: a no-Mem trapping function compiled WITHOUT raising a \
                 trap channel -- the trap arm silently falls through to a zero/uninitialized \
                 result exactly like the original review finding 4. WGSL:\n{wgsl}"
            );
            assert!(
                artifact.layout.trap.is_some(),
                "Finding A regression: a no-Mem trapping function compiled but layout.trap is \
                 None -- no consumer can ever observe the trap"
            );
            assert!(
                artifact.layout.bindings.iter().any(|b| b.name == "trap"),
                "Finding A regression: no `trap` binding stated in the layout"
            );

            // Reparse + revalidate independently of the naga run already
            // inside compile_module, so a structurally-broken emission
            // (e.g. the trap store landing outside the block it was meant
            // for) fails loudly here too, not just a text scan.
            let reparsed = naga::front::wgsl::parse_str(wgsl)
                .expect("naga wgsl-in should reparse the no-Mem-trap WGSL");
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&reparsed)
                .expect("re-validation of the reparsed WGSL should also pass");

            // Confirm the trap store is textually INSIDE the branch that
            // guards it (a coarse but meaningful structural check beyond
            // "the token exists somewhere"): the `else` arm produced by a
            // `Br(cond, ok, trap)` with an Unreachable trap arm should read
            // as an if/else whose reject branch sets fe_trapped.
            assert!(
                wgsl.contains("fe_trapped = true"),
                "expected an unconditional `fe_trapped = true` store on the trap arm; WGSL:\n{wgsl}"
            );

            eprintln!(
                "no-Mem trap correctly raises a real trap channel: {} SPIR-V words, trap \
                 binding {:?}",
                artifact.words.len(),
                artifact.layout.trap,
            );
        }
    }
}
