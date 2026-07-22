use sonatina_codegen::Backend;
use sonatina_codegen::isa::spirv::{
    LayoutMode, Role, SpirvBackend, SpirvBuiltinInput, SpirvBuiltinSource, SpirvLayout,
    SpirvBindingMember, SpirvScalarKind, WordKind,
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

fn assert_layout_metadata_invariants(layout: &SpirvLayout, arg_count: usize) {
    let mut seen = vec![false; arg_count];
    for builtin in &layout.builtin_inputs {
        let slot = seen.get_mut(builtin.arg_index as usize).expect("builtin arg index in range");
        assert!(!*slot, "argument {} described twice", builtin.arg_index);
        *slot = true;
    }
    for binding in &layout.bindings {
        assert!(binding.stride >= binding.span, "stride must cover span");
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
            eprintln!("spirv-val not found — skipping validation (structural check passed)");
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

    let output = run_grid_u32(wgsl, 8, 8, 8, 8, &[]);
    assert_eq!(output, vec![2211; 64], "three parallel swaps must end at (22, 11)");
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
/// runtime value (not an immediate); this must fail closed under the u32 word.
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

/// Test 3.3.2: the u32 `Sar` arm emits bitcast-i32 / shift / bitcast-u32 and
/// validates; a non-immediate shift amount fails closed with the named error.
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

    // A non-immediate shift amount fails closed with the named error.
    let bad = build_u32_sar_nonimm();
    let err = match SpirvBackend::new().with_workgroup_size(1, 1, 1).compile_module(&bad) {
        Ok(_) => panic!("non-immediate-bits Sar under u32 must fail closed"),
        Err(errs) => errs.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("; "),
    };
    assert!(
        err.contains("non-immediate shift amount"),
        "non-imm Sar must name the fail-closed reason: {err}"
    );
    eprintln!("spirv_u32_sar_shape OK: bitcast-shift-bitcast + non-imm fails closed");
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

/// The new op arm: a u32 kernel `shr_probe(a) = a >> 4` emits a DIRECT logical
/// `>>` on the u32 (no bitcast dance, unlike Sar) and validates browser-profile.
/// A non-immediate u32 shift amount and an i64 `Shr` both fail closed.
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

    // A non-immediate u32 shift amount fails closed with the named reason.
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
        let bad = mb2.build();
        let err = match SpirvBackend::new().with_workgroup_size(1, 1, 1).compile_module(&bad) {
            Ok(_) => panic!("non-immediate-bits Shr under u32 must fail closed"),
            Err(errs) => errs.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("; "),
        };
        assert!(
            err.contains("non-immediate shift amount"),
            "non-imm Shr must name the fail-closed reason: {err}"
        );
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
    eprintln!("spirv_u32_shr_shape OK: direct logical `>>`, non-imm + i64 fail closed");
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
    assert!(wgsl.contains("p0_: f32") && wgsl.contains("p1_: f32"));
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
