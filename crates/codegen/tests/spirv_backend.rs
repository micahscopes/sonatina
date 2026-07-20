use sonatina_codegen::Backend;
use sonatina_codegen::isa::spirv::{LayoutMode, Role, SpirvBackend, WordKind};
use sonatina_ir::{
    Linkage, Signature, Type,
    builder::ModuleBuilder,
    func_cursor::InstInserter,
    inst::{arith, cmp, control_flow, data},
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
    assert_eq!(layout.mode, LayoutMode::Grid, "grid mode");
    assert_eq!(layout.word, WordKind::U32, "u32 word");
    assert!(layout.result.is_none(), "grid states no single-slot result");
    assert_eq!(layout.workgroup_size, [8, 8, 1], "workgroup size");
    let out_b = layout.bindings.iter().find(|b| b.role == Role::Output).expect("output binding");
    let in_b = layout.bindings.iter().find(|b| b.role == Role::Input).expect("input binding");
    assert_eq!(out_b.stride, 4, "output stride = per-element word width");
    assert_eq!(in_b.stride, 4, "input stride = broadcast span (one dummy member)");

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
    assert!(e.contains("u32 word"), "i64 word must name the u32 requirement: {e}");

    let m = build_grid_1arg_module();
    let e = expect_grid_err(&m, SpirvBackend::new().with_grid().with_workgroup_size(8, 8, 1));
    assert!(e.contains("(px, py)"), "1-arg must name the px, py minimum: {e}");

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

/// Test 3.3.4 (the keystone): the FIRST u32 loop EXECUTED on lavapipe. A grid
/// escape-time kernel (signed `Slt` escape + `Sar`) runs on the GPU and every
/// pixel equals the in-test integer reference.
#[test]
fn u32_escape_grid_executes_on_lavapipe() {
    let module = build_u32_escape_grid();
    let art = SpirvBackend::new()
        .with_grid()
        .with_workgroup_size(8, 8, 1)
        .compile_module(&module)
        .expect("u32 escape grid must compile");
    assert_eq!(art.layout.word, WordKind::U32, "u32 word");
    assert_eq!(art.layout.mode, LayoutMode::Grid, "grid mode");
    let wgsl = art.wgsl.as_ref().expect("WGSL");
    // Honesty checks: the loop really emitted, and the signed ops really went
    // through the i32 bitcast sign mapping.
    assert!(wgsl.contains("loop"), "the escape loop must emit a `loop`:\n{wgsl}");
    assert!(wgsl.contains("bitcast<i32>"), "signed Slt/Sar must bitcast to i32:\n{wgsl}");
    for tok in ["i64", "u64"] {
        assert!(!wgsl.contains(tok), "browser profile: no `{tok}`:\n{wgsl}");
    }

    let (w, h, wgx, wgy) = (32u32, 32u32, 8u32, 8u32);
    let out = run_grid_u32(wgsl, w, h, wgx, wgy, &[]);
    assert_eq!(out.len(), (w * h) as usize, "full grid readback");

    let mut distinct = std::collections::HashSet::new();
    let mut saw_interior = false;
    let mut saw_early = false;
    for y in 0..h {
        for x in 0..w {
            let got = out[(y * w + x) as usize];
            let want = escape_ref(x as i32, y as i32);
            assert_eq!(got, want, "escape[{y}*{w}+{x}] gpu={got} != ref={want}");
            distinct.insert(got);
            if got == 50 { saw_interior = true; }
            if got <= 3 { saw_early = true; }
        }
    }
    assert!(distinct.len() >= 5, "escape-time grid must have variety; got {} distinct", distinct.len());
    assert!(saw_interior, "some pixel must be interior (== 50)");
    assert!(saw_early, "some pixel must escape early (<= 3)");
    eprintln!(
        "u32_escape_grid_executes_on_lavapipe OK: {w}x{h}, {} distinct escape counts, all == reference",
        distinct.len()
    );
}
