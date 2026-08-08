use sonatina_codegen::{Backend, Compile, OptLevel};
use sonatina_codegen::isa::cranelift::CraneliftBackend;
use sonatina_ir::{
    Linkage, Signature, Type,
    builder::ModuleBuilder,
    func_cursor::InstInserter,
    global_variable::{GlobalVariableData, GvInitializer},
    inst::{arith, cmp, control_flow, data},
    isa::{Isa, native::Native},
    module::ModuleCtx,
};
use sonatina_triple::{Architecture, OperatingSystem, TargetTriple, Vendor};

fn native_triple() -> TargetTriple {
    let arch = if cfg!(target_arch = "x86_64") {
        Architecture::X86_64
    } else if cfg!(target_arch = "aarch64") {
        Architecture::Aarch64
    } else {
        panic!("unsupported host architecture for cranelift tests");
    };
    TargetTriple::new(arch, Vendor::Unknown, OperatingSystem::Native)
}

fn native_isa() -> Native {
    Native::new(native_triple())
}

fn native_module_builder() -> ModuleBuilder {
    let isa = native_isa();
    let ctx = ModuleCtx::new(&isa);
    ModuleBuilder::new(ctx)
}

#[test]
fn cranelift_add_two_i64s() {
    let isa = native_isa();
    let is = isa.inst_set();
    let mb = {
        let ctx = ModuleCtx::new(&isa);
        ModuleBuilder::new(ctx)
    };

    let sig = Signature::new_single("add_i64", Linkage::Public, &[Type::I64, Type::I64], Type::I64);
    let func_ref = mb.declare_function(sig).unwrap();

    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);

    let a = fb.args()[0];
    let b = fb.args()[1];
    let sum = fb.insert_inst(arith::Add::new(is, a, b), Type::I64);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, sum));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let backend = CraneliftBackend::new();
    let artifact = backend.compile_module(&module).expect("compilation failed");

    let add_fn: fn(i64, i64) -> i64 = unsafe {
        let ptr = artifact.get_func_ptr::<fn(i64, i64) -> i64>("add_i64").unwrap();
        std::mem::transmute(ptr)
    };

    assert_eq!(add_fn(3, 4), 7);
    assert_eq!(add_fn(-10, 25), 15);
    assert_eq!(add_fn(0, 0), 0);
}

#[test]
fn cranelift_arithmetic_chain() {
    let isa = native_isa();
    let is = isa.inst_set();
    let mb = {
        let ctx = ModuleCtx::new(&isa);
        ModuleBuilder::new(ctx)
    };

    let sig = Signature::new_single("arith", Linkage::Public, &[Type::I32, Type::I32], Type::I32);
    let func_ref = mb.declare_function(sig).unwrap();

    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);

    let a = fb.args()[0];
    let b = fb.args()[1];
    let sum = fb.insert_inst(arith::Add::new(is, a, b), Type::I32);
    let diff = fb.insert_inst(arith::Sub::new(is, a, b), Type::I32);
    let product = fb.insert_inst(arith::Mul::new(is, sum, diff), Type::I32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, product));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let backend = CraneliftBackend::new();
    let artifact = backend.compile_module(&module).expect("compilation failed");

    let f: fn(i32, i32) -> i32 = unsafe {
        let ptr = artifact.get_func_ptr::<fn(i32, i32) -> i32>("arith").unwrap();
        std::mem::transmute(ptr)
    };

    // (5+3) * (5-3) = 8 * 2 = 16
    assert_eq!(f(5, 3), 16);
    // (10+7) * (10-7) = 17 * 3 = 51
    assert_eq!(f(10, 7), 51);
}

#[test]
fn cranelift_through_generic_compile_pipeline() {
    let isa = native_isa();
    let is = isa.inst_set();
    let mb = {
        let ctx = ModuleCtx::new(&isa);
        ModuleBuilder::new(ctx)
    };

    let sig = Signature::new_single("identity", Linkage::Public, &[Type::I64], Type::I64);
    let func_ref = mb.declare_function(sig).unwrap();

    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);
    let a = fb.args()[0];
    fb.insert_inst_no_result(control_flow::Return::new_single(is, a));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let backend = CraneliftBackend::new();
    let compile = Compile::new(module, backend).with_opt_level(OptLevel::O0);
    let artifact = compile.compile().expect("Compile<CraneliftBackend> failed");

    let f: fn(i64) -> i64 = unsafe {
        let ptr = artifact.get_func_ptr::<fn(i64) -> i64>("identity").unwrap();
        std::mem::transmute(ptr)
    };

    assert_eq!(f(42), 42);
    assert_eq!(f(-1), -1);
}

#[test]
fn cranelift_array_index() {
    let isa = native_isa();
    let is = isa.inst_set();
    let mb = native_module_builder();

    // Build: fn get_elem(idx: i64) -> i64 {
    //   let arr: [i64; 3] = [10, 20, 30]
    //   return arr[idx]
    // }
    let sig = Signature::new_single("get_elem", Linkage::Public, &[Type::I64], Type::I64);
    let func_ref = mb.declare_function(sig).unwrap();

    let arr_ty = mb.declare_array_type(Type::I64, 3);
    let arr_objref_ty = mb.objref_type(arr_ty);
    let elem_objref_ty = mb.objref_type(Type::I64);

    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);

    let idx = fb.args()[0];

    // obj.alloc [i64; 3]
    let arr = fb.insert_inst(data::ObjAlloc::new(is, arr_ty), arr_objref_ty);

    // Store elements: arr[0]=10, arr[1]=20, arr[2]=30
    let imm0 = fb.make_imm_value(0i64);
    let imm1 = fb.make_imm_value(1i64);
    let imm2 = fb.make_imm_value(2i64);

    let val10 = fb.make_imm_value(10i64);
    let val20 = fb.make_imm_value(20i64);
    let val30 = fb.make_imm_value(30i64);

    let p0 = fb.insert_inst(data::ObjIndex::new(is, arr, imm0), elem_objref_ty);
    fb.insert_inst_no_result(data::ObjStore::new(is, p0, val10));

    let p1 = fb.insert_inst(data::ObjIndex::new(is, arr, imm1), elem_objref_ty);
    fb.insert_inst_no_result(data::ObjStore::new(is, p1, val20));

    let p2 = fb.insert_inst(data::ObjIndex::new(is, arr, imm2), elem_objref_ty);
    fb.insert_inst_no_result(data::ObjStore::new(is, p2, val30));

    // Dynamic index: val = arr[idx]
    let pi = fb.insert_inst(data::ObjIndex::new(is, arr, idx), elem_objref_ty);
    let val = fb.insert_inst(data::ObjLoad::new(is, pi), Type::I64);

    fb.insert_inst_no_result(control_flow::Return::new_single(is, val));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let backend = CraneliftBackend::new();
    let artifact = backend.compile_module(&module).expect("compilation failed");

    let f: fn(i64) -> i64 = unsafe {
        let ptr = artifact.get_func_ptr::<fn(i64) -> i64>("get_elem").unwrap();
        std::mem::transmute(ptr)
    };

    assert_eq!(f(0), 10);
    assert_eq!(f(1), 20);
    assert_eq!(f(2), 30);
}

#[test]
fn cranelift_array_sum() {
    let isa = native_isa();
    let is = isa.inst_set();
    let mb = native_module_builder();

    // Build: fn sum_arr() -> i64 {
    //   let arr: [i64; 4] = [100, 200, 300, 400]
    //   return arr[0] + arr[1] + arr[2] + arr[3]
    // }
    let sig = Signature::new_single("sum_arr", Linkage::Public, &[], Type::I64);
    let func_ref = mb.declare_function(sig).unwrap();

    let arr_ty = mb.declare_array_type(Type::I64, 4);
    let arr_objref_ty = mb.objref_type(arr_ty);
    let elem_objref_ty = mb.objref_type(Type::I64);

    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);

    let arr = fb.insert_inst(data::ObjAlloc::new(is, arr_ty), arr_objref_ty);

    // Store 4 elements
    for (i, val) in [100i64, 200, 300, 400].iter().enumerate() {
        let idx = fb.make_imm_value(i as i64);
        let imm_val = fb.make_imm_value(*val);
        let p = fb.insert_inst(data::ObjIndex::new(is, arr, idx), elem_objref_ty);
        fb.insert_inst_no_result(data::ObjStore::new(is, p, imm_val));
    }

    // Load and sum
    let mut acc = {
        let idx = fb.make_imm_value(0i64);
        let p = fb.insert_inst(data::ObjIndex::new(is, arr, idx), elem_objref_ty);
        fb.insert_inst(data::ObjLoad::new(is, p), Type::I64)
    };
    for i in 1..4 {
        let idx = fb.make_imm_value(i as i64);
        let p = fb.insert_inst(data::ObjIndex::new(is, arr, idx), elem_objref_ty);
        let elem = fb.insert_inst(data::ObjLoad::new(is, p), Type::I64);
        acc = fb.insert_inst(arith::Add::new(is, acc, elem), Type::I64);
    }

    fb.insert_inst_no_result(control_flow::Return::new_single(is, acc));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let backend = CraneliftBackend::new();
    let artifact = backend.compile_module(&module).expect("compilation failed");

    let f: fn() -> i64 = unsafe {
        let ptr = artifact.get_func_ptr::<fn() -> i64>("sum_arr").unwrap();
        std::mem::transmute(ptr)
    };

    assert_eq!(f(), 1000);
}

#[test]
fn cranelift_const_ref_array() {
    let isa = native_isa();
    let is = isa.inst_set();
    let mb = native_module_builder();

    // Declare a global constant array: [i64; 4] = [10, 20, 30, 40]
    let arr_ty = mb.declare_array_type(Type::I64, 4);
    let gv = mb.declare_gv(GlobalVariableData::constant(
        "ROUND_CONSTANTS".to_string(),
        arr_ty,
        Linkage::Private,
        GvInitializer::Array(vec![
            GvInitializer::Immediate(sonatina_ir::Immediate::I64(10)),
            GvInitializer::Immediate(sonatina_ir::Immediate::I64(20)),
            GvInitializer::Immediate(sonatina_ir::Immediate::I64(30)),
            GvInitializer::Immediate(sonatina_ir::Immediate::I64(40)),
        ]),
    ));

    // fn sum_consts(idx: i64) -> i64 { ROUND_CONSTANTS[idx] }
    let sig = Signature::new_single("get_const", Linkage::Public, &[Type::I64], Type::I64);
    let func_ref = mb.declare_function(sig).unwrap();

    let constref_ty = mb.make_compound(sonatina_ir::types::CompoundType::ConstRef(arr_ty));
    let constref_type = Type::Compound(constref_ty);
    let elem_constref_ty = mb.make_compound(sonatina_ir::types::CompoundType::ConstRef(Type::I64));
    let elem_constref_type = Type::Compound(elem_constref_ty);

    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);

    let idx = fb.args()[0];

    // const.ref → pointer to global array
    let arr_ref = fb.insert_inst(data::ConstRef::new(is, gv.into()), constref_type);
    // const.index → pointer to element
    let elem_ref = fb.insert_inst(data::ConstIndex::new(is, arr_ref, idx), elem_constref_type);
    // const.load → load the element value
    let val = fb.insert_inst(data::ConstLoad::new(is, elem_ref), Type::I64);

    fb.insert_inst_no_result(control_flow::Return::new_single(is, val));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let backend = CraneliftBackend::new();
    let artifact = backend.compile_module(&module).expect("compilation failed");

    let f: fn(i64) -> i64 = unsafe {
        let ptr = artifact.get_func_ptr::<fn(i64) -> i64>("get_const").unwrap();
        std::mem::transmute(ptr)
    };

    assert_eq!(f(0), 10);
    assert_eq!(f(1), 20);
    assert_eq!(f(2), 30);
    assert_eq!(f(3), 40);
}

#[test]
fn cranelift_poseidon_loop_with_const_array() {
    let isa = native_isa();
    let is = isa.inst_set();
    let mb = native_module_builder();

    // Global constant round array: [i64; 4] = [3, 5, 7, 11]
    let arr_ty = mb.declare_array_type(Type::I64, 4);
    let gv = mb.declare_gv(GlobalVariableData::constant(
        "ROUND_CONSTS".to_string(),
        arr_ty,
        Linkage::Private,
        GvInitializer::Array(vec![
            GvInitializer::Immediate(sonatina_ir::Immediate::I64(3)),
            GvInitializer::Immediate(sonatina_ir::Immediate::I64(5)),
            GvInitializer::Immediate(sonatina_ir::Immediate::I64(7)),
            GvInitializer::Immediate(sonatina_ir::Immediate::I64(11)),
        ]),
    ));

    // fn poseidon_sum() -> i64 {
    //   let C = ROUND_CONSTS;  // const array
    //   let mut acc: i64 = 1;
    //   for i in 0..4 {
    //     let c = C[i];
    //     acc = (acc + c) * (acc + c) + (acc + c);  // sigma(acc + c)
    //   }
    //   return acc;
    // }
    let sig = Signature::new_single("poseidon_sum", Linkage::Public, &[], Type::I64);
    let func_ref = mb.declare_function(sig).unwrap();

    let constref_ty = mb.make_compound(sonatina_ir::types::CompoundType::ConstRef(arr_ty));
    let constref_type = Type::Compound(constref_ty);
    let elem_constref_ty = mb.make_compound(sonatina_ir::types::CompoundType::ConstRef(Type::I64));
    let elem_constref_type = Type::Compound(elem_constref_ty);

    let mut fb = mb.func_builder::<InstInserter>(func_ref);

    let entry = fb.append_block();
    let loop_header = fb.append_block();
    let loop_body = fb.append_block();
    let exit = fb.append_block();

    // entry: jump to loop with (acc=1, i=0)
    fb.switch_to_block(entry);
    let init_acc = fb.make_imm_value(1i64);
    let init_i = fb.make_imm_value(0i64);
    fb.insert_inst_no_result(control_flow::Jump::new(is, loop_header));

    // loop_header: phi(acc, i), check i < 4
    fb.switch_to_block(loop_header);
    let acc_phi = fb.insert_inst(
        control_flow::Phi::new(is, vec![(init_acc, entry)]),
        Type::I64,
    );
    let i_phi = fb.insert_inst(
        control_flow::Phi::new(is, vec![(init_i, entry)]),
        Type::I64,
    );
    let four = fb.make_imm_value(4i64);
    let cond = fb.insert_inst(cmp::Lt::new(is, i_phi, four), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, cond, loop_body, exit));

    // loop_body: c = C[i], sigma(acc + c), i++
    fb.switch_to_block(loop_body);
    let arr_ref = fb.insert_inst(data::ConstRef::new(is, gv.into()), constref_type);
    let elem_ref = fb.insert_inst(data::ConstIndex::new(is, arr_ref, i_phi), elem_constref_type);
    let c = fb.insert_inst(data::ConstLoad::new(is, elem_ref), Type::I64);
    let sum = fb.insert_inst(arith::Add::new(is, acc_phi, c), Type::I64);
    let sq = fb.insert_inst(arith::Mul::new(is, sum, sum), Type::I64);
    let new_acc = fb.insert_inst(arith::Add::new(is, sq, sum), Type::I64);
    let one = fb.make_imm_value(1i64);
    let new_i = fb.insert_inst(arith::Add::new(is, i_phi, one), Type::I64);

    // Update phis and jump back
    fb.append_phi_arg(acc_phi, new_acc, loop_body);
    fb.append_phi_arg(i_phi, new_i, loop_body);
    fb.insert_inst_no_result(control_flow::Jump::new(is, loop_header));

    // exit: return acc
    fb.switch_to_block(exit);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, acc_phi));

    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let backend = CraneliftBackend::new();
    let artifact = backend.compile_module(&module).expect("compilation failed");

    let f: fn() -> i64 = unsafe {
        let ptr = artifact.get_func_ptr::<fn() -> i64>("poseidon_sum").unwrap();
        std::mem::transmute(ptr)
    };

    // Manual computation:
    // i=0: acc=1, c=3, sum=4, sq=16, new_acc=20
    // i=1: acc=20, c=5, sum=25, sq=625, new_acc=650
    // i=2: acc=650, c=7, sum=657, sq=431649, new_acc=432306
    // i=3: acc=432306, c=11, sum=432317, sq=186897988489, new_acc=186898420806
    let result = f();
    assert_eq!(result, 186898420806, "poseidon_sum with const array rounds");
}

/// Known-answer cross-target test: same Sonatina IR compiled to Cranelift, WASM,
/// and SPIR-V. All three validated. Cranelift+WASM executed and compared.
/// SPIR-V validated with spirv-val (execution requires GPU runtime).
#[test]
fn cross_target_three_backend_known_answer() {
    use sonatina_codegen::isa::wasm::WasmBackend;
    use sonatina_codegen::isa::spirv::SpirvBackend;

    let isa = native_isa();
    let is = isa.inst_set();
    let mb = native_module_builder();

    // fn compute(a: i64, b: i64) -> i64 { (a + b) * (a - b) + 42 }
    let sig = Signature::new_single("compute", Linkage::Public, &[Type::I64, Type::I64], Type::I64);
    let func_ref = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);
    let a = fb.args()[0];
    let b = fb.args()[1];
    let sum = fb.insert_inst(arith::Add::new(is, a, b), Type::I64);
    let diff = fb.insert_inst(arith::Sub::new(is, a, b), Type::I64);
    let prod = fb.insert_inst(arith::Mul::new(is, sum, diff), Type::I64);
    let c42 = fb.make_imm_value(42i64);
    let result = fb.insert_inst(arith::Add::new(is, prod, c42), Type::I64);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, result));
    fb.seal_all();
    fb.finish();

    let module = mb.build();

    // Cranelift execution
    let cranelift_backend = CraneliftBackend::new();
    let cranelift_artifact = cranelift_backend.compile_module(&module).expect("cranelift failed");
    let cranelift_fn: fn(i64, i64) -> i64 = unsafe {
        let ptr = cranelift_artifact.get_func_ptr::<fn(i64, i64) -> i64>("compute").unwrap();
        std::mem::transmute(ptr)
    };

    // WASM execution
    let wasm_backend = WasmBackend::new();
    let wasm_artifact = wasm_backend.compile_module(&module).expect("wasm failed");
    wasmparser::validate(&wasm_artifact.bytes).expect("invalid wasm");
    let engine = wasmtime::Engine::default();
    let wasm_module = wasmtime::Module::new(&engine, &wasm_artifact.bytes).expect("load failed");
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = wasmtime::Instance::new(&mut store, &wasm_module, &[]).expect("instantiate failed");
    let wasm_fn = instance.get_typed_func::<(i64, i64), i64>(&mut store, "compute").expect("export");

    // SPIR-V compilation and validation
    let spirv_backend = SpirvBackend::new();
    let spirv_artifact = spirv_backend.compile_module(&module).expect("spirv failed");
    assert_eq!(spirv_artifact.words[0], 0x07230203, "SPIR-V magic number");

    let tmp = std::env::temp_dir().join("cross_target_test.spv");
    std::fs::write(&tmp, spirv_artifact.as_bytes()).unwrap();
    if let Ok(output) = std::process::Command::new("spirv-val").arg(tmp.to_str().unwrap()).output() {
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            eprintln!("spirv-val: {stderr}");
        }
        assert!(output.status.success(), "SPIR-V module should validate");
    }
    let _ = std::fs::remove_file(&tmp);

    // Known-answer comparison: Cranelift vs WASM (SPIR-V validated but not executed — needs GPU)
    for (a, b) in [(10, 3), (100, 7), (5, 5), (0, 0), (1000, 1)] {
        let cranelift_result = cranelift_fn(a, b);
        let wasm_result = wasm_fn.call(&mut store, (a, b)).expect("wasm call failed");
        assert_eq!(
            cranelift_result, wasm_result,
            "Cranelift and WASM should produce same result for compute({a}, {b})"
        );
        let expected = a * a - b * b + 42;
        assert_eq!(cranelift_result, expected, "compute({a}, {b}) should be {expected}");
    }
}

// ---------------------------------------------------------------------------
// Float numeric intrinsics: `Fabs`/`Fmin`/`Fmax`/`Fclamp`.
//
// PINNED semantics for Fmin/Fmax: the "WebAssembly rules" (IEEE 754-2019
// `minimum`/`maximum`): NaN-propagating, and -0.0 is treated as strictly less
// than +0.0 regardless of argument order. This is exactly wasm's `f32.min`/
// `f32.max` and (by its own doc comment) cranelift's `fmin`/`fmax`, so those
// two backends are expected to agree bit-for-bit on every input, including
// NaN/-0.0 edge cases.
//
// naga/SPIR-V is NOT included in the bit-for-bit differential below, and is
// not exercised by this synthetic module at all (`SpirvBackend::compile_module`'s
// kernel ABI only accepts an i32/i64 return, which these raw-`f32`-returning
// functions don't fit). This is now a pure reachability gap, not a semantics
// one: naga/SPIR-V's `Fmin`/`Fmax`/`Fabs`/`Fclamp` lowering (`emit_exact_fminmax`,
// `crates/codegen/src/isa/spirv/mod.rs`) is a branch-free integer
// key-compare-and-select expansion that is pinned-exact ("WebAssembly rules"),
// matching wasm/cranelift bit-for-bit including NaN/-0.0 — it no longer uses
// GLSL.std.450 `FMin`/`FMax`, whose NaN/-0.0 behavior is implementation-defined
// by spec (that WAS the OPEN DECISION; see docs/numeric-intrinsics-semantics.md,
// now resolved). The exact GPU expansion is validated structurally (legal
// SPIR-V/WGSL, no branch/phi, only `select`/`OpSelect`) via the real Fe demo
// pipeline (`demos/sketches/cga3d` etc. via `fe-codegen`'s `demo_compile_gate.rs`),
// which returns i32 RGBA and so fits the kernel/render ABI; no GPU adapter
// exists in this sandbox to execute the shader and observe real driver
// behavior either way.

/// Reference implementation of the PINNED "WebAssembly rules" float minimum,
/// independent of both `sonatina_ir::interpret`'s copy and the backends under
/// test (a from-scratch transcription of the spec, so a bug shared between
/// the interpreter and this oracle would still be caught here).
fn wasm_rules_fmin_oracle(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        return f32::from_bits(0x7fc0_0000);
    }
    if a == 0.0 && b == 0.0 {
        return if a.is_sign_negative() || b.is_sign_negative() {
            -0.0
        } else {
            0.0
        };
    }
    if a < b { a } else { b }
}

fn wasm_rules_fmax_oracle(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        return f32::from_bits(0x7fc0_0000);
    }
    if a == 0.0 && b == 0.0 {
        return if a.is_sign_negative() && b.is_sign_negative() {
            -0.0
        } else {
            0.0
        };
    }
    if a > b { a } else { b }
}

/// Build a module with `abs(x)`, `min(a,b)`, `max(a,b)`, `clamp(x,lo,hi)`, all
/// `f32 -> f32`, using the new `Fabs`/`Fmin`/`Fmax`/`Fclamp` Sonatina ops.
fn build_f32_intrinsics_module() -> sonatina_ir::Module {
    let mb = native_module_builder();
    let isa = native_isa();
    let is = isa.inst_set();

    let sig = Signature::new_single("f32_abs", Linkage::Public, &[Type::F32], Type::F32);
    let func_ref = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);
    let x = fb.args()[0];
    let result = fb.insert_inst(arith::Fabs::new(is, x), Type::F32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, result));
    fb.seal_all();
    fb.finish();

    let sig = Signature::new_single(
        "f32_min",
        Linkage::Public,
        &[Type::F32, Type::F32],
        Type::F32,
    );
    let func_ref = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);
    let a = fb.args()[0];
    let b = fb.args()[1];
    let result = fb.insert_inst(arith::Fmin::new(is, a, b), Type::F32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, result));
    fb.seal_all();
    fb.finish();

    let sig = Signature::new_single(
        "f32_max",
        Linkage::Public,
        &[Type::F32, Type::F32],
        Type::F32,
    );
    let func_ref = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);
    let a = fb.args()[0];
    let b = fb.args()[1];
    let result = fb.insert_inst(arith::Fmax::new(is, a, b), Type::F32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, result));
    fb.seal_all();
    fb.finish();

    let sig = Signature::new_single(
        "f32_clamp",
        Linkage::Public,
        &[Type::F32, Type::F32, Type::F32],
        Type::F32,
    );
    let func_ref = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);
    let x = fb.args()[0];
    let lo = fb.args()[1];
    let hi = fb.args()[2];
    let result = fb.insert_inst(arith::Fclamp::new(is, x, lo, hi), Type::F32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, result));
    fb.seal_all();
    fb.finish();

    mb.build()
}

/// Oracle test (VERIFY item 4): native/cranelift executes `Fabs`/`Fmin`/
/// `Fmax`/`Fclamp` correctly on ordinary values, bit-exact vs a plain Rust
/// reference.
#[test]
fn cranelift_f32_abs_min_max_clamp_oracle() {
    let module = build_f32_intrinsics_module();
    let backend = CraneliftBackend::new();
    let artifact = backend.compile_module(&module).expect("cranelift compile");

    let abs: fn(f32) -> f32 = unsafe {
        std::mem::transmute(artifact.get_func_ptr::<fn(f32) -> f32>("f32_abs").unwrap())
    };
    let min: fn(f32, f32) -> f32 = unsafe {
        std::mem::transmute(
            artifact
                .get_func_ptr::<fn(f32, f32) -> f32>("f32_min")
                .unwrap(),
        )
    };
    let max: fn(f32, f32) -> f32 = unsafe {
        std::mem::transmute(
            artifact
                .get_func_ptr::<fn(f32, f32) -> f32>("f32_max")
                .unwrap(),
        )
    };
    let clamp: fn(f32, f32, f32) -> f32 = unsafe {
        std::mem::transmute(
            artifact
                .get_func_ptr::<fn(f32, f32, f32) -> f32>("f32_clamp")
                .unwrap(),
        )
    };

    // `.to_bits()`, not float `==`: IEEE equality treats `-0.0 == +0.0`, which
    // would silently hide a sign-of-zero divergence (the review's gap).
    for x in [-3.5f32, 3.5, 0.0, -0.0, -1.0, 1.0, 42.25] {
        assert_eq!(abs(x).to_bits(), x.abs().to_bits(), "abs({x})");
    }
    for (a, b) in [(1.0f32, 2.0), (2.0, 1.0), (-5.0, 5.0), (3.0, 3.0), (-2.5, -7.5)] {
        assert_eq!(min(a, b).to_bits(), wasm_rules_fmin_oracle(a, b).to_bits(), "min({a}, {b})");
        assert_eq!(max(a, b).to_bits(), wasm_rules_fmax_oracle(a, b).to_bits(), "max({a}, {b})");
    }
    let clamp_cases: &[(f32, f32, f32, f32)] = &[
        (5.0, 0.0, 1.0, 1.0),
        (-5.0, 0.0, 1.0, 0.0),
        (0.5, 0.0, 1.0, 0.5),
        (0.0, 0.0, 1.0, 0.0),
        (1.0, 0.0, 1.0, 1.0),
        (100.0, -10.0, 10.0, 10.0),
        // lo > hi (the review's test gap): composed min(max(x,lo),hi) is
        // deterministically `hi`, never GLSL.std.450 `FClamp` poison.
        (5.0, 10.0, -10.0, -10.0),
        (-100.0, 10.0, -10.0, -10.0),
    ];
    for &(x, lo, hi, expected) in clamp_cases {
        assert_eq!(clamp(x, lo, hi).to_bits(), expected.to_bits(), "clamp({x}, {lo}, {hi})");
    }
}

/// THE SHARP EDGE (VERIFY item 5): cross-backend differential over NaN/-0.0/
/// +-inf edge inputs for `Fmin`/`Fmax`, plus a `Fclamp` `lo > hi` case. Wasm
/// and cranelift are asserted to agree bit-for-bit with each other AND with
/// the from-scratch oracle above, because both backends implement the same
/// PINNED "WebAssembly rules". The naga/SPIR-V path is not reachable from
/// this synthetic module at all (see the comment below) so it is not part of
/// this differential; its exactness is now a structural property of its
/// lowering (branch-free integer expansion, no GLSL.std.450 FMin/FMax
/// dependency) validated elsewhere — see the module doc comment above and
/// docs/numeric-intrinsics-semantics.md (OPEN DECISION, now RESOLVED).
#[test]
fn cross_backend_f32_min_max_nan_zero_inf_differential() {
    use sonatina_codegen::isa::wasm::WasmBackend;

    let module = build_f32_intrinsics_module();

    let cranelift_backend = CraneliftBackend::new();
    let cranelift_artifact = cranelift_backend
        .compile_module(&module)
        .expect("cranelift compile");
    let cranelift_min: fn(f32, f32) -> f32 = unsafe {
        std::mem::transmute(
            cranelift_artifact
                .get_func_ptr::<fn(f32, f32) -> f32>("f32_min")
                .unwrap(),
        )
    };
    let cranelift_max: fn(f32, f32) -> f32 = unsafe {
        std::mem::transmute(
            cranelift_artifact
                .get_func_ptr::<fn(f32, f32) -> f32>("f32_max")
                .unwrap(),
        )
    };

    let wasm_backend = WasmBackend::new();
    let wasm_artifact = wasm_backend.compile_module(&module).expect("wasm compile");
    wasmparser::validate(&wasm_artifact.bytes).expect("valid wasm");
    let engine = wasmtime::Engine::default();
    let wasm_module = wasmtime::Module::new(&engine, &wasm_artifact.bytes).expect("load wasm");
    let mut store = wasmtime::Store::new(&engine, ());
    let instance =
        wasmtime::Instance::new(&mut store, &wasm_module, &[]).expect("instantiate wasm");
    let wasm_min = instance
        .get_typed_func::<(f32, f32), f32>(&mut store, "f32_min")
        .expect("f32_min export");
    let wasm_max = instance
        .get_typed_func::<(f32, f32), f32>(&mut store, "f32_max")
        .expect("f32_max export");
    let cranelift_clamp: fn(f32, f32, f32) -> f32 = unsafe {
        std::mem::transmute(
            cranelift_artifact
                .get_func_ptr::<fn(f32, f32, f32) -> f32>("f32_clamp")
                .unwrap(),
        )
    };
    let wasm_clamp = instance
        .get_typed_func::<(f32, f32, f32), f32>(&mut store, "f32_clamp")
        .expect("f32_clamp export");

    // naga/SPIR-V is NOT exercised here: `SpirvBackend::compile_module`'s
    // "kernel" entry ABI only accepts an i32 (u32 word) or i64 return value
    // (this synthetic module's raw `f32`-returning functions don't fit that
    // envelope; there is no raw bitcast op wired to the SPIR-V backend to
    // route around it). The real naga/SPIR-V validation evidence for
    // `Fmin`/`Fmax`/`Fabs`/`Fclamp` comes from the actual Fe demo pipeline
    // (`demos/sketches/cga3d`, `demos/sketches/desargues` via
    // `fe-codegen`'s `demo_compile_gate.rs`), which returns i32 RGBA and so
    // fits the kernel/render ABI, and is validated there with
    // `naga::valid::Validator` — see docs/numeric-intrinsics-semantics.md.

    let nan_a = f32::from_bits(0x7fc0_1234);
    let nan_b = f32::NAN;
    let edge_inputs: &[(f32, f32)] = &[
        (1.0, 2.0),
        (2.0, 1.0),
        (-1.0, 1.0),
        (0.0, 0.0),
        (0.0, -0.0),
        (-0.0, 0.0),
        (-0.0, -0.0),
        (f32::INFINITY, 1.0),
        (f32::NEG_INFINITY, 1.0),
        (f32::INFINITY, f32::NEG_INFINITY),
        (nan_a, 1.0),
        (1.0, nan_a),
        (nan_a, nan_b),
        (nan_a, f32::INFINITY),
    ];

    for &(a, b) in edge_inputs {
        let oracle_min = wasm_rules_fmin_oracle(a, b);
        let oracle_max = wasm_rules_fmax_oracle(a, b);
        let cl_min = cranelift_min(a, b);
        let cl_max = cranelift_max(a, b);
        let wa_min = wasm_min.call(&mut store, (a, b)).expect("wasm min call");
        let wa_max = wasm_max.call(&mut store, (a, b)).expect("wasm max call");

        if oracle_min.is_nan() {
            // Sign/payload of a NaN result is spec-unspecified even under the
            // pinned "WebAssembly rules"; only "is it NaN at all" is pinned.
            assert!(cl_min.is_nan(), "cranelift min({a:?}, {b:?}) should be NaN, got {cl_min:?}");
            assert!(wa_min.is_nan(), "wasm min({a:?}, {b:?}) should be NaN, got {wa_min:?}");
        } else {
            assert_eq!(
                cl_min.to_bits(),
                oracle_min.to_bits(),
                "cranelift min({a:?}, {b:?}) diverges from the WebAssembly-rules oracle"
            );
            assert_eq!(
                wa_min.to_bits(),
                oracle_min.to_bits(),
                "wasm min({a:?}, {b:?}) diverges from the WebAssembly-rules oracle"
            );
            assert_eq!(
                cl_min.to_bits(),
                wa_min.to_bits(),
                "cranelift and wasm min({a:?}, {b:?}) disagree bit-for-bit"
            );
        }

        if oracle_max.is_nan() {
            assert!(cl_max.is_nan(), "cranelift max({a:?}, {b:?}) should be NaN, got {cl_max:?}");
            assert!(wa_max.is_nan(), "wasm max({a:?}, {b:?}) should be NaN, got {wa_max:?}");
        } else {
            assert_eq!(
                cl_max.to_bits(),
                oracle_max.to_bits(),
                "cranelift max({a:?}, {b:?}) diverges from the WebAssembly-rules oracle"
            );
            assert_eq!(
                wa_max.to_bits(),
                oracle_max.to_bits(),
                "wasm max({a:?}, {b:?}) diverges from the WebAssembly-rules oracle"
            );
            assert_eq!(
                cl_max.to_bits(),
                wa_max.to_bits(),
                "cranelift and wasm max({a:?}, {b:?}) disagree bit-for-bit"
            );
        }
    }

    // `Fclamp`'s `lo > hi` case (the review's test gap): the composed
    // `min(max(x, lo), hi)` definition makes this deterministically `hi` on
    // every backend and every `x` (`max(x, lo) >= lo > hi`, so the outer
    // `min` always picks `hi`) -- never the GLSL.std.450 `FClamp` poison this
    // feature exists to avoid. Checked via `.to_bits()` (not `==`) so a
    // `-0.0`-vs-`+0.0` divergence in the "always hi" answer cannot hide
    // behind IEEE equality.
    let clamp_lo_gt_hi_inputs: &[(f32, f32, f32)] = &[
        (5.0, 10.0, -10.0),
        (-5.0, 10.0, -10.0),
        (0.0, 1.0, -1.0),
        (-0.0, 1.0, -1.0),
        (f32::INFINITY, 1.0, -1.0),
        (f32::NEG_INFINITY, 1.0, -1.0),
    ];
    for &(x, lo, hi) in clamp_lo_gt_hi_inputs {
        let oracle = wasm_rules_fmin_oracle(wasm_rules_fmax_oracle(x, lo), hi);
        assert_eq!(
            oracle.to_bits(),
            hi.to_bits(),
            "sanity: lo > hi composed clamp oracle must equal hi for clamp({x}, {lo}, {hi})"
        );
        let cl = cranelift_clamp(x, lo, hi);
        let wa = wasm_clamp.call(&mut store, (x, lo, hi)).expect("wasm clamp call");
        assert_eq!(
            cl.to_bits(),
            oracle.to_bits(),
            "cranelift clamp({x}, {lo}, {hi}) [lo > hi] diverges from the composed oracle"
        );
        assert_eq!(
            wa.to_bits(),
            oracle.to_bits(),
            "wasm clamp({x}, {lo}, {hi}) [lo > hi] diverges from the composed oracle"
        );
        assert_eq!(
            cl.to_bits(),
            wa.to_bits(),
            "cranelift and wasm clamp({x}, {lo}, {hi}) [lo > hi] disagree bit-for-bit"
        );
    }
}

/// Build a module with `min_relaxed(a,b)`, `max_relaxed(a,b)`, `f32 -> f32`,
/// using the new `FminRelaxed`/`FmaxRelaxed` Sonatina ops. Mirrors
/// `build_f32_intrinsics_module` above.
fn build_f32_relaxed_intrinsics_module() -> sonatina_ir::Module {
    let mb = native_module_builder();
    let isa = native_isa();
    let is = isa.inst_set();

    let sig = Signature::new_single(
        "f32_min_relaxed",
        Linkage::Public,
        &[Type::F32, Type::F32],
        Type::F32,
    );
    let func_ref = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);
    let a = fb.args()[0];
    let b = fb.args()[1];
    let result = fb.insert_inst(arith::FminRelaxed::new(is, a, b), Type::F32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, result));
    fb.seal_all();
    fb.finish();

    let sig = Signature::new_single(
        "f32_max_relaxed",
        Linkage::Public,
        &[Type::F32, Type::F32],
        Type::F32,
    );
    let func_ref = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);
    let a = fb.args()[0];
    let b = fb.args()[1];
    let result = fb.insert_inst(arith::FmaxRelaxed::new(is, a, b), Type::F32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, result));
    fb.seal_all();
    fb.finish();

    mb.build()
}

/// PROVE item 3 (slice 1, float-semantics type API): on wasm and cranelift,
/// `FminRelaxed`/`FmaxRelaxed` lower to the EXACT SAME native instruction as
/// `Fmin`/`Fmax` (`f32.min`/`f32.max`, `fmin`/`fmax`) -- the relaxed contract
/// is satisfied trivially because native already IS a conforming
/// implementation, with zero new backend surface. This is a differential/
/// oracle test proving relaxed == exact bit-for-bit on BOTH backends, over
/// the SAME edge-input set (normal values, NaN, +-0.0, +-inf) used by
/// `cross_backend_f32_min_max_nan_zero_inf_differential` above: if this ever
/// diverges, it means a backend's `FminRelaxed`/`FmaxRelaxed` arm stopped
/// reusing the exact op's native instruction (e.g. someone "improved" it into
/// a different lowering), which would be a regression even though it stays
/// within the relaxed contract's latitude, because wasm/cranelift's whole
/// design rationale is "zero new backend surface, same instruction".
#[test]
fn cranelift_and_wasm_f32_relaxed_minmax_equals_exact_native() {
    use sonatina_codegen::isa::wasm::WasmBackend;

    let exact_module = build_f32_intrinsics_module();
    let relaxed_module = build_f32_relaxed_intrinsics_module();

    let cranelift_backend = CraneliftBackend::new();
    let cranelift_exact = cranelift_backend
        .compile_module(&exact_module)
        .expect("cranelift exact compile");
    let cranelift_relaxed = cranelift_backend
        .compile_module(&relaxed_module)
        .expect("cranelift relaxed compile");

    let cl_min: fn(f32, f32) -> f32 = unsafe {
        std::mem::transmute(cranelift_exact.get_func_ptr::<fn(f32, f32) -> f32>("f32_min").unwrap())
    };
    let cl_max: fn(f32, f32) -> f32 = unsafe {
        std::mem::transmute(cranelift_exact.get_func_ptr::<fn(f32, f32) -> f32>("f32_max").unwrap())
    };
    let cl_min_relaxed: fn(f32, f32) -> f32 = unsafe {
        std::mem::transmute(
            cranelift_relaxed
                .get_func_ptr::<fn(f32, f32) -> f32>("f32_min_relaxed")
                .unwrap(),
        )
    };
    let cl_max_relaxed: fn(f32, f32) -> f32 = unsafe {
        std::mem::transmute(
            cranelift_relaxed
                .get_func_ptr::<fn(f32, f32) -> f32>("f32_max_relaxed")
                .unwrap(),
        )
    };

    let wasm_backend = WasmBackend::new();
    let wasm_exact_artifact = wasm_backend.compile_module(&exact_module).expect("wasm exact compile");
    let wasm_relaxed_artifact = wasm_backend.compile_module(&relaxed_module).expect("wasm relaxed compile");
    wasmparser::validate(&wasm_exact_artifact.bytes).expect("valid exact wasm");
    wasmparser::validate(&wasm_relaxed_artifact.bytes).expect("valid relaxed wasm");
    let engine = wasmtime::Engine::default();

    let exact_wasm_module = wasmtime::Module::new(&engine, &wasm_exact_artifact.bytes).expect("load exact wasm");
    let mut exact_store = wasmtime::Store::new(&engine, ());
    let exact_instance =
        wasmtime::Instance::new(&mut exact_store, &exact_wasm_module, &[]).expect("instantiate exact wasm");
    let wasm_min = exact_instance
        .get_typed_func::<(f32, f32), f32>(&mut exact_store, "f32_min")
        .expect("f32_min export");
    let wasm_max = exact_instance
        .get_typed_func::<(f32, f32), f32>(&mut exact_store, "f32_max")
        .expect("f32_max export");

    let relaxed_wasm_module = wasmtime::Module::new(&engine, &wasm_relaxed_artifact.bytes).expect("load relaxed wasm");
    let mut relaxed_store = wasmtime::Store::new(&engine, ());
    let relaxed_instance =
        wasmtime::Instance::new(&mut relaxed_store, &relaxed_wasm_module, &[]).expect("instantiate relaxed wasm");
    let wasm_min_relaxed = relaxed_instance
        .get_typed_func::<(f32, f32), f32>(&mut relaxed_store, "f32_min_relaxed")
        .expect("f32_min_relaxed export");
    let wasm_max_relaxed = relaxed_instance
        .get_typed_func::<(f32, f32), f32>(&mut relaxed_store, "f32_max_relaxed")
        .expect("f32_max_relaxed export");

    let nan_a = f32::from_bits(0x7fc0_1234);
    let nan_b = f32::NAN;
    let edge_inputs: &[(f32, f32)] = &[
        (1.0, 2.0),
        (2.0, 1.0),
        (-1.0, 1.0),
        (0.0, 0.0),
        (0.0, -0.0),
        (-0.0, 0.0),
        (-0.0, -0.0),
        (f32::INFINITY, 1.0),
        (f32::NEG_INFINITY, 1.0),
        (f32::INFINITY, f32::NEG_INFINITY),
        (nan_a, 1.0),
        (1.0, nan_a),
        (nan_a, nan_b),
        (nan_a, f32::INFINITY),
    ];

    for &(a, b) in edge_inputs {
        let cl_min_v = cl_min(a, b);
        let cl_min_relaxed_v = cl_min_relaxed(a, b);
        let cl_max_v = cl_max(a, b);
        let cl_max_relaxed_v = cl_max_relaxed(a, b);
        let wa_min_v = wasm_min.call(&mut exact_store, (a, b)).expect("wasm min call");
        let wa_min_relaxed_v = wasm_min_relaxed
            .call(&mut relaxed_store, (a, b))
            .expect("wasm min_relaxed call");
        let wa_max_v = wasm_max.call(&mut exact_store, (a, b)).expect("wasm max call");
        let wa_max_relaxed_v = wasm_max_relaxed
            .call(&mut relaxed_store, (a, b))
            .expect("wasm max_relaxed call");

        // NaN payload/sign is spec-unspecified even for the EXACT op, so
        // compare NaN-ness for NaN cases and bits otherwise (same convention
        // as `cross_backend_f32_min_max_nan_zero_inf_differential`).
        if cl_min_v.is_nan() {
            assert!(cl_min_relaxed_v.is_nan(), "cranelift min_relaxed({a:?},{b:?}) should be NaN");
            assert!(wa_min_v.is_nan() && wa_min_relaxed_v.is_nan(), "wasm min/min_relaxed({a:?},{b:?}) should be NaN");
        } else {
            assert_eq!(
                cl_min_v.to_bits(), cl_min_relaxed_v.to_bits(),
                "cranelift: relaxed min({a:?},{b:?}) must equal exact min bit-for-bit (same native instruction)"
            );
            assert_eq!(
                wa_min_v.to_bits(), wa_min_relaxed_v.to_bits(),
                "wasm: relaxed min({a:?},{b:?}) must equal exact min bit-for-bit (same native instruction)"
            );
        }
        if cl_max_v.is_nan() {
            assert!(cl_max_relaxed_v.is_nan(), "cranelift max_relaxed({a:?},{b:?}) should be NaN");
            assert!(wa_max_v.is_nan() && wa_max_relaxed_v.is_nan(), "wasm max/max_relaxed({a:?},{b:?}) should be NaN");
        } else {
            assert_eq!(
                cl_max_v.to_bits(), cl_max_relaxed_v.to_bits(),
                "cranelift: relaxed max({a:?},{b:?}) must equal exact max bit-for-bit (same native instruction)"
            );
            assert_eq!(
                wa_max_v.to_bits(), wa_max_relaxed_v.to_bits(),
                "wasm: relaxed max({a:?},{b:?}) must equal exact max bit-for-bit (same native instruction)"
            );
        }
    }
}

/// Build a module with `floor(x)`, `ceil(x)`, `trunc(x)`, `round(x)`, all
/// `f32 -> f32`, using the new `Ffloor`/`Fceil`/`Ftrunc`/`Fround` Sonatina
/// ops. Mirrors `build_f32_intrinsics_module` above.
fn build_f32_rounding_module() -> sonatina_ir::Module {
    let mb = native_module_builder();
    let isa = native_isa();
    let is = isa.inst_set();

    let sig = Signature::new_single("f32_floor", Linkage::Public, &[Type::F32], Type::F32);
    let func_ref = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);
    let x = fb.args()[0];
    let result = fb.insert_inst(arith::Ffloor::new(is, x), Type::F32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, result));
    fb.seal_all();
    fb.finish();

    let sig = Signature::new_single("f32_ceil", Linkage::Public, &[Type::F32], Type::F32);
    let func_ref = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);
    let x = fb.args()[0];
    let result = fb.insert_inst(arith::Fceil::new(is, x), Type::F32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, result));
    fb.seal_all();
    fb.finish();

    let sig = Signature::new_single("f32_trunc", Linkage::Public, &[Type::F32], Type::F32);
    let func_ref = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);
    let x = fb.args()[0];
    let result = fb.insert_inst(arith::Ftrunc::new(is, x), Type::F32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, result));
    fb.seal_all();
    fb.finish();

    let sig = Signature::new_single("f32_round", Linkage::Public, &[Type::F32], Type::F32);
    let func_ref = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);
    let x = fb.args()[0];
    let result = fb.insert_inst(arith::Fround::new(is, x), Type::F32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, result));
    fb.seal_all();
    fb.finish();

    mb.build()
}

/// Oracle test (VERIFY item 1): native/cranelift executes `Ffloor`/`Fceil`/
/// `Ftrunc`/`Fround` correctly, bit-exact vs Rust's own `f32::floor`/`ceil`/
/// `trunc`/`round_ties_even` (NOT `f32::round`, which is ties-away-from-zero
/// -- the wrong oracle for `Fround`'s pinned `roundTiesToEven` semantics).
/// Covers ties, negatives, -0.0, +-inf, and NaN-passthrough.
#[test]
fn cranelift_f32_rounding_oracle() {
    let module = build_f32_rounding_module();
    let backend = CraneliftBackend::new();
    let artifact = backend.compile_module(&module).expect("cranelift compile");

    let floor: fn(f32) -> f32 = unsafe {
        std::mem::transmute(
            artifact
                .get_func_ptr::<fn(f32) -> f32>("f32_floor")
                .unwrap(),
        )
    };
    let ceil: fn(f32) -> f32 = unsafe {
        std::mem::transmute(artifact.get_func_ptr::<fn(f32) -> f32>("f32_ceil").unwrap())
    };
    let trunc: fn(f32) -> f32 = unsafe {
        std::mem::transmute(
            artifact
                .get_func_ptr::<fn(f32) -> f32>("f32_trunc")
                .unwrap(),
        )
    };
    let round: fn(f32) -> f32 = unsafe {
        std::mem::transmute(
            artifact
                .get_func_ptr::<fn(f32) -> f32>("f32_round")
                .unwrap(),
        )
    };

    // Ties, negatives, -0.0, ordinary values: bit-exact vs the Rust oracle.
    let cases: &[f32] = &[
        0.0, -0.0, 1.0, -1.0, 0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, -3.5, 0.25, 0.75, -0.25,
        -0.75, 3.14159, -3.14159, 42.0, -42.0, 1e10, -1e10, f32::MIN_POSITIVE, -f32::MIN_POSITIVE,
    ];
    for &x in cases {
        assert_eq!(floor(x).to_bits(), x.floor().to_bits(), "floor({x})");
        assert_eq!(ceil(x).to_bits(), x.ceil().to_bits(), "ceil({x})");
        assert_eq!(trunc(x).to_bits(), x.trunc().to_bits(), "trunc({x})");
        assert_eq!(
            round(x).to_bits(),
            x.round_ties_even().to_bits(),
            "round({x}) [ties-to-even]"
        );
    }

    // +-inf: already integral, all four ops are identity.
    for &x in &[f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(floor(x).to_bits(), x.to_bits(), "floor({x})");
        assert_eq!(ceil(x).to_bits(), x.to_bits(), "ceil({x})");
        assert_eq!(trunc(x).to_bits(), x.to_bits(), "trunc({x})");
        assert_eq!(round(x).to_bits(), x.to_bits(), "round({x})");
    }

    // NaN passthrough: result must be NaN. Payload/sign are not pinned (some
    // backends canonicalize), so this checks `.is_nan()`, not bit-exactness.
    for &x in &[f32::NAN, f32::from_bits(0x7fc0_1234), -f32::NAN] {
        assert!(floor(x).is_nan(), "floor({x:?}) must be NaN");
        assert!(ceil(x).is_nan(), "ceil({x:?}) must be NaN");
        assert!(trunc(x).is_nan(), "trunc({x:?}) must be NaN");
        assert!(round(x).is_nan(), "round({x:?}) must be NaN");
    }

    // The exact roundTiesToEven answers, spelled out (not just delegated to
    // the Rust oracle above): this is THE semantic check for this family.
    assert_eq!(
        round(0.5).to_bits(),
        0.0f32.to_bits(),
        "round(0.5) == 0 (ties to even)"
    );
    assert_eq!(
        round(1.5).to_bits(),
        2.0f32.to_bits(),
        "round(1.5) == 2 (ties to even)"
    );
    assert_eq!(
        round(2.5).to_bits(),
        2.0f32.to_bits(),
        "round(2.5) == 2 (ties to even)"
    );
    assert_eq!(
        round(-0.5).to_bits(),
        (-0.0f32).to_bits(),
        "round(-0.5) == -0 (ties to even)"
    );
}

/// Cross-backend differential (wasm vs cranelift) for the rounding family,
/// mirroring `cross_backend_f32_min_max_nan_zero_inf_differential`. Unlike
/// `Fmin`/`Fmax`, there is no NaN/-0.0 divergence to pin around here: floor/
/// ceil/trunc are monotone bit-exact IEEE ops on every backend, and `round`
/// is ties-to-even on both wasm's `f32.nearest` and cranelift's `nearest` by
/// their own spec/doc comment (see `arith::Fround`'s doc comment for the
/// naga/SPIR-V side, not exercised here for the same ABI reason as the
/// `Fmin`/`Fmax` differential above).
#[test]
fn cross_backend_f32_rounding_differential() {
    use sonatina_codegen::isa::wasm::WasmBackend;

    let module = build_f32_rounding_module();

    let cranelift_backend = CraneliftBackend::new();
    let cranelift_artifact = cranelift_backend
        .compile_module(&module)
        .expect("cranelift compile");
    let cranelift_fn = |name: &str| -> fn(f32) -> f32 {
        unsafe { std::mem::transmute(cranelift_artifact.get_func_ptr::<fn(f32) -> f32>(name).unwrap()) }
    };
    let cl_floor = cranelift_fn("f32_floor");
    let cl_ceil = cranelift_fn("f32_ceil");
    let cl_trunc = cranelift_fn("f32_trunc");
    let cl_round = cranelift_fn("f32_round");

    let wasm_backend = WasmBackend::new();
    let wasm_artifact = wasm_backend.compile_module(&module).expect("wasm compile");
    wasmparser::validate(&wasm_artifact.bytes).expect("valid wasm");
    let engine = wasmtime::Engine::default();
    let wasm_module = wasmtime::Module::new(&engine, &wasm_artifact.bytes).expect("load wasm");
    let mut store = wasmtime::Store::new(&engine, ());
    let instance =
        wasmtime::Instance::new(&mut store, &wasm_module, &[]).expect("instantiate wasm");
    let wasm_floor = instance
        .get_typed_func::<f32, f32>(&mut store, "f32_floor")
        .expect("f32_floor export");
    let wasm_ceil = instance
        .get_typed_func::<f32, f32>(&mut store, "f32_ceil")
        .expect("f32_ceil export");
    let wasm_trunc = instance
        .get_typed_func::<f32, f32>(&mut store, "f32_trunc")
        .expect("f32_trunc export");
    let wasm_round = instance
        .get_typed_func::<f32, f32>(&mut store, "f32_round")
        .expect("f32_round export");

    let edge_inputs: &[f32] = &[
        0.0,
        -0.0,
        0.5,
        1.5,
        2.5,
        3.5,
        -0.5,
        -1.5,
        -2.5,
        -3.5,
        1.0,
        -1.0,
        42.75,
        -42.75,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::from_bits(0x7fc0_1234),
    ];

    for &x in edge_inputs {
        let oracle_floor = x.floor();
        let oracle_ceil = x.ceil();
        let oracle_trunc = x.trunc();
        let oracle_round = x.round_ties_even();

        let cl = (cl_floor(x), cl_ceil(x), cl_trunc(x), cl_round(x));
        let wa = (
            wasm_floor.call(&mut store, x).expect("wasm floor call"),
            wasm_ceil.call(&mut store, x).expect("wasm ceil call"),
            wasm_trunc.call(&mut store, x).expect("wasm trunc call"),
            wasm_round.call(&mut store, x).expect("wasm round call"),
        );

        if x.is_nan() {
            assert!(cl.0.is_nan() && cl.1.is_nan() && cl.2.is_nan() && cl.3.is_nan(),
                "cranelift rounding family should be NaN for {x:?}, got {cl:?}");
            assert!(wa.0.is_nan() && wa.1.is_nan() && wa.2.is_nan() && wa.3.is_nan(),
                "wasm rounding family should be NaN for {x:?}, got {wa:?}");
            continue;
        }

        assert_eq!(cl.0.to_bits(), oracle_floor.to_bits(), "cranelift floor({x}) diverges from oracle");
        assert_eq!(cl.1.to_bits(), oracle_ceil.to_bits(), "cranelift ceil({x}) diverges from oracle");
        assert_eq!(cl.2.to_bits(), oracle_trunc.to_bits(), "cranelift trunc({x}) diverges from oracle");
        assert_eq!(cl.3.to_bits(), oracle_round.to_bits(), "cranelift round({x}) diverges from oracle [ties-to-even]");

        assert_eq!(wa.0.to_bits(), oracle_floor.to_bits(), "wasm floor({x}) diverges from oracle");
        assert_eq!(wa.1.to_bits(), oracle_ceil.to_bits(), "wasm ceil({x}) diverges from oracle");
        assert_eq!(wa.2.to_bits(), oracle_trunc.to_bits(), "wasm trunc({x}) diverges from oracle");
        assert_eq!(wa.3.to_bits(), oracle_round.to_bits(), "wasm round({x}) diverges from oracle [ties-to-even]");

        assert_eq!(cl.0.to_bits(), wa.0.to_bits(), "cranelift and wasm floor({x}) disagree bit-for-bit");
        assert_eq!(cl.1.to_bits(), wa.1.to_bits(), "cranelift and wasm ceil({x}) disagree bit-for-bit");
        assert_eq!(cl.2.to_bits(), wa.2.to_bits(), "cranelift and wasm trunc({x}) disagree bit-for-bit");
        assert_eq!(cl.3.to_bits(), wa.3.to_bits(), "cranelift and wasm round({x}) disagree bit-for-bit [ties-to-even]");
    }
}

// ===========================================================================
// Rung 3 STEP 2 (native leg): `MemAllocDynamic`/`Mload`/`Mstore` -- function-
// local `[u32; N]` arrays -- lowered to real cranelift stack slots + native
// loads/stores. Before this rung, `MemAllocDynamic` was unhandled
// ("unsupported instruction for CraneliftBackend: Opaque"); `Mload`/`Mstore`
// and `Unreachable` (traps) were ALREADY lowered (pre-existing, used by the
// object model), so those two need no test coverage here beyond what already
// exists elsewhere in this file -- these tests specifically exercise the
// NEW `MemAllocDynamic` arm and its pre-scan guards.
// ===========================================================================

/// The S2-A-shaped probe, executed (not just compiled): allocate an 8-word
/// array, store into two elements, bounds-check a dynamic index (the exact
/// Lt+Br+Unreachable shape wasm_lower.rs emits for every Fe array access),
/// and load back through the ok arm. Confirms MemAllocDynamic + Mstore +
/// Mload + the trap arm all execute correctly together on native, and that
/// untouched elements read back as zero (the explicit `emit_small_memset`
/// zero-init).
#[test]
fn cranelift_mem_alloc_dynamic_array_executes() {
    let isa = native_isa();
    let is = isa.inst_set();
    let mb = {
        let ctx = ModuleCtx::new(&isa);
        ModuleBuilder::new(ctx)
    };

    let sig = Signature::new_single("probe", Linkage::Public, &[Type::I32], Type::I32);
    let func_ref = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(func_ref);

    let entry = fb.append_block();
    let ok = fb.append_block();
    let trap = fb.append_block();

    fb.switch_to_block(entry);
    let k = fb.args()[0];
    let alloc_size = fb.make_imm_value(32i32); // 8 elements * 4 bytes
    let base = fb.insert_inst(data::MemAllocDynamic::new(is, alloc_size), Type::I32);
    let twelve = fb.make_imm_value(12i32);
    let addr3 = fb.insert_inst(arith::Add::new(is, base, twelve), Type::I32);
    let val3 = fb.make_imm_value(0xABCDi32);
    fb.insert_inst_no_result(data::Mstore::new(is, addr3, val3, Type::I32));
    let twenty = fb.make_imm_value(20i32);
    let addr5 = fb.insert_inst(arith::Add::new(is, base, twenty), Type::I32);
    let val5 = fb.make_imm_value(0x1234i32);
    fb.insert_inst_no_result(data::Mstore::new(is, addr5, val5, Type::I32));

    let eight = fb.make_imm_value(8i32);
    let in_bounds = fb.insert_inst(cmp::Lt::new(is, k, eight), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, in_bounds, ok, trap));

    fb.switch_to_block(ok);
    let four = fb.make_imm_value(4i32);
    let off = fb.insert_inst(arith::Mul::new(is, k, four), Type::I32);
    let addr_k = fb.insert_inst(arith::Add::new(is, base, off), Type::I32);
    let loaded = fb.insert_inst(data::Mload::new(is, addr_k, Type::I32), Type::I32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, loaded));

    fb.switch_to_block(trap);
    fb.insert_inst_no_result(control_flow::Unreachable::new(is));

    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let backend = CraneliftBackend::new();
    let artifact = backend
        .compile_module(&module)
        .expect("MemAllocDynamic array kernel should compile natively");

    let probe: fn(i32) -> i32 = unsafe {
        let ptr = artifact.get_func_ptr::<fn(i32) -> i32>("probe").unwrap();
        std::mem::transmute(ptr)
    };

    assert_eq!(probe(3), 0xABCD, "a[3] should read back what was stored");
    assert_eq!(probe(5), 0x1234, "a[5] should read back what was stored");
    assert_eq!(probe(0), 0, "an untouched element should read back zero (explicit zero-init)");
    assert_eq!(probe(7), 0, "an untouched element should read back zero (explicit zero-init)");
}

/// Two independent `MemAllocDynamic` calls in the same function must never
/// alias: each gets its OWN stack slot (unlike SPIR-V's shared emulated
/// heap, native has no capacity to exhaust or bump pointer to collide).
#[test]
fn cranelift_mem_alloc_dynamic_two_arrays_do_not_alias() {
    let isa = native_isa();
    let is = isa.inst_set();
    let mb = {
        let ctx = ModuleCtx::new(&isa);
        ModuleBuilder::new(ctx)
    };

    let sig = Signature::new_single("two_arrays", Linkage::Public, &[], Type::I32);
    let func_ref = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);

    let size = fb.make_imm_value(16i32);
    let a = fb.insert_inst(data::MemAllocDynamic::new(is, size), Type::I32);
    let b = fb.insert_inst(data::MemAllocDynamic::new(is, size), Type::I32);
    let va = fb.make_imm_value(111i32);
    fb.insert_inst_no_result(data::Mstore::new(is, a, va, Type::I32));
    let vb = fb.make_imm_value(222i32);
    fb.insert_inst_no_result(data::Mstore::new(is, b, vb, Type::I32));
    let la = fb.insert_inst(data::Mload::new(is, a, Type::I32), Type::I32);
    let lb = fb.insert_inst(data::Mload::new(is, b, Type::I32), Type::I32);
    let sum = fb.insert_inst(arith::Add::new(is, la, lb), Type::I32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, sum));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let backend = CraneliftBackend::new();
    let artifact = backend.compile_module(&module).expect("two-array kernel should compile");

    let f: fn() -> i32 = unsafe {
        let ptr = artifact.get_func_ptr::<fn() -> i32>("two_arrays").unwrap();
        std::mem::transmute(ptr)
    };
    assert_eq!(f(), 333, "storing into b must not clobber a (or vice versa)");
}

/// Regression: `lower_alloc_object` (wasm_lower.rs) over-allocates by up to
/// ALIGN-1 bytes so the RETURNED POINTER can be rounded up after the fact
/// (e.g. a logically-N*8-byte array can arrive at `MemAllocDynamic` as
/// N*8+7 bytes) -- real Fe-emitted sizes are NOT always clean multiples of
/// 8. `emit_small_memset`'s own internal invariant panicked
/// ("size is smaller than dest's alignment value") when this arm claimed a
/// flat 8-byte `buffer_align` for the zero-init regardless of the actual
/// size; found live against `poseidon_merkle_root_loop.fe`. Exercises a
/// deliberately odd (13-byte) allocation to pin the fix.
#[test]
fn cranelift_mem_alloc_dynamic_odd_size_zero_inits_and_executes() {
    let isa = native_isa();
    let is = isa.inst_set();
    let mb = {
        let ctx = ModuleCtx::new(&isa);
        ModuleBuilder::new(ctx)
    };

    let sig = Signature::new_single("odd_size", Linkage::Public, &[], Type::I32);
    let func_ref = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);

    // 13 bytes: not a multiple of 8, 4, or even 2 -- greatest_divisible_
    // power_of_two(13) == 1, the tightest possible case.
    let size = fb.make_imm_value(13i32);
    let base = fb.insert_inst(data::MemAllocDynamic::new(is, size), Type::I32);
    let nine = fb.make_imm_value(9i32);
    let addr9 = fb.insert_inst(arith::Add::new(is, base, nine), Type::I32);
    let val = fb.make_imm_value(7i32);
    fb.insert_inst_no_result(data::Mstore::new(is, addr9, val, Type::I32));
    // Reading back an UNTOUCHED byte-addressed i32 window (base+0) must be
    // zero (explicit zero-init), not garbage.
    let loaded = fb.insert_inst(data::Mload::new(is, base, Type::I32), Type::I32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, loaded));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let backend = CraneliftBackend::new();
    let artifact = backend
        .compile_module(&module)
        .expect("an odd-sized (13-byte) MemAllocDynamic should compile and zero-init cleanly");

    let f: fn() -> i32 = unsafe {
        let ptr = artifact.get_func_ptr::<fn() -> i32>("odd_size").unwrap();
        std::mem::transmute(ptr)
    };
    assert_eq!(f(), 0, "an untouched window of an odd-sized allocation must read back zero");
}

/// Codex bug 1's analog (heap-exhaustion aliasing), the compile-time half:
/// a `MemAllocDynamic` whose size is not a compile-time constant is
/// unsupported (cranelift stack slots are sized at IR-construction time),
/// and must fail closed rather than silently guessing a size.
///
/// `CraneliftBackend::compile_module` does NOT propagate a per-function
/// translation error as a module-level `Err` (`translate_module`'s
/// established, pre-existing convention: skip that one definition, log it,
/// keep compiling everything else -- exactly what
/// `native.rs::compile_and_verify_definitions` on the fe-codegen side
/// exists to turn into a hard failure at the wrapper level). So "fails
/// closed" here is verified the SAME way that wrapper verifies it: the
/// skipped function is declared but never defined, and
/// `get_finalized_function` panics for a declared-but-skipped definition
/// (an upstream cranelift-jit assertion, not a Sonatina-side panic) --
/// caught via `catch_unwind`, matching the established pattern.
#[test]
fn cranelift_mem_alloc_non_constant_size_fails_closed() {
    let isa = native_isa();
    let is = isa.inst_set();
    let mb = {
        let ctx = ModuleCtx::new(&isa);
        ModuleBuilder::new(ctx)
    };
    let sig = Signature::new_single("dyn_size", Linkage::Public, &[Type::I32], Type::I32);
    let func_ref = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);
    let n = fb.args()[0];
    let base = fb.insert_inst(data::MemAllocDynamic::new(is, n), Type::I32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, base));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let backend = CraneliftBackend::new();
    let artifact = backend
        .compile_module(&module)
        .expect("compile_module itself succeeds; the ONE bad definition is skipped, not hard-failed");
    let defined = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        artifact.get_func_ptr::<fn(i32) -> i32>("dyn_size")
    }))
    .ok()
    .flatten();
    assert!(
        defined.is_none(),
        "a non-constant MemAllocDynamic size must fail closed: `dyn_size` must be \
         declared-but-undefined (skipped), never a callable pointer"
    );
    eprintln!(
        "non-constant MemAllocDynamic size correctly fails closed: the function was skipped \
         (declared, never defined), the same postcondition native.rs::compile_and_verify_\
         definitions checks for"
    );
}

/// Cross-backend consistency guard (not a native memory-safety concern: a
/// loop-carried allocation on native just reuses the same, correctly-sized
/// stack slot every iteration, which is memory-safe). Ported from the
/// SPIR-V pre-scan: fail closed rather than silently diverge from wasm's
/// growing-arena semantics for a hypothetical future loop-carried-array
/// kernel.
#[test]
fn cranelift_mem_alloc_inside_loop_fails_closed() {
    let isa = native_isa();
    let is = isa.inst_set();
    let mb = {
        let ctx = ModuleCtx::new(&isa);
        ModuleBuilder::new(ctx)
    };
    let sig = Signature::new_single("loop_alloc", Linkage::Public, &[], Type::I32);
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
    let backend = CraneliftBackend::new();
    let artifact = backend
        .compile_module(&module)
        .expect("compile_module itself succeeds; the ONE bad definition is skipped, not hard-failed");
    let defined = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        artifact.get_func_ptr::<fn() -> i32>("loop_alloc")
    }))
    .ok()
    .flatten();
    assert!(
        defined.is_none(),
        "MemAllocDynamic inside a loop must fail closed: `loop_alloc` must be \
         declared-but-undefined (skipped), never a callable pointer"
    );
    eprintln!(
        "loop-carried allocation correctly fails closed on native: the function was skipped \
         (declared, never defined)"
    );
}

/// The native analog of the adversarial review's Finding A: a function that
/// traps with NO Mem ops at all (`fn(k): Br(Lt(k,8), ok, trap); ok: Return
/// 42; trap: Unreachable`). Unlike SPIR-V (which had to simulate poison via
/// an OR-accumulator flag because a GPU shader cannot actually trap), native
/// code has a REAL trap instruction available, and `Unreachable` already
/// lowered to one (`builder.ins().trap(...)`, pre-existing, untouched by
/// this rung) -- so there is no "silent zero" failure mode possible here by
/// construction: the function either returns the correct in-bounds value or
/// the process traps, never a wrong answer.
///
/// This test proves the function compiles (the module-level concern
/// `Unreachable` used to raise before it was lowered at all -- though on
/// this fork it has been supported since before this rung) and that the
/// non-trapping path computes the right answer. It deliberately does NOT
/// invoke the trapping path: a native trap is a real SIGILL/UD2, which
/// would abort this test binary, not something `catch_unwind` can recover
/// from. The trap arm's presence as a genuine `trap` instruction (not a
/// silently-dropped no-op) is structural, inherited from the pre-existing
/// `Unreachable` arm this rung does not touch.
#[test]
fn cranelift_no_mem_trap_compiles_and_ok_path_executes_correctly() {
    let isa = native_isa();
    let is = isa.inst_set();
    let mb = {
        let ctx = ModuleCtx::new(&isa);
        ModuleBuilder::new(ctx)
    };
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
    let backend = CraneliftBackend::new();
    let artifact = backend
        .compile_module(&module)
        .expect("a no-Mem-ops trapping function must still compile on native");

    let f: fn(i32) -> i32 = unsafe {
        let ptr = artifact.get_func_ptr::<fn(i32) -> i32>("no_mem_trap").unwrap();
        std::mem::transmute(ptr)
    };
    assert_eq!(f(0), 42, "the non-trapping (in-bounds) path must compute the correct value");
    assert_eq!(f(7), 42, "the non-trapping (in-bounds) path must compute the correct value");
    eprintln!(
        "no-Mem trap: function compiles, ok path executes correctly; trap arm not invoked \
         (a real native trap would abort the test process) but structurally present as a \
         genuine `trap` instruction via the pre-existing Unreachable arm."
    );
}
