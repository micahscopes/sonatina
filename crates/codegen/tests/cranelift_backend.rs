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
// naga/SPIR-V's `MathFunction::Min`/`Max` (GLSL.std.450 `FMin`/`FMax`) are NOT
// included in the bit-for-bit differential below, and are not exercised by
// this synthetic module at all (`SpirvBackend::compile_module`'s kernel ABI
// only accepts an i32/i64 return, which these raw-`f32`-returning functions
// don't fit). Even where naga/SPIR-V IS reachable (the real Fe demo pipeline,
// which returns i32 RGBA), the GLSL.std.450 spec leaves NaN/-0.0 behavior for
// FMin/FMax implementation-defined (unlike wasm/cranelift's pinned rules).
// This is a deliberate, documented divergence (OPEN DECISION — see
// NUMERIC_INTRINSICS_STAGING.md), not an oversight: no GPU adapter exists in
// this sandbox to observe real driver behavior either way.

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

    for x in [-3.5f32, 3.5, 0.0, -0.0, -1.0, 1.0, 42.25] {
        assert_eq!(abs(x), x.abs(), "abs({x})");
    }
    for (a, b) in [(1.0f32, 2.0), (2.0, 1.0), (-5.0, 5.0), (3.0, 3.0), (-2.5, -7.5)] {
        assert_eq!(min(a, b), wasm_rules_fmin_oracle(a, b), "min({a}, {b})");
        assert_eq!(max(a, b), wasm_rules_fmax_oracle(a, b), "max({a}, {b})");
    }
    for (x, lo, hi, expected) in [
        (5.0f32, 0.0, 1.0, 1.0),
        (-5.0, 0.0, 1.0, 0.0),
        (0.5, 0.0, 1.0, 0.5),
        (0.0, 0.0, 1.0, 0.0),
        (1.0, 0.0, 1.0, 1.0),
        (100.0, -10.0, 10.0, 10.0),
    ] {
        assert_eq!(clamp(x, lo, hi), expected, "clamp({x}, {lo}, {hi})");
    }
}

/// THE SHARP EDGE (VERIFY item 5): cross-backend differential over NaN/-0.0/
/// +-inf edge inputs for `Fmin`/`Fmax`. Wasm and cranelift are asserted to
/// agree bit-for-bit with each other AND with the from-scratch oracle above,
/// because both backends implement the same PINNED "WebAssembly rules". The
/// naga/SPIR-V path is validated (legal SPIR-V) but NOT executed (no GPU
/// adapter here) and NOT asserted to agree on NaN/-0.0 (GLSL.std.450 FMin/
/// FMax leaves that implementation-defined by spec — see the module doc
/// comment above and NUMERIC_INTRINSICS_STAGING.md's OPEN DECISION).
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

    // naga/SPIR-V is NOT exercised here: `SpirvBackend::compile_module`'s
    // "kernel" entry ABI only accepts an i32 (u32 word) or i64 return value
    // (this synthetic module's raw `f32`-returning functions don't fit that
    // envelope; there is no raw bitcast op wired to the SPIR-V backend to
    // route around it). The real naga/SPIR-V validation evidence for
    // `Fmin`/`Fmax`/`Fabs`/`Fclamp` comes from the actual Fe demo pipeline
    // (`demos/sketches/cga3d`, `demos/sketches/desargues` via
    // `fe-codegen`'s `demo_compile_gate.rs`), which returns i32 RGBA and so
    // fits the kernel/render ABI, and is validated there with
    // `naga::valid::Validator` — see NUMERIC_INTRINSICS_STAGING.md.

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
}
