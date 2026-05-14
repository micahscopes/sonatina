use sonatina_codegen::{Backend, Compile, OptLevel};
use sonatina_codegen::isa::cranelift::CraneliftBackend;
use sonatina_ir::{
    Linkage, Signature, Type,
    builder::ModuleBuilder,
    func_cursor::InstInserter,
    inst::{arith, control_flow},
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
