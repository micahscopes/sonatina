use sonatina_codegen::Backend;
use sonatina_codegen::isa::wasm::WasmBackend;
use sonatina_ir::{
    Linkage, Signature, Type,
    builder::ModuleBuilder,
    func_cursor::InstInserter,
    inst::{arith, cmp, control_flow, data},
    isa::{Isa, native::Native, wasm32::Wasm32},
    module::ModuleCtx,
};
use sonatina_triple::{Architecture, OperatingSystem, TargetTriple, Vendor};

fn wasm32_triple() -> TargetTriple {
    TargetTriple::new(
        Architecture::Wasm32,
        Vendor::Unknown,
        OperatingSystem::Native,
    )
}

fn wasm32_module_builder() -> ModuleBuilder {
    let isa = Wasm32::new(wasm32_triple());
    let ctx = ModuleCtx::new(&isa);
    ModuleBuilder::new(ctx)
}

#[test]
fn canonical_arena_is_opt_in_checked_growable_and_resettable() {
    let engine = wasmtime::Engine::default();
    let ordinary = WasmBackend::new()
        .compile_module(&wasm32_module_builder().build())
        .unwrap();
    let module = wasmtime::Module::new(&engine, &ordinary.bytes).unwrap();
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = wasmtime::Instance::new(&mut store, &module, &[]).unwrap();
    assert!(instance.get_memory(&mut store, "memory").is_some());
    assert!(instance.get_func(&mut store, "fe_cabi_alloc").is_none());
    assert!(instance.get_func(&mut store, "fe_cabi_reset").is_none());

    let artifact = WasmBackend::new()
        .with_canonical_arena()
        .compile_module(&wasm32_module_builder().build())
        .unwrap();
    wasmparser::validate(&artifact.bytes).unwrap();
    let module = wasmtime::Module::new(&engine, &artifact.bytes).unwrap();
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = wasmtime::Instance::new(&mut store, &module, &[]).unwrap();
    let memory = instance.get_memory(&mut store, "memory").unwrap();
    let alloc = instance
        .get_typed_func::<(i32, i32), i32>(&mut store, "fe_cabi_alloc")
        .unwrap();
    let reset = instance
        .get_typed_func::<(), ()>(&mut store, "fe_cabi_reset")
        .unwrap();

    let first = alloc.call(&mut store, (3, 1)).unwrap();
    let aligned = alloc.call(&mut store, (8, 8)).unwrap();
    let next = alloc.call(&mut store, (7, 4)).unwrap();
    assert_eq!(first, 1024);
    assert_eq!(aligned % 8, 0);
    assert!(aligned >= first + 3);
    assert_eq!(next % 4, 0);
    assert!(next >= aligned + 8);

    let pages_before = memory.size(&store);
    assert_eq!(alloc.call(&mut store, (200_000, 16)).unwrap() % 16, 0);
    assert!(memory.size(&store) > pages_before);
    reset.call(&mut store, ()).unwrap();
    assert_eq!(alloc.call(&mut store, (1, 1)).unwrap(), 1024);

    for invalid in [(1, 0), (1, 3), (-1, 8)] {
        assert!(alloc.call(&mut store, invalid).is_err());
        reset.call(&mut store, ()).unwrap();
    }
    assert!(alloc.call(&mut store, (20_000_000, 1)).is_err());
}

#[test]
fn wasm_scalar_memory_ops_and_memzero_are_byte_exact() {
    let isa = Wasm32::new(wasm32_triple());
    let is = isa.inst_set();
    let mb = wasm32_module_builder();

    for (name, ty) in [
        ("mem_i1", Type::I1),
        ("mem_i8", Type::I8),
        ("mem_i16", Type::I16),
        ("mem_i32", Type::I32),
        ("mem_i64", Type::I64),
        ("mem_f32", Type::F32),
    ] {
        let func = mb
            .declare_function(Signature::new_single(
                name,
                Linkage::Public,
                &[Type::I32, ty],
                ty,
            ))
            .unwrap();
        let mut fb = mb.func_builder::<InstInserter>(func);
        let entry = fb.append_block();
        fb.switch_to_block(entry);
        let addr = fb.args()[0];
        let value = fb.args()[1];
        fb.insert_inst_no_result(data::Mstore::new(is, addr, value, ty));
        let loaded = fb.insert_inst(data::Mload::new(is, addr, ty), ty);
        fb.insert_inst_no_result(control_flow::Return::new_single(is, loaded));
        fb.seal_all();
        fb.finish();
    }

    let zero_func = mb
        .declare_function(Signature::new_unit(
            "memzero",
            Linkage::Public,
            &[Type::I32, Type::I32],
        ))
        .unwrap();
    let mut fb = mb.func_builder::<InstInserter>(zero_func);
    let entry = fb.append_block();
    fb.switch_to_block(entry);
    fb.insert_inst_no_result(data::Memzero::new(is, fb.args()[0], fb.args()[1]));
    fb.insert_inst_no_result(control_flow::Return::new(is, Default::default()));
    fb.seal_all();
    fb.finish();

    let artifact = WasmBackend::new().compile_module(&mb.build()).unwrap();
    wasmparser::validate(&artifact.bytes).unwrap();
    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, &artifact.bytes).unwrap();
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = wasmtime::Instance::new(&mut store, &module, &[]).unwrap();
    let memory = instance.get_memory(&mut store, "memory").unwrap();

    let i1 = instance.get_typed_func::<(i32, i32), i32>(&mut store, "mem_i1").unwrap();
    let i8 = instance.get_typed_func::<(i32, i32), i32>(&mut store, "mem_i8").unwrap();
    let i16 = instance.get_typed_func::<(i32, i32), i32>(&mut store, "mem_i16").unwrap();
    let i32f = instance.get_typed_func::<(i32, i32), i32>(&mut store, "mem_i32").unwrap();
    let i64f = instance.get_typed_func::<(i32, i64), i64>(&mut store, "mem_i64").unwrap();
    let f32f = instance.get_typed_func::<(i32, f32), f32>(&mut store, "mem_f32").unwrap();

    assert_eq!(i1.call(&mut store, (17, 1)).unwrap(), 1);
    assert_eq!(i8.call(&mut store, (18, 0xab)).unwrap(), 0xab);
    assert_eq!(i16.call(&mut store, (19, 0xcdef)).unwrap(), 0xcdef);
    assert_eq!(i32f.call(&mut store, (21, 0x78563412)).unwrap(), 0x78563412);
    assert_eq!(
        i64f.call(&mut store, (25, 0x0807060504030201)).unwrap(),
        0x0807060504030201
    );
    assert_eq!(f32f.call(&mut store, (33, -13.25)).unwrap(), -13.25);
    let bytes = memory.data(&store);
    assert_eq!(&bytes[17..21], &[1, 0xab, 0xef, 0xcd]);
    assert_eq!(&bytes[21..25], &0x78563412_i32.to_le_bytes());
    assert_eq!(&bytes[25..33], &0x0807060504030201_i64.to_le_bytes());
    assert_eq!(&bytes[33..37], &(-13.25_f32).to_le_bytes());

    memory.write(&mut store, 40, &[0xaa; 12]).unwrap();
    let memzero = instance.get_typed_func::<(i32, i32), ()>(&mut store, "memzero").unwrap();
    memzero.call(&mut store, (43, 5)).unwrap();
    assert_eq!(&memory.data(&store)[40..52], &[0xaa, 0xaa, 0xaa, 0, 0, 0, 0, 0, 0xaa, 0xaa, 0xaa, 0xaa]);
}

/// The minted `Wasm32` ISA drives the same WAFFLE backend end to end:
/// build `add(a,b)=a+b` under the Wasm32 target, compile, execute under
/// wasmtime. This is the ISA the Fe wasm lowering targets.
#[test]
fn wasm32_isa_add_wasmtime() {
    let isa = Wasm32::new(wasm32_triple());
    let is = isa.inst_set();
    let mb = wasm32_module_builder();

    let sig = Signature::new_single("add", Linkage::Public, &[Type::I64, Type::I64], Type::I64);
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
    let artifact = WasmBackend::new()
        .compile_module(&module)
        .expect("WASM compilation failed");
    wasmparser::validate(&artifact.bytes).expect("produced invalid WASM");

    let engine = wasmtime::Engine::default();
    let wm = wasmtime::Module::new(&engine, &artifact.bytes).unwrap();
    let mut store = wasmtime::Store::new(&engine, ());
    let inst = wasmtime::Instance::new(&mut store, &wm, &[]).unwrap();
    let f = inst
        .get_typed_func::<(i64, i64), i64>(&mut store, "add")
        .expect("add export");
    assert_eq!(f.call(&mut store, (2, 3)).unwrap(), 5);
}

/// A two-function program: `caller(a,b)` calls `callee(a,b)=a+b`. Exercises the
/// WAFFLE `Call` translation and the up-front `FuncRef -> Func` mapping.
#[test]
fn wasm32_isa_call_pair_wasmtime() {
    let isa = Wasm32::new(wasm32_triple());
    let is = isa.inst_set();
    let mb = wasm32_module_builder();

    let callee_sig =
        Signature::new_single("callee", Linkage::Private, &[Type::I64, Type::I64], Type::I64);
    let callee_ref = mb.declare_function(callee_sig).unwrap();

    let caller_sig =
        Signature::new_single("caller", Linkage::Public, &[Type::I64, Type::I64], Type::I64);
    let caller_ref = mb.declare_function(caller_sig).unwrap();

    // callee(a, b) = a + b
    {
        let mut fb = mb.func_builder::<InstInserter>(callee_ref);
        let entry = fb.append_block();
        fb.switch_to_block(entry);
        let a = fb.args()[0];
        let b = fb.args()[1];
        let sum = fb.insert_inst(arith::Add::new(is, a, b), Type::I64);
        fb.insert_inst_no_result(control_flow::Return::new_single(is, sum));
        fb.seal_all();
        fb.finish();
    }

    // caller(a, b) = callee(a, b)
    {
        let mut fb = mb.func_builder::<InstInserter>(caller_ref);
        let entry = fb.append_block();
        fb.switch_to_block(entry);
        let a = fb.args()[0];
        let b = fb.args()[1];
        let ret = fb.insert_inst(
            control_flow::Call::new(is, callee_ref, [a, b].into_iter().collect()),
            Type::I64,
        );
        fb.insert_inst_no_result(control_flow::Return::new_single(is, ret));
        fb.seal_all();
        fb.finish();
    }

    let module = mb.build();
    let artifact = WasmBackend::new()
        .compile_module(&module)
        .expect("WASM compilation failed");
    wasmparser::validate(&artifact.bytes).expect("produced invalid WASM");

    let engine = wasmtime::Engine::default();
    let wm = wasmtime::Module::new(&engine, &artifact.bytes).unwrap();
    let mut store = wasmtime::Store::new(&engine, ());
    let inst = wasmtime::Instance::new(&mut store, &wm, &[]).unwrap();
    let f = inst
        .get_typed_func::<(i64, i64), i64>(&mut store, "caller")
        .expect("caller export");
    assert_eq!(f.call(&mut store, (2, 3)).unwrap(), 5);
    assert_eq!(f.call(&mut store, (40, 2)).unwrap(), 42);
}

#[test]
fn wasm32_f32_ops_execute_in_wasmtime() {
    let source = r#"
target = "wasm32-unknown-native"

func public %float_chain(v0.f32, v1.f32) -> f32 {
    block0:
        v2.f32 = fadd v0 v1;
        v3.f32 = fsub v2 v1;
        v4.f32 = fmul v3 v1;
        v5.f32 = fdiv v4 v1;
        v6.f32 = fsqrt v5;
        v7.f32 = fneg v6;
        return v7;
}

func public %float_cmp(v0.f32, v1.f32) -> (i1, i1, i1) {
    block0:
        v2.i1 = feq v0 v1;
        v3.i1 = flt v0 v1;
        v4.i1 = fle v0 v1;
        return (v2, v3, v4);
}

func public %float_bits() -> f32 {
    block0:
        return 0x40490fdb.f32;
}
"#;
    let module = sonatina_parser::parse_module(source)
        .expect("float module should parse")
        .module;
    let artifact = WasmBackend::new()
        .compile_module(&module)
        .expect("WASM float compilation failed");
    wasmparser::validate(&artifact.bytes).expect("produced invalid float WASM");

    let engine = wasmtime::Engine::default();
    let wasm_module = wasmtime::Module::new(&engine, &artifact.bytes).unwrap();
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = wasmtime::Instance::new(&mut store, &wasm_module, &[]).unwrap();

    let chain = instance
        .get_typed_func::<(f32, f32), f32>(&mut store, "float_chain")
        .expect("float_chain export");
    assert_eq!(
        chain.call(&mut store, (9.0, 3.0)).unwrap().to_bits(),
        (-3.0_f32).to_bits()
    );

    let compare = instance
        .get_typed_func::<(f32, f32), (i32, i32, i32)>(&mut store, "float_cmp")
        .expect("float_cmp export");
    assert_eq!(compare.call(&mut store, (3.0, 9.0)).unwrap(), (0, 1, 1));
    assert_eq!(
        compare.call(&mut store, (f32::NAN, f32::NAN)).unwrap(),
        (0, 0, 0)
    );

    let bits = instance
        .get_typed_func::<(), f32>(&mut store, "float_bits")
        .expect("float_bits export");
    assert_eq!(bits.call(&mut store, ()).unwrap().to_bits(), 0x4049_0fdb);
}

fn native_triple() -> TargetTriple {
    let arch = if cfg!(target_arch = "x86_64") {
        Architecture::X86_64
    } else if cfg!(target_arch = "aarch64") {
        Architecture::Aarch64
    } else {
        panic!("unsupported host architecture");
    };
    TargetTriple::new(arch, Vendor::Unknown, OperatingSystem::Native)
}

fn native_module_builder() -> ModuleBuilder {
    let isa = Native::new(native_triple());
    let ctx = ModuleCtx::new(&isa);
    ModuleBuilder::new(ctx)
}

#[test]
fn wasm_add_two_i64s_wasmtime() {
    let isa = Native::new(native_triple());
    let is = isa.inst_set();
    let mb = native_module_builder();

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
    let backend = WasmBackend::new();
    let artifact = backend.compile_module(&module).expect("WASM compilation failed");

    // Validate WASM
    wasmparser::validate(&artifact.bytes).expect("produced invalid WASM");

    // Execute via wasmtime
    let engine = wasmtime::Engine::default();
    let wasm_module = wasmtime::Module::new(&engine, &artifact.bytes)
        .expect("wasmtime should load the WASM module");
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = wasmtime::Instance::new(&mut store, &wasm_module, &[])
        .expect("wasmtime should instantiate the module");

    let add_fn = instance
        .get_typed_func::<(i64, i64), i64>(&mut store, "add_i64")
        .expect("add_i64 export should exist");

    let result = add_fn.call(&mut store, (3, 4)).expect("call should succeed");
    assert_eq!(result, 7);

    let result = add_fn.call(&mut store, (100, 200)).expect("call should succeed");
    assert_eq!(result, 300);
}

#[test]
fn wasm_arithmetic_chain_wasmtime() {
    let isa = Native::new(native_triple());
    let is = isa.inst_set();
    let mb = native_module_builder();

    // f(a, b) = (a + b) * (a - b)
    let sig = Signature::new_single("arith", Linkage::Public, &[Type::I64, Type::I64], Type::I64);
    let func_ref = mb.declare_function(sig).unwrap();

    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);

    let a = fb.args()[0];
    let b = fb.args()[1];
    let sum = fb.insert_inst(arith::Add::new(is, a, b), Type::I64);
    let diff = fb.insert_inst(arith::Sub::new(is, a, b), Type::I64);
    let product = fb.insert_inst(arith::Mul::new(is, sum, diff), Type::I64);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, product));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let backend = WasmBackend::new();
    let artifact = backend.compile_module(&module).expect("WASM compilation failed");

    wasmparser::validate(&artifact.bytes).expect("produced invalid WASM");

    let engine = wasmtime::Engine::default();
    let wasm_module = wasmtime::Module::new(&engine, &artifact.bytes)
        .expect("wasmtime should load the module");
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = wasmtime::Instance::new(&mut store, &wasm_module, &[])
        .expect("wasmtime should instantiate");

    let f = instance
        .get_typed_func::<(i64, i64), i64>(&mut store, "arith")
        .expect("arith export");

    // (5+3) * (5-3) = 16
    assert_eq!(f.call(&mut store, (5, 3)).unwrap(), 16);
    // (10+7) * (10-7) = 51
    assert_eq!(f.call(&mut store, (10, 7)).unwrap(), 51);
}

#[test]
fn wasm_constant_return_wasmtime() {
    let isa = Native::new(native_triple());
    let is = isa.inst_set();
    let mb = native_module_builder();

    let sig = Signature::new_single("the_answer", Linkage::Public, &[], Type::I64);
    let func_ref = mb.declare_function(sig).unwrap();

    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);

    let val = fb.make_imm_value(42i64);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, val));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let backend = WasmBackend::new();
    let artifact = backend.compile_module(&module).expect("WASM compilation failed");

    wasmparser::validate(&artifact.bytes).expect("invalid WASM");

    let engine = wasmtime::Engine::default();
    let wasm_module = wasmtime::Module::new(&engine, &artifact.bytes).unwrap();
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = wasmtime::Instance::new(&mut store, &wasm_module, &[]).unwrap();

    let f = instance
        .get_typed_func::<(), i64>(&mut store, "the_answer")
        .expect("the_answer export");

    assert_eq!(f.call(&mut store, ()).unwrap(), 42);
}

#[test]
fn wasm_loop_sum_wasmtime() {
    let isa = Native::new(native_triple());
    let is = isa.inst_set();
    let mb = native_module_builder();

    // fn sum_to(n: i64) -> i64 { let mut acc=0, i=0; while i<n { acc+=i; i++; } return acc; }
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
    let backend = WasmBackend::new();
    let artifact = backend.compile_module(&module).expect("WASM compilation failed");

    wasmparser::validate(&artifact.bytes).expect("invalid WASM");

    let engine = wasmtime::Engine::default();
    let wasm_module = wasmtime::Module::new(&engine, &artifact.bytes).expect("load failed");
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = wasmtime::Instance::new(&mut store, &wasm_module, &[]).expect("instantiate");

    let f = instance.get_typed_func::<i64, i64>(&mut store, "sum_to").expect("sum_to export");

    // sum_to(5) = 0+1+2+3+4 = 10
    let r = f.call(&mut store, 5).unwrap();
    eprintln!("sum_to(5) = {r}");
    assert_eq!(r, 10);
    assert_eq!(f.call(&mut store, 10).unwrap(), 45);
    assert_eq!(f.call(&mut store, 0).unwrap(), 0);
}

/// Poseidon-style sigma loop on WASM with known-answer verification.
/// Same computation as cranelift_poseidon_loop_with_const_array but
/// uses inline constants (WASM doesn't have ConstRef yet).
#[test]
fn wasm_poseidon_sigma_loop_wasmtime() {
    let isa = Native::new(native_triple());
    let is = isa.inst_set();
    let mb = native_module_builder();

    // fn poseidon_sigma() -> i64 {
    //   let mut acc = 1;
    //   // Round constants inline: [3, 5, 7, 11]
    //   // Round 0: acc = sigma(acc + 3) where sigma(x) = x*x + x
    //   // ... repeat for each constant
    //   // Return acc after 4 rounds
    //   for i in 0..4: acc = (acc + C[i])^2 + (acc + C[i])
    //   Using a simpler version: acc += i*i + i per round (avoids needing const array)
    // }
    // Actually, let's do the exact same computation as the Cranelift test:
    // acc=1, C=[3,5,7,11], sigma(x)=x*x+x
    // Round 0: x=1+3=4, acc=4*4+4=20
    // Round 1: x=20+5=25, acc=25*25+25=650
    // Round 2: x=650+7=657, acc=657*657+657=432306
    // Round 3: x=432306+11=432317, acc=432317*432317+432317=186898420806
    //
    // We'll build this with a loop where the "constant" is computed as a function of i.
    // C[0]=3, C[1]=5, C[2]=7, C[3]=11
    // C[i] = 2*i + 3 for i=0,1,2 but C[3]=11 != 2*3+3=9
    // So we can't use a simple formula. Instead, use 4 unrolled iterations.

    let sig = Signature::new_single("poseidon_sigma", Linkage::Public, &[], Type::I64);
    let func_ref = mb.declare_function(sig).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);

    // Unrolled 4 rounds with inline constants
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
    let backend = WasmBackend::new();
    let artifact = backend.compile_module(&module).expect("WASM compilation failed");
    wasmparser::validate(&artifact.bytes).expect("invalid WASM");

    let engine = wasmtime::Engine::default();
    let wasm_module = wasmtime::Module::new(&engine, &artifact.bytes).expect("load");
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = wasmtime::Instance::new(&mut store, &wasm_module, &[]).expect("instantiate");
    let f = instance.get_typed_func::<(), i64>(&mut store, "poseidon_sigma").expect("export");

    let result = f.call(&mut store, ()).unwrap();
    assert_eq!(result, 186898420806, "WASM Poseidon sigma should match Cranelift known answer");
}

/// Cross-target loop known-answer: same loop IR → Cranelift + WASM, compare.
#[test]
fn cross_target_loop_cranelift_vs_wasm() {
    use sonatina_codegen::isa::cranelift::CraneliftBackend;

    let isa = Native::new(native_triple());
    let is = isa.inst_set();
    let mb = native_module_builder();

    // sum_to(n) with loop — same IR compiled to both backends
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

    // Cranelift
    let cl = CraneliftBackend::new();
    let cl_art = cl.compile_module(&module).expect("cranelift");
    let cl_fn: fn(i64) -> i64 = unsafe {
        std::mem::transmute(cl_art.get_func_ptr::<fn(i64) -> i64>("sum_to").unwrap())
    };

    // WASM
    let wasm = WasmBackend::new();
    let wasm_art = wasm.compile_module(&module).expect("wasm");
    wasmparser::validate(&wasm_art.bytes).expect("invalid");
    let engine = wasmtime::Engine::default();
    let wm = wasmtime::Module::new(&engine, &wasm_art.bytes).unwrap();
    let mut store = wasmtime::Store::new(&engine, ());
    let inst = wasmtime::Instance::new(&mut store, &wm, &[]).unwrap();
    let wasm_fn = inst.get_typed_func::<i64, i64>(&mut store, "sum_to").unwrap();

    for n in [0, 1, 5, 10, 100] {
        let cl_result = cl_fn(n);
        let wasm_result = wasm_fn.call(&mut store, n).unwrap();
        assert_eq!(cl_result, wasm_result, "Cranelift vs WASM for sum_to({n})");
        assert_eq!(cl_result, n * (n - 1) / 2, "sum_to({n}) formula check");
    }
}

// ---------------------------------------------------------------------------
// R3.1: WAFFLE import emission (external declarations -> wasm imports).
// ---------------------------------------------------------------------------

/// Build a Wasm32 module with an EXTERNAL declaration `host_add(i64, i64) ->
/// i64` (no body) and a defined `compute(a, b) = host_add(a, b)` that calls it.
/// Shared by the import tests below (each chooses how to compile it).
fn build_import_demo_module() -> sonatina_ir::Module {
    let isa = Wasm32::new(wasm32_triple());
    let is = isa.inst_set();
    let mb = wasm32_module_builder();

    // External host function: declared, no body -> a wasm import.
    let host_sig = Signature::new_single(
        "host_add",
        Linkage::External,
        &[Type::I64, Type::I64],
        Type::I64,
    );
    let host_ref = mb.declare_function(host_sig).unwrap();

    // Defined function calling the import: compute(a, b) = host_add(a, b).
    let compute_sig =
        Signature::new_single("compute", Linkage::Public, &[Type::I64, Type::I64], Type::I64);
    let compute_ref = mb.declare_function(compute_sig).unwrap();
    {
        let mut fb = mb.func_builder::<InstInserter>(compute_ref);
        let entry = fb.append_block();
        fb.switch_to_block(entry);
        let a = fb.args()[0];
        let b = fb.args()[1];
        let called = fb.insert_inst(
            control_flow::Call::new(is, host_ref, [a, b].into_iter().collect()),
            Type::I64,
        );
        fb.insert_inst_no_result(control_flow::Return::new_single(is, called));
        fb.seal_all();
        fb.finish();
    }

    mb.build()
}

/// Translate the import demo module to wasm bytes with the default (empty)
/// import-module table. Shared by the two R3.1 tests below.
fn build_import_demo_wasm() -> Vec<u8> {
    WasmBackend::new()
        .compile_module(&build_import_demo_module())
        .expect("WASM compilation failed")
        .bytes
}

/// R3.1 (a): the emitted import is genuinely wired and executable. wasmtime
/// satisfies the `("fe", "host_add")` import through a `Linker` binding and
/// the defined `compute` returns the host result. Because `host_add` has no
/// body, the only way `compute` can translate at all is via the emitted
/// import (the `Call` arm fails closed otherwise), so a passing run proves
/// the import path end to end.
#[test]
fn wasm32_isa_import_host_add_wasmtime() {
    let bytes = build_import_demo_wasm();
    wasmparser::validate(&bytes).expect("produced invalid WASM");

    let engine = wasmtime::Engine::default();
    let wm = wasmtime::Module::new(&engine, &bytes).unwrap();
    let mut store = wasmtime::Store::new(&engine, ());
    let mut linker = wasmtime::Linker::new(&engine);
    linker
        .func_wrap("fe", "host_add", |a: i64, b: i64| a + b)
        .unwrap();
    let inst = linker.instantiate(&mut store, &wm).unwrap();
    let f = inst
        .get_typed_func::<(i64, i64), i64>(&mut store, "compute")
        .expect("compute export");
    assert_eq!(f.call(&mut store, (2, 3)).unwrap(), 5);
    assert_eq!(f.call(&mut store, (40, 2)).unwrap(), 42);
}

/// R3.1 (b): scan the emitted bytes and assert the wasm import section holds
/// the `("fe", "host_add")` func import, and that imported function indices
/// precede defined ones (imports occupy the low slots of the index space).
/// Asserted from the bytes, not assumed from WAFFLE.
#[test]
fn wasm32_isa_import_precedes_defined_in_index_space() {
    use wasmparser::{ExternalKind, Payload, TypeRef};

    let bytes = build_import_demo_wasm();

    let mut func_imports: Vec<(String, String)> = Vec::new();
    let mut compute_index: Option<u32> = None;

    for payload in wasmparser::Parser::new(0).parse_all(&bytes) {
        match payload.expect("valid wasm payload") {
            Payload::ImportSection(reader) => {
                // wasmparser groups imports; each group iterates to (idx, Import).
                for group in reader {
                    let group = group.expect("valid import group");
                    for entry in group {
                        let (_idx, import) = entry.expect("valid import entry");
                        if let TypeRef::Func(_) = import.ty {
                            func_imports
                                .push((import.module.to_string(), import.name.to_string()));
                        }
                    }
                }
            }
            Payload::ExportSection(reader) => {
                for export in reader {
                    let export = export.expect("valid export entry");
                    if export.kind == ExternalKind::Func && export.name == "compute" {
                        compute_index = Some(export.index);
                    }
                }
            }
            _ => {}
        }
    }

    // The ("fe", "host_add") func import is present.
    assert!(
        func_imports.contains(&("fe".to_string(), "host_add".to_string())),
        "expected a (\"fe\", \"host_add\") func import, found {func_imports:?}"
    );
    let num_func_imports = func_imports.len() as u32;
    assert_eq!(num_func_imports, 1, "exactly one func import expected");

    // Index-space invariant: func imports occupy [0, num_func_imports), so
    // every DEFINED function index is >= num_func_imports. Check it against the
    // concrete defined function `compute` (index resolved via its export)
    // rather than trusting WAFFLE to have ordered the arena.
    let compute_index = compute_index.expect("compute must be an exported func");
    assert!(
        compute_index >= num_func_imports,
        "defined `compute` at index {compute_index} must not fall in the import \
         range [0, {num_func_imports})"
    );
}

/// Scan the emitted wasm and return the `(module, name)` of every func import.
fn scan_func_imports(bytes: &[u8]) -> Vec<(String, String)> {
    use wasmparser::{Payload, TypeRef};
    let mut imports = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        if let Payload::ImportSection(reader) = payload.expect("valid wasm payload") {
            for group in reader {
                for entry in group.expect("valid import group") {
                    let (_idx, import) = entry.expect("valid import entry");
                    if let TypeRef::Func(_) = import.ty {
                        imports.push((import.module.to_string(), import.name.to_string()));
                    }
                }
            }
        }
    }
    imports
}

/// R3.3: `WasmBackend::with_import_modules` names an external declaration's wasm
/// import MODULE from a symbol -> module side table, so the import lands as
/// `("fe:host", "host_add")` instead of the flat `("fe", "host_add")` default. A
/// symbol absent from the table keeps the `"fe"` fallback (no Sonatina IR change,
/// no symbol interning touched: the field name stays the symbol).
#[test]
fn wasm32_import_module_from_side_table() {
    // With the side table: the import module is the supplied name.
    let mut table = std::collections::HashMap::new();
    table.insert("host_add".to_string(), "fe:host".to_string());
    let bytes = WasmBackend::new()
        .with_import_modules(table)
        .compile_module(&build_import_demo_module())
        .expect("WASM compilation failed")
        .bytes;
    wasmparser::validate(&bytes).expect("produced invalid WASM");
    assert!(
        scan_func_imports(&bytes).contains(&("fe:host".to_string(), "host_add".to_string())),
        "expected a (\"fe:host\", \"host_add\") func import, found {:?}",
        scan_func_imports(&bytes)
    );

    // A table that does NOT list this symbol leaves the flat "fe" default.
    let mut other = std::collections::HashMap::new();
    other.insert("some_other_symbol".to_string(), "fe:webgpu".to_string());
    let bytes = WasmBackend::new()
        .with_import_modules(other)
        .compile_module(&build_import_demo_module())
        .expect("WASM compilation failed")
        .bytes;
    assert!(
        scan_func_imports(&bytes).contains(&("fe".to_string(), "host_add".to_string())),
        "an unlisted symbol must keep the \"fe\" default, found {:?}",
        scan_func_imports(&bytes)
    );
}

/// M2a fork push #2: i32-operand `Lt`/`Slt`/`Sar` are keyed on the operand type
/// (were hardwired to the i64 operators). An i32 compare must use `i32.lt_u` /
/// `i32.lt_s` and an i32 shift `i32.shr_s`, or wasmtime rejects the module at
/// validation. This test would have failed to even validate before the fix, and
/// signed vs unsigned disagree on a 0x80000000-class operand.
#[test]
fn wasm_i32_signed_ops_execute() {
    let isa = Native::new(native_triple());
    let is = isa.inst_set();
    let mb = native_module_builder();

    // fn lt_u(a: i32, b: i32) -> i32 { (a <u b) as i32 }
    {
        let sig = Signature::new_single("lt_u", Linkage::Public, &[Type::I32, Type::I32], Type::I32);
        let fr = mb.declare_function(sig).unwrap();
        let mut fb = mb.func_builder::<InstInserter>(fr);
        let e = fb.append_block();
        fb.switch_to_block(e);
        let a = fb.args()[0];
        let b = fb.args()[1];
        let c = fb.insert_inst(cmp::Lt::new(is, a, b), Type::I1);
        fb.insert_inst_no_result(control_flow::Return::new_single(is, c));
        fb.seal_all();
        fb.finish();
    }
    // fn lt_s(a: i32, b: i32) -> i32 { (a <s b) as i32 }
    {
        let sig = Signature::new_single("lt_s", Linkage::Public, &[Type::I32, Type::I32], Type::I32);
        let fr = mb.declare_function(sig).unwrap();
        let mut fb = mb.func_builder::<InstInserter>(fr);
        let e = fb.append_block();
        fb.switch_to_block(e);
        let a = fb.args()[0];
        let b = fb.args()[1];
        let c = fb.insert_inst(cmp::Slt::new(is, a, b), Type::I1);
        fb.insert_inst_no_result(control_flow::Return::new_single(is, c));
        fb.seal_all();
        fb.finish();
    }
    // fn sar(a: i32) -> i32 { a >> 12 }  (Sar constructor order is (bits, value))
    {
        let sig = Signature::new_single("sar", Linkage::Public, &[Type::I32], Type::I32);
        let fr = mb.declare_function(sig).unwrap();
        let mut fb = mb.func_builder::<InstInserter>(fr);
        let e = fb.append_block();
        fb.switch_to_block(e);
        let a = fb.args()[0];
        let twelve = fb.make_imm_value(12i32);
        let s = fb.insert_inst(arith::Sar::new(is, twelve, a), Type::I32);
        fb.insert_inst_no_result(control_flow::Return::new_single(is, s));
        fb.seal_all();
        fb.finish();
    }

    let module = mb.build();
    let artifact = WasmBackend::new().compile_module(&module).expect("wasm compile");
    wasmparser::validate(&artifact.bytes).expect("i32 signed ops must produce valid WASM");

    let engine = wasmtime::Engine::default();
    let wm = wasmtime::Module::new(&engine, &artifact.bytes).unwrap();
    let mut store = wasmtime::Store::new(&engine, ());
    let inst = wasmtime::Instance::new(&mut store, &wm, &[]).unwrap();
    let lt_u = inst.get_typed_func::<(i32, i32), i32>(&mut store, "lt_u").unwrap();
    let lt_s = inst.get_typed_func::<(i32, i32), i32>(&mut store, "lt_s").unwrap();
    let sar = inst.get_typed_func::<i32, i32>(&mut store, "sar").unwrap();

    // Signed and unsigned disagree on a 0x80000000-class operand.
    let big = i32::MIN; // 0x8000_0000
    assert_eq!(lt_u.call(&mut store, (big, 1)).unwrap(), 0, "unsigned: 0x80000000 <u 1 is false");
    assert_eq!(lt_s.call(&mut store, (big, 1)).unwrap(), 1, "signed: -2147483648 <s 1 is true");
    // Ordinary positive operands agree.
    assert_eq!(lt_u.call(&mut store, (3, 7)).unwrap(), 1);
    assert_eq!(lt_s.call(&mut store, (3, 7)).unwrap(), 1);

    // Arithmetic (not logical) shift: -4096 >> 12 == -1.
    assert_eq!(sar.call(&mut store, -4096).unwrap(), -1, "arithmetic shift keeps the sign");
    assert_eq!(sar.call(&mut store, 4096).unwrap(), 1);
    assert_eq!(sar.call(&mut store, -1).unwrap(), -1);

    eprintln!(
        "wasm_i32_signed_ops_execute OK: i32 Lt/Slt/Sar execute; signed vs unsigned disagree on 0x80000000; -4096>>12 == -1"
    );
}
