use sonatina_codegen::{Backend, isa::wasm::WasmBackend};

#[test]
fn wasm32_guarded_unsigned_subtraction_preserves_values_and_overflow() {
    let source = r#"
target = "wasm32-unknown-native"
func public %guarded(v0.i32, v1.i32) -> i32 {
 block0:
  v2.i1 = lt v0 v1;
  br v2 block2 block1;
 block1:
  (v3.i32, v4.i1) = usubo v0 v1;
  br v4 block2 block3;
 block2:
  return 0.i32;
 block3:
  return v3;
}
func public %unguarded(v0.i32, v1.i32) -> i32 {
 block0:
  (v2.i32, v3.i1) = usubo v0 v1;
  br v3 block1 block2;
 block1:
  return 4294967295.i32;
 block2:
  return v2;
}
"#;
    let mut samples = vec![0u32, 1, 2, 16383, 16384, 16385,
        0x7fffffff, 0x80000000, u32::MAX - 1, u32::MAX];
    let mut seed = 37u32;
    for _ in 0..54 {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        samples.push(seed);
    }
    for optimize in [false, true] {
        let module = sonatina_parser::parse_module(source).unwrap().module;
        if optimize {
            sonatina_codegen::optim::pipeline::run_function_passes_on(
                &module, &module.funcs(),
                &[sonatina_codegen::optim::pipeline::Pass::CheckedArithElim]);
            for id in module.funcs() {
                module.func_store.view(id, |func| {
                    let text = sonatina_ir::ir_writer::FuncWriter::new(id, func).dump_string();
                    let guarded = module.ctx.func_sig(id, |sig| sig.name() == "guarded");
                    assert_eq!(text.contains("usubo"), !guarded, "{text}");
                });
            }
        }
        let config = sonatina_verifier::VerifierConfig::for_level(
            sonatina_verifier::VerificationLevel::Full);
        assert!(!sonatina_verifier::verify_module(&module, &config).has_errors());
        let artifact = WasmBackend::new().compile_module(&module).unwrap();
        wasmparser::validate(&artifact.bytes).unwrap();
        let engine = wasmtime::Engine::default();
        let compiled = wasmtime::Module::new(&engine, &artifact.bytes).unwrap();
        let mut store = wasmtime::Store::new(&engine, ());
        let instance = wasmtime::Instance::new(&mut store, &compiled, &[]).unwrap();
        let guarded = instance.get_typed_func::<(i32, i32), i32>(&mut store, "guarded").unwrap();
        let unguarded = instance.get_typed_func::<(i32, i32), i32>(&mut store, "unguarded").unwrap();
        for &a in &samples {
            for &b in &samples {
                assert_eq!(guarded.call(&mut store, (a as i32, b as i32)).unwrap() as u32,
                    a.saturating_sub(b), "optimized={optimize}, {a}-{b}");
                assert_eq!(unguarded.call(&mut store, (a as i32, b as i32)).unwrap() as u32,
                    a.checked_sub(b).unwrap_or(u32::MAX), "optimized={optimize}, {a}-{b}");
            }
        }
    }
}

#[test]
fn wasm32_f32_conversions_execute_with_saturation() {
    let source = r#"
target = "wasm32-unknown-native"
func public %s2f(v0.i32) -> f32 {
    block0:
        v1.f32 = i32_to_f32 v0;
        return v1;
}
func public %u2f(v0.i32) -> f32 {
    block0:
        v1.f32 = u32_to_f32 v0;
        return v1;
}
func public %f2s(v0.f32) -> i32 {
    block0:
        v1.i32 = f32_to_i32 v0;
        return v1;
}
func public %f2u(v0.f32) -> i32 {
    block0:
        v1.i32 = f32_to_u32 v0;
        return v1;
}
func public %bits2f(v0.i32) -> f32 {
    block0:
        v1.f32 = bitcast v0 f32;
        return v1;
}
func public %f2bits(v0.f32) -> i32 {
    block0:
        v1.i32 = bitcast v0 i32;
        return v1;
}
"#;
    let module = sonatina_parser::parse_module(source)
        .expect("module should parse")
        .module;
    let artifact = WasmBackend::new()
        .compile_module(&module)
        .expect("WASM compilation failed");
    wasmparser::validate(&artifact.bytes).expect("invalid WASM");
    let engine = wasmtime::Engine::default();
    let wasm_module = wasmtime::Module::new(&engine, &artifact.bytes).unwrap();
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = wasmtime::Instance::new(&mut store, &wasm_module, &[]).unwrap();
    let s2f = instance
        .get_typed_func::<i32, f32>(&mut store, "s2f")
        .unwrap();
    let u2f = instance
        .get_typed_func::<i32, f32>(&mut store, "u2f")
        .unwrap();
    let f2s = instance
        .get_typed_func::<f32, i32>(&mut store, "f2s")
        .unwrap();
    let f2u = instance
        .get_typed_func::<f32, i32>(&mut store, "f2u")
        .unwrap();
    let bits2f = instance
        .get_typed_func::<i32, f32>(&mut store, "bits2f")
        .unwrap();
    let f2bits = instance
        .get_typed_func::<f32, i32>(&mut store, "f2bits")
        .unwrap();
    assert_eq!(s2f.call(&mut store, -1).unwrap().to_bits(), 0xbf80_0000);
    assert_eq!(u2f.call(&mut store, -1).unwrap().to_bits(), 0x4f80_0000);
    for (value, signed, unsigned) in [
        (f32::NAN, 0, 0),
        (f32::INFINITY, i32::MAX, -1),
        (f32::NEG_INFINITY, i32::MIN, 0),
    ] {
        assert_eq!(f2s.call(&mut store, value).unwrap(), signed);
        assert_eq!(f2u.call(&mut store, value).unwrap(), unsigned);
    }
    for bits in [0u32, 1, 0x3f80_0000, 0x8000_0000, 0x7fc0_1234] {
        let value = bits2f.call(&mut store, bits as i32).unwrap();
        assert_eq!(value.to_bits(), bits);
        assert_eq!(f2bits.call(&mut store, value).unwrap() as u32, bits);
    }
}
