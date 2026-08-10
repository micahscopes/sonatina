use sonatina_codegen::{Backend, isa::wasm::WasmBackend};

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
