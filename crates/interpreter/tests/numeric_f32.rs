use sonatina_interpreter::Machine;
use sonatina_ir::{Immediate, interpret::EvalValue};

#[test]
fn interpreter_executes_saturating_f32_conversions() {
    let source = r#"
target = "wasm32-unknown-native"
func public %from_i32(v0.i32) -> (f32, f32) {
    block0:
        v1.f32 = i32_to_f32 v0;
        v2.f32 = u32_to_f32 v0;
        return (v1, v2);
}
func public %to_i32(v0.f32) -> (i32, i32) {
    block0:
        v1.i32 = f32_to_i32 v0;
        v2.i32 = f32_to_u32 v0;
        return (v1, v2);
}
"#;
    let parsed = sonatina_parser::parse_module(source).expect("module should parse");
    let find = |name| {
        parsed
            .module
            .ctx
            .declared_funcs
            .iter()
            .find_map(|entry| (entry.value().name() == name).then(|| *entry.key()))
            .expect("function should be declared")
    };
    let from_i32 = find("from_i32");
    let to_i32 = find("to_i32");
    let mut machine = Machine::new(parsed.module);
    assert_eq!(
        machine
            .run_results(from_i32, vec![EvalValue::Imm(Immediate::I32(-1))])
            .as_slice(),
        &[
            EvalValue::Imm(Immediate::F32(0xbf80_0000)),
            EvalValue::Imm(Immediate::F32(0x4f80_0000)),
        ]
    );
    for (bits, signed, unsigned) in [
        (0x7fc0_0000, 0, 0),
        (0x7f80_0000, i32::MAX, -1),
        (0xff80_0000, i32::MIN, 0),
    ] {
        assert_eq!(
            machine
                .run_results(to_i32, vec![EvalValue::Imm(Immediate::F32(bits))])
                .as_slice(),
            &[
                EvalValue::Imm(Immediate::I32(signed)),
                EvalValue::Imm(Immediate::I32(unsigned)),
            ]
        );
    }
}
