use sonatina_parser::parse_module;
use sonatina_verifier::{VerificationLevel, VerifierConfig, verify_module};

#[test]
fn numeric_f32_conversions_verify_and_reject_wrong_types() {
    let cfg = VerifierConfig::for_level(VerificationLevel::Standard);
    let valid = r#"
target = "wasm32-unknown-native"
func public %conversions(v0.i32, v1.f32) -> (f32, f32, i32, i32) {
    block0:
        v2.f32 = i32_to_f32 v0;
        v3.f32 = u32_to_f32 v0;
        v4.i32 = f32_to_i32 v1;
        v5.i32 = f32_to_u32 v1;
        return (v2, v3, v4, v5);
}
"#;
    let parsed = parse_module(valid).expect("conversion module should parse");
    assert!(verify_module(&parsed.module, &cfg).is_ok());
    let invalid = r#"
target = "wasm32-unknown-native"
func public %bad(v0.f32) -> f32 {
    block0:
        v1.i32 = i32_to_f32 v0;
        return v1;
}
"#;
    let parsed = parse_module(invalid).expect("invalid typed module should parse");
    let report = verify_module(&parsed.module, &cfg);
    let has_code = |code| report.diagnostics.iter().any(|d| d.code.as_str() == code);
    assert!(
        has_code("IR0600"),
        "expected operand mismatch, got {report}"
    );
    assert!(
        has_code("IR0601"),
        "expected result mismatch, got {report}"
    );
}
