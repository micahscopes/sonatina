use sonatina_parser::parse_module;
use sonatina_verifier::{VerificationLevel, VerifierConfig, verify_module};

#[test]
fn arena_scope_requires_pointer_checkpoints() {
    let cfg = VerifierConfig::for_level(VerificationLevel::Standard);
    let valid = r#"
target = "wasm32-unknown-native"
func public %scoped(v0.i32) {
    block0:
        v1.*i8 = mem.checkpoint;
        v2.*i8 = mem.alloc_dynamic v0;
        mem.rewind v1;
        return;
}
"#;
    let parsed = parse_module(valid).expect("arena scope module should parse");
    assert!(verify_module(&parsed.module, &cfg).is_ok());

    let invalid = r#"
target = "wasm32-unknown-native"
func public %bad(v0.i32) {
    block0:
        v1.i32 = mem.checkpoint;
        mem.rewind v0;
        return;
}
"#;
    let parsed = parse_module(invalid).expect("invalid arena scope module should parse");
    let report = verify_module(&parsed.module, &cfg);
    let has_code = |code| report.diagnostics.iter().any(|d| d.code.as_str() == code);
    assert!(
        has_code("IR0600"),
        "expected checkpoint operand mismatch, got {report}"
    );
    assert!(
        has_code("IR0601"),
        "expected checkpoint result mismatch, got {report}"
    );
}
