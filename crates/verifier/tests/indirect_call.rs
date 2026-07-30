use sonatina_parser::parse_module;
use sonatina_verifier::{VerificationLevel, VerifierConfig, verify_module};

fn verify(source: &str) -> sonatina_verifier::VerificationReport {
    let parsed = parse_module(source).expect("indirect-call module should parse");
    verify_module(
        &parsed.module,
        &VerifierConfig::for_level(VerificationLevel::Standard),
    )
}

#[test]
fn typed_indirect_call_verifies() {
    let report = verify(
        r#"
target = "wasm32-unknown-native"
func private %add_one(v0.i32) -> i32 {
    block0:
        v1.i32 = add v0 1.i32;
        return v1;
}
func public %run(v0.i32) -> i32 {
    block0:
        v1.*(i32) -> i32 = get_function_ptr %add_one;
        v2.i32 = call_indirect v1 *(i32) -> i32 v0;
        return v2;
}
"#,
    );
    assert!(report.is_ok(), "{report}");
}

#[test]
fn indirect_call_rejects_pointer_signature_and_argument_drift() {
    let report = verify(
        r#"
target = "wasm32-unknown-native"
func private %id(v0.i32) -> i32 {
    block0:
        return v0;
}
func public %bad(v0.i64) -> i32 {
    block0:
        v1.*(i32) -> i32 = get_function_ptr %id;
        v2.i32 = call_indirect v1 *(i64) -> i32 v0;
        return v2;
}
"#,
    );
    assert!(!report.is_ok());
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code.as_str() == "IR0602"),
        "{report}"
    );
}
