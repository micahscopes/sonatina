use sonatina_parser::parse_module;
use sonatina_verifier::{VerificationLevel, VerifierConfig, verify_module};

#[test]
fn atomic_object_rmw_requires_word_reference_operand_and_result() {
    let source = r#"
target = "wasm32-unknown-native"
func private %claim(v0.objref<i32>, v1.i32) -> i32 {
    block0:
        v2.i32 = obj.atomic.add v0 v1;
        v3.i32 = obj.atomic.umin v0 v2;
        obj.atomic.store v0 v3;
        v4.i32 = obj.atomic.load v0;
        return v4;
}
"#;
    let config = VerifierConfig::for_level(VerificationLevel::Standard);
    let parsed = parse_module(source).expect("atomic instructions parse");
    let report = verify_module(&parsed.module, &config);
    assert!(report.is_ok(), "{report}");
    for invalid in [
        source.replace("objref<i32>", "objref<f32>"),
        source.replace("v1.i32", "v1.f32"),
        source.replace("v2.i32", "v2.f32"),
        source.replace("v3.i32", "v3.f32"),
        source.replace("v4.i32", "v4.f32"),
    ] {
        let parsed = parse_module(&invalid).expect("malformed types still parse");
        assert!(!verify_module(&parsed.module, &config).is_ok(), "{invalid}");
    }
}
