//! Unsupported input is rejected with an error that names it.

use sonatina_codegen::{
    Backend, BackendOptions,
    isa::wasm::{WasmBackend, WasmError},
};

use crate::harness::{compile_valid, parse, parse_with_target};

fn compile_error(source: &str) -> WasmError {
    let module = parse(source);
    let mut errors = WasmBackend::new()
        .compile_module(&module, &BackendOptions::default())
        .expect_err("the module should be rejected");
    assert_eq!(errors.len(), 1, "{errors:?}");
    errors.remove(0)
}

fn message(source: &str) -> String {
    compile_error(source).to_string()
}

#[test]
fn i256_parameters_results_and_operands_are_named() {
    let parameter = message(
        "func public %f(v0.i256) -> i32 {
    block0:
        return 0.i32;
}",
    );
    assert!(
        parameter.contains("i256") && parameter.contains("parameter 0"),
        "{parameter}"
    );
    assert!(parameter.contains("function `f`"), "{parameter}");

    let result = message(
        "func public %g() -> i256 {
    block0:
        return 1.i256;
}",
    );
    assert!(
        result.contains("i256") && result.contains("result 0"),
        "{result}"
    );

    let operation = message(
        "func public %h(v0.i32) -> i32 {
    block0:
        v1.i256 = zext v0 i256;
        v2.i32 = trunc v1 i32;
        return v2;
}",
    );
    assert!(
        operation.contains("i256") && operation.contains("zext"),
        "{operation}"
    );
}

#[test]
fn i128_is_rejected_like_i256() {
    let message = message(
        "func public %f(v0.i128) -> i128 {
    block0:
        return v0;
}",
    );
    assert!(message.contains("i128"), "{message}");
}

#[test]
fn an_i256_immediate_is_named() {
    let message = message(
        "func public %f(v0.i32) -> i32 {
    block0:
        v1.i1 = eq v0 0.i32;
        v2.i256 = zext v1 i256;
        v3.i1 = is_zero v2;
        br v3 block1 block2;
    block1:
        return 1.i32;
    block2:
        return 2.i32;
}",
    );
    assert!(message.contains("i256"), "{message}");
}

#[test]
fn unsupported_instructions_are_named() {
    let cases = [
        (
            "alloca",
            "func public %f() -> i32 {
    block0:
        v0.*i32 = alloca i32;
        return 0.i32;
}",
        ),
        (
            "memzero",
            "func public %f(v0.*i8) {
    block0:
        memzero v0 4.i32;
        return;
}",
        ),
        (
            "gep",
            "func public %f(v0.*[i8; 4]) -> *i8 {
    block0:
        v1.*i8 = gep v0 0.i32 1.i32;
        return v1;
}",
        ),
        (
            "mem.alloc_dynamic",
            "func public %f(v0.i32) -> *i8 {
    block0:
        v1.*i8 = mem.alloc_dynamic v0;
        return v1;
}",
        ),
    ];
    for (name, source) in cases {
        let error = compile_error(source);
        let WasmError::Unsupported(text) = &error else {
            panic!("{name}: expected an unsupported error, got {error:?}");
        };
        assert!(text.contains(&format!("`{name}`")), "{name}: {text}");
        assert!(text.contains("function `f`"), "{name}: {text}");
    }
}

#[test]
fn global_variables_are_rejected() {
    let message = message(
        "global private i32 $counter = 3;
func public %f() -> i32 {
    block0:
        return 0.i32;
}",
    );
    assert!(
        message.contains("global variable") && message.contains("counter"),
        "{message}"
    );
}

#[test]
fn other_targets_are_rejected() {
    let module = parse_with_target(
        "evm-ethereum-london",
        "func public %f() {
    block0:
        return;
}",
    );
    let errors = WasmBackend::new()
        .compile_module(&module, &BackendOptions::default())
        .expect_err("an EVM module is not a wasm32 module");
    assert!(
        matches!(errors.as_slice(), [WasmError::UnsupportedTarget(text)] if text.contains("evm-ethereum-london"))
    );
}

#[test]
fn malformed_input_is_an_error_not_a_panic() {
    // `call` arity and a phi type mismatch are rejected by the verifier, but
    // the backend does not depend on it having run.
    let arity = message(
        "func private %callee(v0.i32) -> i32 {
    block0:
        return v0;
}
func public %f() -> i32 {
    block0:
        v0.i32 = call %callee;
        return v0;
}",
    );
    assert!(arity.contains("arguments"), "{arity}");

    let operand_types = message(
        "func public %f(v0.i32, v1.i64) -> i32 {
    block0:
        v2.i32 = add v0 v1;
        return v2;
}",
    );
    assert!(operand_types.contains("`add`"), "{operand_types}");
}

#[test]
fn compiling_a_module_twice_gives_identical_bytes() {
    let source = "func private %inc(v0.i64) -> i64 {
    block0:
        v1.i64 = add v0 1.i64;
        return v1;
}
func public %loop_sum(v0.i64) -> i64 {
    block0:
        jump block1;
    block1:
        v1.i64 = phi (0.i64 block0) (v3 block2);
        v2.i64 = phi (0.i64 block0) (v4 block2);
        v5.i1 = lt v1 v0;
        br v5 block2 block3;
    block2:
        v3.i64 = call %inc v1;
        v4.i64 = add v2 v1;
        jump block1;
    block3:
        return v2;
}
func public %pair(v0.i32) -> (i32, i64) {
    block0:
        v1.i64 = zext v0 i64;
        return (v0, v1);
}
";
    // Separate parses of the same text, not two compilations of one module.
    let first = compile_valid(&parse(source));
    let second = compile_valid(&parse(source));
    assert_eq!(first, second);
    assert!(!first.is_empty());
}
