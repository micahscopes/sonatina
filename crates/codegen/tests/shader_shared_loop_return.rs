use sonatina_codegen::{Backend, isa::spirv::SpirvBackend};

#[test]
fn sibling_guarded_loop_returns_remain_callable_helpers() {
    let source = r#"
target = "shader-unknown-unknown"
func public %entry(v0.i32, v1.i32, v2.i1, v3.i1, v4.i32) -> i32 {
block0:
    v5.i32 = call %sibling_guarded_loop_return v2 v3 v4;
    return v5;
}
func private %sibling_guarded_loop_return(v0.i1, v1.i1, v2.i32) -> i32 {
block0:
    br v0 block1 block4;
block1:
    v3.i32 = phi (0.i32 block0) (v5 block3);
    v4.i1 = lt v3 v2;
    br v4 block2 block7;
block2:
    br v1 block8 block3;
block3:
    v5.i32 = add v3 1.i32;
    jump block1;
block4:
    v6.i32 = phi (0.i32 block0) (v8 block6);
    v7.i1 = lt v6 v2;
    br v7 block5 block7;
block5:
    br v1 block8 block6;
block6:
    v8.i32 = add v6 1.i32;
    jump block4;
block7:
    v9.i32 = add v2 1.i32;
    return v9;
block8:
    unreachable;
}

"#;
    let module = sonatina_parser::parse_module(source).unwrap().module;
    let config = sonatina_verifier::VerifierConfig::for_level(
        sonatina_verifier::VerificationLevel::Full,
    );
    let report = sonatina_verifier::verify_module(&module, &config);
    assert!(!report.has_errors(), "{report:?}");
    let artifact = SpirvBackend::new().with_grid().with_workgroup_size(1, 1, 1)
        .compile_module(&module, &sonatina_codegen::BackendOptions::default())
        .expect("shared returning loop exits must not force helper expansion");
    let wgsl = artifact.wgsl.as_deref().unwrap();
    assert!(wgsl.contains("fn sibling_guarded_loop_return"), "{wgsl}");
    let parsed = naga::front::wgsl::parse_str(wgsl).unwrap();
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty()).validate(&parsed).unwrap();
}
