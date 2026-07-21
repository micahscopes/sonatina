use ir::ir_writer::ModuleWriter;
use sonatina_parser::parse_module;

const SOURCE: &str = r#"target = "evm-ethereum-london"

func public %negative_zero() -> f32 {
block0:
    return 0x80000000.f32;
}

func public %nan_payload() -> f32 {
block0:
    return 0x7fc01234.f32;
}
"#;

#[test]
fn f32_bits_parse_print_reparse_exactly() {
    let parsed = parse_module(SOURCE).expect("raw f32 bit literals should parse");
    let printed = ModuleWriter::with_debug_provider(&parsed.module, &parsed.debug).dump_string();

    assert!(
        printed.contains("0x80000000.f32"),
        "signed zero bits changed:\n{printed}"
    );
    assert!(
        printed.contains("0x7fc01234.f32"),
        "NaN payload bits changed:\n{printed}"
    );

    let reparsed = parse_module(&printed).expect("printed f32 bit literals should reparse");
    let reprinted =
        ModuleWriter::with_debug_provider(&reparsed.module, &reparsed.debug).dump_string();
    assert_eq!(
        reprinted, printed,
        "f32 textual IR must be an exact fixed point"
    );
}
