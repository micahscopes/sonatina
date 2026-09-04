use sonatina_triple::{Architecture, TargetTriple};
use sonatina_ir::{Type, module::ModuleCtx};
use sonatina_ir::isa::{Endian, Isa, shader::Shader, wasm32::Wasm32};

#[test]
fn shader_triple_selects_shader_module_context() {
    let parsed = sonatina_parser::parse_module("target = \"shader-unknown-unknown\"\n")
        .unwrap_or_else(|errors| panic!("shader target must parse: {errors:?}"));
    assert_eq!(parsed.module.ctx.triple.architecture, Architecture::Shader);
}

#[test]
fn shader_identity_preserves_arena_representation() {
    let triple = TargetTriple::parse("shader-unknown-unknown").unwrap();
    assert_eq!(triple.to_string(), "shader-unknown-unknown");
    assert!(TargetTriple::parse("shader-unknown-native").is_err());
    let shader = Shader::new(triple);
    let wasm = Wasm32::new(TargetTriple::parse("wasm32-unknown-native").unwrap());
    let ctx = ModuleCtx::new(&shader);
    assert_eq!(shader.triple().architecture, Architecture::Shader);
    assert_eq!(shader.type_layout().pointer_repl(), Type::I32);
    assert_eq!(shader.type_layout().endian(), Endian::Le);
    assert_eq!(shader.address_spaces().all_spaces().len(), 1);
    for (ty, size, align) in [
        (Type::Unit, 0, 1), (Type::I1, 1, 1), (Type::I8, 1, 1),
        (Type::I16, 2, 2), (Type::I32, 4, 4), (Type::I64, 8, 8),
        (Type::I128, 16, 16), (Type::I256, 32, 32), (Type::F32, 4, 4),
    ] {
        for layout in [shader.type_layout(), wasm.type_layout()] {
            assert_eq!(layout.size_of(ty, &ctx).unwrap(), size);
            assert_eq!(layout.align_of(ty, &ctx).unwrap(), align);
        }
    }
    // A byte-arena bool is not a four-byte typed Naga local.
    assert_eq!(shader.type_layout().size_of(Type::I1, &ctx).unwrap(), 1);
    let builder = sonatina_ir::builder::ModuleBuilder::new(ctx);
    let mixed = builder.declare_struct_type("mixed", &[Type::I1, Type::I32, Type::I8], false);
    let array = builder.declare_array_type(mixed, 3);
    let module = builder.build();
    for layout in [shader.type_layout(), wasm.type_layout()] {
        assert_eq!(layout.size_of(mixed, &module.ctx).unwrap(), 12);
        assert_eq!(layout.align_of(mixed, &module.ctx).unwrap(), 4);
        assert_eq!(layout.size_of(array, &module.ctx).unwrap(), 36);
        assert_eq!(layout.align_of(array, &module.ctx).unwrap(), 4);
    }
}
