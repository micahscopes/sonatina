use sonatina_codegen::{Backend, BackendOptions, isa::spirv::SpirvBackend};
use sonatina_ir::{
    Linkage, Signature, Type,
    builder::ModuleBuilder,
    func_cursor::InstInserter,
    inst::{cast, control_flow, data},
    isa::{Isa, native::Native},
    module::ModuleCtx,
};
use sonatina_triple::{Architecture, OperatingSystem, TargetTriple, Vendor};

#[test]
fn typed_struct_source_labels_only_change_shader_debug_names() {
    let isa = Native::new(TargetTriple::new(
        Architecture::X86_64, Vendor::Unknown, OperatingSystem::Native,
    ));
    let is = isa.inst_set();
    let mb = ModuleBuilder::new(ModuleCtx::new(&isa));
    let pair = mb.declare_struct_type("Complex", &[Type::I32, Type::I32], false);
    let pair_ptr = mb.ptr_type(pair);
    let word_ptr = mb.ptr_type(Type::I32);
    let entry = mb.declare_function(Signature::new_single(
        "source_names", Linkage::Public, &[], Type::I32,
    )).unwrap();
    let mut fb = mb.func_builder::<InstInserter>(entry);
    let block = fb.append_block();
    fb.switch_to_block(block);
    let local = fb.insert_inst(data::Alloca::new_unchecked(is, pair), pair_ptr);
    let first = fb.insert_inst(cast::Bitcast::new_unchecked(is, local, word_ptr), word_ptr);
    let value = fb.make_imm_value(37i32);
    fb.insert_inst_no_result(data::Mstore::new_unchecked(is, first, value, Type::I32));
    let loaded = fb.insert_inst(data::Mload::new_unchecked(is, first, Type::I32), Type::I32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, loaded));
    fb.seal_all();
    fb.finish();
    let module = mb.build();
    let anonymous = SpirvBackend::new().compile_module(&module, &BackendOptions::default()).unwrap();
    module.ctx.set_struct_field_names(pair, vec!["re".into(), "im".into()]).unwrap();
    let named = SpirvBackend::new().compile_module(&module, &BackendOptions::default()).unwrap();
    let wgsl = named.wgsl.as_deref().unwrap();
    assert!(wgsl.contains("re:"), "{wgsl}");
    assert!(wgsl.contains("im:"), "{wgsl}");
    let parsed = naga::front::wgsl::parse_str(wgsl).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty(),
    ).validate(&parsed).unwrap();
    fn executable_words(words: &[u32]) -> Vec<u32> {
        let mut result = words[..5].to_vec();
        let mut offset = 5;
        while offset < words.len() {
            let count = (words[offset] >> 16) as usize;
            assert!(count > 0);
            let opcode = words[offset] & 65535;
            if opcode != 5 && opcode != 6 { // OpName and OpMemberName
                result.extend_from_slice(&words[offset..offset + count]);
            }
            offset += count;
        }
        result
    }
    assert_eq!(executable_words(&anonymous.words), executable_words(&named.words));
}
