//! Switch legalization for structured backends. Not a general CFG repair pass.

use sonatina_ir::{
    Function, InstDowncast, Type, Value,
    inst::{
        cmp::Eq,
        control_flow::{Br, BrTable, Jump, Phi},
    },
};

/// Preserve ordered cases and one predecessor identity per original switch
/// edge. Edge blocks keep repeated destinations from duplicating or losing phi
/// inputs. Unsupported default-less tables fail before changing the function.
/// The caller must provide verified SSA and an ISA supporting Eq, Br and Jump.
pub fn lower_switches(function: &mut Function) -> Result<usize, String> {
    let is = function.inst_set();
    let mut switches = Vec::new();
    for block in function.layout.iter_block() {
        let Some(inst) = function.layout.last_inst_of(block) else {
            continue;
        };
        if let Some(table) = <&BrTable as InstDowncast>::downcast(is, function.dfg.inst(inst)) {
            let default = (*table.default()).ok_or_else(|| {
                format!("structured switch at {block:?} requires an explicit default")
            })?;
            switches.push((
                block,
                inst,
                *table.scrutinee(),
                default,
                table.table().clone(),
            ));
        }
    }
    let count = switches.len();
    for (source, terminator, selector, default, cases) in switches {
        let mut edges = Vec::new();
        for destination in cases
            .iter()
            .map(|(_, destination)| *destination)
            .chain([default])
        {
            if edges.iter().any(|(target, _)| *target == destination) {
                continue;
            }
            let edge = function.dfg.make_block();
            function.layout.append_block(edge);
            let jump = function.dfg.make_inst(Jump::new(is, destination));
            function.layout.append_inst(jump, edge);
            let instructions = function.layout.iter_inst(destination).collect::<Vec<_>>();
            for inst in instructions {
                let Some(phi) = <&Phi as InstDowncast>::downcast(is, function.dfg.inst(inst))
                else {
                    continue;
                };
                let mut args = phi.args().clone();
                for (_, predecessor) in &mut args {
                    if *predecessor == source {
                        *predecessor = edge;
                    }
                }
                function
                    .dfg
                    .replace_inst(inst, Box::new(Phi::new(is, args)));
            }
            edges.push((destination, edge));
        }
        let edge_for = |destination| {
            edges
                .iter()
                .find(|(target, _)| *target == destination)
                .unwrap()
                .1
        };
        if cases.is_empty() {
            function
                .dfg
                .replace_inst(terminator, Box::new(Jump::new(is, edge_for(default))));
            continue;
        }
        let mut current = source;
        for (index, (value, destination)) in cases.iter().enumerate() {
            let comparison = function.dfg.make_inst(Eq::new(is, selector, *value));
            let condition = function.dfg.make_value(Value::Inst {
                inst: comparison,
                result_idx: 0,
                ty: Type::I1,
            });
            function.dfg.attach_result(comparison, condition);
            if current == source {
                function.layout.insert_inst_before(comparison, terminator);
            } else {
                function.layout.append_inst(comparison, current);
            }
            let otherwise = if index + 1 == cases.len() {
                edge_for(default)
            } else {
                let next = function.dfg.make_block();
                function.layout.append_block(next);
                next
            };
            let branch = Br::new(is, condition, edge_for(*destination), otherwise);
            if current == source {
                function.dfg.replace_inst(terminator, Box::new(branch));
            } else {
                let inst = function.dfg.make_inst(branch);
                function.layout.append_inst(inst, current);
            }
            current = otherwise;
        }
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonatina_ir::{
        Linkage, Signature,
        builder::ModuleBuilder,
        cfg::ControlFlowGraph,
        func_cursor::InstInserter,
        inst::control_flow::Return,
        ir_writer::FuncWriter,
        isa::{Isa, shader::Shader},
        module::ModuleCtx,
    };
    use sonatina_triple::{Architecture, OperatingSystem, TargetTriple, Vendor};

    #[test]
    fn missing_default_rejects_without_partial_normalization() {
        let isa = Shader::new(TargetTriple::new(
            Architecture::Shader,
            Vendor::Unknown,
            OperatingSystem::Unknown,
        ));
        let is = isa.inst_set();
        let builder = ModuleBuilder::new(ModuleCtx::new(&isa));
        let function = builder
            .declare_function(Signature::new(
                "no_default",
                Linkage::Public,
                &[Type::I32],
                &[Type::I32],
            ))
            .unwrap();
        let mut fb = builder.func_builder::<InstInserter>(function);
        let entry = fb.append_block();
        let second = fb.append_block();
        let exit = fb.append_block();
        fb.switch_to_block(entry);
        let selector = fb.args()[0];
        let zero = fb.make_imm_value(0i32);
        fb.insert_inst_no_result(BrTable::new(is, selector, Some(second), vec![(zero, exit)]));
        fb.switch_to_block(second);
        fb.insert_inst_no_result(BrTable::new(is, selector, None, vec![(zero, exit)]));
        fb.switch_to_block(exit);
        fb.insert_inst_no_result(Return::new_single(is, zero));
        fb.seal_all();
        fb.finish();
        let module = builder.build();
        module.func_store.modify(function, |body| {
            let before = FuncWriter::new(function, body).dump_string();
            assert!(
                lower_switches(body)
                    .unwrap_err()
                    .contains("requires an explicit default")
            );
            assert_eq!(FuncWriter::new(function, body).dump_string(), before);
        });
    }

    #[test]
    fn shared_switch_destinations_preserve_phi_edges_and_are_idempotent() {
        let isa = Shader::new(TargetTriple::new(
            Architecture::Shader,
            Vendor::Unknown,
            OperatingSystem::Unknown,
        ));
        let is = isa.inst_set();
        let builder = ModuleBuilder::new(ModuleCtx::new(&isa));
        let function = builder
            .declare_function(Signature::new(
                "switch",
                Linkage::Public,
                &[Type::I32],
                &[Type::I32],
            ))
            .unwrap();
        let mut fb = builder.func_builder::<InstInserter>(function);
        let entry = fb.append_block();
        let other = fb.append_block();
        let join = fb.append_block();
        fb.switch_to_block(entry);
        let selector = fb.args()[0];
        let zero = fb.make_imm_value(0i32);
        let one = fb.make_imm_value(1i32);
        let two = fb.make_imm_value(2i32);
        let direct = fb.make_imm_value(40i32);
        let indirect = fb.make_imm_value(7i32);
        fb.insert_inst_no_result(BrTable::new(
            is,
            selector,
            Some(join),
            vec![(zero, join), (one, join), (two, other)],
        ));
        fb.switch_to_block(other);
        fb.insert_inst_no_result(Jump::new(is, join));
        fb.switch_to_block(join);
        let result = fb.insert_inst(
            Phi::new(is, vec![(direct, entry), (indirect, other)]),
            Type::I32,
        );
        fb.insert_inst_no_result(Return::new_single(is, result));
        fb.seal_all();
        fb.finish();
        let module = builder.build();
        module.func_store.modify(function, |body| {
            assert_eq!(lower_switches(body).unwrap(), 1);
            let mut cfg = ControlFlowGraph::default();
            cfg.compute(body);
            let phi_inst = body.layout.first_inst_of(join).unwrap();
            let phi = <&Phi as InstDowncast>::downcast(is, body.dfg.inst(phi_inst)).unwrap();
            assert_eq!(
                phi.args().len(),
                2,
                "repeated destinations must share one phi edge"
            );
            assert!(phi.args().contains(&(indirect, other)));
            let new_edge = phi
                .args()
                .iter()
                .find(|(value, _)| *value == direct)
                .unwrap()
                .1;
            assert_ne!(new_edge, entry);
            assert_eq!(
                cfg.preds_of(join)
                    .copied()
                    .collect::<std::collections::HashSet<_>>(),
                phi.args().iter().map(|(_, block)| *block).collect()
            );
            let once = FuncWriter::new(function, body).dump_string();
            assert_eq!(lower_switches(body).unwrap(), 0);
            assert_eq!(FuncWriter::new(function, body).dump_string(), once);
            crate::structurize::structurize_function(body)
                .expect("normalized switch must structure");
        });
    }
}
