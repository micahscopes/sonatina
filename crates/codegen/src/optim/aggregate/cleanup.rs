use rustc_hash::FxHashSet;
use smallvec::SmallVec;
use sonatina_ir::{
    Function, InstId, ValueId,
    func_cursor::{CursorLocation, FuncCursor, InstInserter},
    inst::{data, downcast},
};

#[derive(Default)]
pub(crate) struct DeadPureInstCleanup {
    worklist: Vec<InstId>,
    queued: FxHashSet<InstId>,
}

impl DeadPureInstCleanup {
    pub(crate) fn run_with_current_users(&mut self, func: &mut Function) -> bool {
        self.worklist.clear();
        self.queued.clear();
        for block in func.layout.iter_block() {
            let mut next_inst = func.layout.first_inst_of(block);
            while let Some(inst) = next_inst {
                next_inst = func.layout.next_inst_of(inst);
                if is_dead_pure_inst(func, inst) && self.queued.insert(inst) {
                    self.worklist.push(inst);
                }
            }
        }

        let mut changed = false;
        while let Some(inst) = self.worklist.pop() {
            self.queued.remove(&inst);
            if !is_dead_pure_inst(func, inst) {
                continue;
            }

            let operands = inst_operands(func, inst);
            InstInserter::at_location(CursorLocation::At(inst)).remove_inst(func);
            changed = true;

            for operand in operands {
                let Some(def_inst) = func.dfg.value_inst(operand) else {
                    continue;
                };
                if self.queued.insert(def_inst) {
                    self.worklist.push(def_inst);
                }
            }
        }

        changed
    }
}

fn is_dead_pure_inst(func: &Function, inst: InstId) -> bool {
    if !func.layout.is_inst_inserted(inst) || func.dfg.side_effect(inst).has_effect() {
        return false;
    }

    let results = func.dfg.inst_results(inst);
    if results.is_empty() {
        return false;
    }

    results.iter().copied().all(|result| {
        !func
            .dfg
            .users(result)
            .copied()
            .any(|user| func.layout.is_inst_inserted(user))
    })
}

fn inst_operands(func: &Function, inst: InstId) -> SmallVec<[ValueId; 8]> {
    let mut operands = SmallVec::new();
    func.dfg
        .inst(inst)
        .for_each_value(&mut |value| operands.push(value));
    operands
}

/// Eliminate alloca instructions whose memory is only written to, never read.
///
/// This handles the case where Fe's ABI decoder allocates and populates an
/// aggregate (e.g., `[u256; 32]`) that exceeds the scalarization limit and is
/// subsequently unused in the handler body. The stores to the alloca and the
/// alloca itself become dead, but normal DCE/LoadStore can't catch them because
/// stores are side-effecting.
///
/// Algorithm:
/// 1. For each alloca, collect all transitive users (through GEP chains).
/// 2. Classify each user as write-only (Mstore with the pointer as addr),
///    pointer-producing (GEP), or read/escape (anything else).
/// 3. If ALL users are writes or pointer-producers, the alloca is dead:
///    remove all the stores, GEPs, and the alloca.
pub(crate) fn eliminate_dead_allocas(func: &mut Function) -> bool {
    let allocas: Vec<_> = func
        .layout
        .iter_block()
        .flat_map(|block| func.layout.iter_inst(block))
        .filter(|&inst| {
            func.layout.is_inst_inserted(inst)
                && downcast::<&data::Alloca>(func.inst_set(), func.dfg.inst(inst)).is_some()
        })
        .collect();

    let mut changed = false;
    for alloca_inst in allocas {
        let Some(alloca_value) = func.dfg.inst_result(alloca_inst) else {
            continue;
        };

        // Collect all pointer values derived from this alloca (through GEP/casts)
        let mut pointer_values: Vec<ValueId> = vec![alloca_value];
        let mut to_remove: Vec<InstId> = Vec::new();
        let mut is_dead = true;
        let mut i = 0;

        while i < pointer_values.len() {
            let ptr = pointer_values[i];
            i += 1;

            let user_insts: Vec<InstId> = func.dfg.users(ptr).copied().collect();
            for user_inst in user_insts {
                if !func.layout.is_inst_inserted(user_inst) {
                    continue;
                }

                // Check if this is a GEP (produces another pointer from the alloca)
                if downcast::<&data::Gep>(func.inst_set(), func.dfg.inst(user_inst)).is_some() {
                    if let Some(gep_result) = func.dfg.inst_result(user_inst) {
                        if !pointer_values.contains(&gep_result) {
                            pointer_values.push(gep_result);
                        }
                    }
                    to_remove.push(user_inst);
                    continue;
                }

                // Check if this is an Mstore with the pointer as the address
                if let Some(store) =
                    downcast::<&data::Mstore>(func.inst_set(), func.dfg.inst(user_inst))
                {
                    if *store.addr() == ptr {
                        to_remove.push(user_inst);
                        continue;
                    }
                }

                // Any other use (Mload, call, return, etc.) → alloca is live
                is_dead = false;
                break;
            }

            if !is_dead {
                break;
            }
        }

        if is_dead {
            // Remove all stores and GEPs, then the alloca
            for inst in to_remove {
                if func.layout.is_inst_inserted(inst) {
                    InstInserter::at_location(CursorLocation::At(inst)).remove_inst(func);
                }
            }
            if func.layout.is_inst_inserted(alloca_inst) {
                InstInserter::at_location(CursorLocation::At(alloca_inst)).remove_inst(func);
            }
            changed = true;
        }
    }

    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonatina_ir::{InstDowncast, Module, module::FuncRef};
    use sonatina_parser::parse_module;

    fn parse_test_module(src: &str) -> Module {
        parse_module(src).expect("parse should succeed").module
    }

    fn lookup_func(module: &Module, name: &str) -> FuncRef {
        module
            .funcs()
            .into_iter()
            .find(|&func_ref| module.ctx.func_sig(func_ref, |sig| sig.name() == name))
            .expect("function should exist")
    }

    #[test]
    fn dead_pure_cleanup_requeues_transitively_dead_defs() {
        let module = parse_test_module(
            r#"
target = "evm-ethereum-london"

func private %f() {
block0:
    v0.i256 = add 1.i256 2.i256;
    v1.i256 = add v0 3.i256;
    v2.i256 = sub v0 4.i256;
    return;
}
"#,
        );
        let func_ref = lookup_func(&module, "f");
        module.func_store.modify(func_ref, |func| {
            func.rebuild_users();
            assert!(DeadPureInstCleanup::default().run_with_current_users(func));
        });

        module.func_store.view(func_ref, |func| {
            for block in func.layout.iter_block() {
                for inst in func.layout.iter_inst(block) {
                    assert!(
                        <&sonatina_ir::inst::arith::Add as InstDowncast>::downcast(
                            func.inst_set(),
                            func.dfg.inst(inst),
                        )
                        .is_none(),
                        "dead add should be removed"
                    );
                    assert!(
                        <&sonatina_ir::inst::arith::Sub as InstDowncast>::downcast(
                            func.inst_set(),
                            func.dfg.inst(inst),
                        )
                        .is_none(),
                        "dead sub should be removed"
                    );
                }
            }
        });
    }
}
