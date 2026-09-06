//! Exact scalar expression reuse along the dominator tree.
//!
//! Unlike complete predicated GVN, this constructs no value-phi expressions and
//! performs no code motion. The scoped table contains only expressions already
//! evaluated on every path to the current instruction. Calls, memory operations,
//! phis, aggregates, and expressions depending on undef are excluded.

use rustc_hash::{FxHashMap, FxHashSet};
use sonatina_ir::{
    BlockId, Function, Value, ValueId,
    func_cursor::{CursorLocation, FuncCursor, InstInserter},
    inst::{InstClassKind, InstKeyExt, OwnedInstKey},
};

use crate::domtree::{DomTree, DominatorTreeTraversable};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DominatorCseStats {
    pub removed_instructions: usize,
    pub reused_values: usize,
    pub peak_available_expressions: usize,
}

enum Visit {
    Enter(BlockId),
    Exit(Vec<OwnedInstKey>),
}

/// Requires a dominator tree for the current CFG; does not change that CFG.
/// Storage is bounded by input values, use edges, and available expressions,
/// without a recursive traversal or a path-expression expansion.
pub fn eliminate_dominated_expressions(
    func: &mut Function,
    domtree: &DomTree,
) -> DominatorCseStats {
    let mut stats = DominatorCseStats::default();
    let Some(entry) = func.layout.entry_block() else {
        return stats;
    };
    func.rebuild_users();
    let tainted = undef_dependent_values(func);
    let mut tree = DominatorTreeTraversable::default();
    tree.compute(domtree);
    let mut available: FxHashMap<OwnedInstKey, Vec<ValueId>> = FxHashMap::default();
    let mut visits = vec![Visit::Enter(entry)];
    while let Some(visit) = visits.pop() {
        let block = match visit {
            Visit::Enter(block) => block,
            Visit::Exit(keys) => {
                for key in keys {
                    available.remove(&key);
                }
                continue;
            }
        };
        let mut inserted = Vec::new();
        let instructions: Vec<_> = func.layout.iter_inst(block).collect();
        for inst in instructions {
            if !matches!(
                func.dfg.inst(inst).kind(),
                InstClassKind::Unary(_) | InstClassKind::Binary(_) | InstClassKind::Cast(_)
            ) || !func.dfg.has_value_semantics(inst)
                || func.dfg.inst_results(inst).is_empty()
                || func
                    .dfg
                    .inst(inst)
                    .collect_values()
                    .iter()
                    .any(|v| tainted.contains(v))
            {
                continue;
            }
            let results = func.dfg.inst_results(inst).to_vec();
            let types: Vec<_> = results.iter().map(|v| func.dfg.value_ty(*v)).collect();
            let key = func.dfg.inst(inst).owned_key(&types);
            if let Some(previous) = available.get(&key) {
                // The key includes all result types, operand order and authored
                // fields. In particular, checked arithmetic reuses BOTH the
                // value and its overflow flag, without discarding the check.
                assert_eq!(previous.len(), results.len());
                for (&result, &replacement) in results.iter().zip(previous) {
                    func.dfg.change_to_alias(result, replacement);
                }
                InstInserter::at_location(CursorLocation::At(inst)).remove_inst(func);
                stats.removed_instructions += 1;
                stats.reused_values += results.len();
            } else {
                inserted.push(key.clone());
                available.insert(key, results);
                stats.peak_available_expressions =
                    stats.peak_available_expressions.max(available.len());
            }
        }
        visits.push(Visit::Exit(inserted));
        for &child in tree.children_of(block).iter().rev() {
            visits.push(Visit::Enter(child));
        }
    }
    stats
}

// Conservative forward taint, including phi cycles. Each value is queued once;
// no expression is recursively expanded. Partial aggregate initialization may
// overtaint later extracts, which is preferable to commoning an undef choice.
fn undef_dependent_values(func: &Function) -> FxHashSet<ValueId> {
    let mut tainted: FxHashSet<_> = func
        .dfg
        .values_iter()
        .filter_map(|(id, value)| matches!(value, Value::Undef { .. }).then_some(id))
        .collect();
    let mut pending: Vec<_> = tainted.iter().copied().collect();
    while let Some(value) = pending.pop() {
        for &user in func.dfg.users(value) {
            for &result in func.dfg.inst_results(user) {
                if tainted.insert(result) {
                    pending.push(result);
                }
            }
        }
    }
    tainted
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonatina_ir::{ControlFlowGraph, ir_writer::FuncWriter};

    fn run(source: &str) -> (String, DominatorCseStats) {
        let module = sonatina_parser::parse_module(source).unwrap().module;
        let config = sonatina_verifier::VerifierConfig::for_level(
            sonatina_verifier::VerificationLevel::Full,
        );
        let before = sonatina_verifier::verify_module(&module, &config);
        assert!(!before.has_errors(), "invalid input: {before:?}");
        let id = module.funcs()[0];
        let stats = module.func_store.modify(id, |func| {
            let mut cfg = ControlFlowGraph::default();
            cfg.compute(func);
            let mut dom = DomTree::default();
            dom.compute(&cfg);
            let stats = eliminate_dominated_expressions(func, &dom);
            assert_eq!(
                eliminate_dominated_expressions(func, &dom).removed_instructions,
                0
            );
            stats
        });
        let text = module
            .func_store
            .view(id, |func| FuncWriter::new(id, func).dump_string());
        let after = sonatina_verifier::verify_module(&module, &config);
        assert!(!after.has_errors(), "invalid output: {after:?}");
        (text, stats)
    }

    #[test]
    fn same_block_reuse_cascades_without_reassociation() {
        let (text, stats) = run(r#"
target = "evm-ethereum-osaka"
func public %test(v0.i32, v1.i32) -> i32 {
 block0:
  v2.i32 = sub v0 v1;
  v3.i32 = sub v0 v1;
  v4.i32 = xor v2 v1;
  v5.i32 = xor v3 v1;
  v6.i32 = sub v1 v0;
  v7.i32 = add v4 v5;
  v8.i32 = add v7 v6;
  return v8;
}
"#);
        assert_eq!(stats.removed_instructions, 2);
        assert_eq!(text.matches(" = sub ").count(), 2);
        assert_eq!(text.matches(" = xor ").count(), 1);
    }

    #[test]
    fn reuse_dominates_both_children_but_not_siblings() {
        let (_, stats) = run(r#"
target = "evm-ethereum-osaka"
func public %test(v0.i32, v1.i1) -> i32 {
 block0:
  v2.i32 = add v0 7.i32;
  br v1 block1 block2;
 block1:
  v3.i32 = add v0 7.i32;
  v4.i32 = mul v3 3.i32;
  jump block3;
 block2:
  v5.i32 = add v0 7.i32;
  v6.i32 = mul v5 3.i32;
  jump block3;
 block3:
  v7.i32 = phi (v4 block1) (v6 block2);
  return v7;
}
"#);
        assert_eq!(stats.removed_instructions, 2);
    }

    #[test]
    fn checked_arithmetic_reuses_all_results() {
        let (text, stats) = run(r#"
target = "evm-ethereum-osaka"
func public %test(v0.i32, v1.i32) -> (i32, i1) {
 block0:
  (v2.i32, v3.i1) = uaddo v0 v1;
  (v4.i32, v5.i1) = uaddo v0 v1;
  v6.i32 = xor v2 v4;
  v7.i1 = or v3 v5;
  return (v6, v7);
}
"#);
        assert_eq!(stats.removed_instructions, 1);
        assert_eq!(stats.reused_values, 2);
        assert_eq!(text.matches(" = uaddo ").count(), 1);
    }

    #[test]
    fn undef_dependence_is_not_commoned() {
        let (_, stats) = run(r#"
target = "evm-ethereum-osaka"
func public %test(v0.i32) -> i32 {
 block0:
  v1.i32 = add undef.i32 v0;
  v2.i32 = mul v1 3.i32;
  v3.i32 = mul v1 3.i32;
  v4.i32 = add v2 v3;
  return v4;
}
"#);
        assert_eq!(stats.removed_instructions, 0);
    }

    #[test]
    fn memory_reads_and_calls_are_not_commoned() {
        let (text, stats) = run(r#"
target = "evm-ethereum-osaka"
func private %test(v0.objref<i32>) -> i32 {
 block0:
  v1.i32 = obj.load v0;
  obj.store v0 9.i32;
  v2.i32 = obj.load v0;
  v3.i32 = call %identity v1;
  v4.i32 = call %identity v1;
  v5.i32 = add v3 v4;
  v6.i32 = add v5 v2;
  return v6;
}
func private %identity(v0.i32) -> i32 {
 block0:
  return v0;
}
"#);
        assert_eq!(stats.removed_instructions, 0);
        assert_eq!(text.matches("obj.load").count(), 2);
        assert_eq!(text.matches("call %identity").count(), 2);
        assert!(text.contains("obj.store"));
    }

    #[test]
    fn loop_carried_values_remain_distinct_from_invariants() {
        let (text, stats) = run(r#"
target = "evm-ethereum-osaka"
func public %test(v0.i32, v1.i32) -> i32 {
 block0:
  v2.i32 = add v0 1.i32;
  jump block1;
 block1:
  v3.i32 = phi (v0 block0) (v4 block2);
  v5.i1 = lt v3 v1;
  br v5 block2 block3;
 block2:
  v4.i32 = add v3 1.i32;
  v6.i32 = add v0 1.i32;
  v7.i32 = xor v4 v6;
  jump block1;
 block3:
  v8.i32 = add v3 v2;
  return v8;
}
"#);
        assert_eq!(stats.removed_instructions, 1);
        assert_eq!(text.matches(" = phi ").count(), 1);
        assert_eq!(text.matches(" = add ").count(), 3);
    }

    #[test]
    fn result_types_are_part_of_expression_identity() {
        let (_, stats) = run(r#"
target = "evm-ethereum-osaka"
func public %test(v0.i64) -> (i32, i16) {
 block0:
  v1.i32 = trunc v0 i32;
  v2.i16 = trunc v0 i16;
  return (v1, v2);
}
"#);
        assert_eq!(stats.removed_instructions, 0);
    }

    #[cfg(feature = "wasm")]
    #[test]
    fn checked_results_and_traps_execute_before_and_after() {
        use crate::Backend;
        let source = r#"
target = "wasm32-unknown-native"
func public %checked(v0.i32, v1.i32) -> i32 {
 block0:
  (v2.i32, v3.i1) = uaddo v0 v1;
  (v4.i32, v5.i1) = uaddo v0 v1;
  v6.i1 = or v3 v5;
  br v6 block1 block2;
 block1:
  unreachable;
 block2:
  v7.i32 = xor v2 v4;
  return v7;
}
"#;
        let engine = wasmtime::Engine::default();
        for optimize in [false, true] {
            let module = sonatina_parser::parse_module(source).unwrap().module;
            if optimize {
                module.func_store.modify(module.funcs()[0], |func| {
                    let mut cfg = ControlFlowGraph::default();
                    cfg.compute(func);
                    let mut dom = DomTree::default();
                    dom.compute(&cfg);
                    assert_eq!(
                        eliminate_dominated_expressions(func, &dom).removed_instructions,
                        1
                    );
                });
            }
            let artifact = crate::isa::wasm::WasmBackend::new()
                .compile_module(&module)
                .unwrap();
            let executable = wasmtime::Module::new(&engine, &artifact.bytes).unwrap();
            let mut store = wasmtime::Store::new(&engine, ());
            let instance = wasmtime::Instance::new(&mut store, &executable, &[]).unwrap();
            let call = instance
                .get_typed_func::<(i32, i32), i32>(&mut store, "checked")
                .unwrap();
            for x in [0_u32, 1, 7, i32::MAX as u32, u32::MAX - 1, u32::MAX] {
                for y in [0_u32, 1, 2, i32::MAX as u32, u32::MAX] {
                    let actual = call.call(&mut store, (x as i32, y as i32));
                    if x.checked_add(y).is_some() {
                        assert_eq!(actual.unwrap(), 0, "optimized={optimize}, x={x}, y={y}");
                    } else {
                        assert!(
                            actual.is_err(),
                            "lost trap: optimized={optimize}, x={x}, y={y}"
                        );
                    }
                }
            }
        }
    }
}
