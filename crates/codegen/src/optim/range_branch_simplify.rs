use sonatina_ir::{BlockId, ControlFlowGraph, Function, InstId, inst::control_flow::BranchKind};

use crate::{
    cfg_edit::{CfgEditor, CleanupMode},
    loop_analysis::LoopTree,
    range_analysis::{RangeAnalysis, RangeEnv, condition_truth_in_env, transfer_inst_with_call_results},
};

#[derive(Default)]
pub struct RangeBranchSimplify {
    plans: Vec<RewritePlan>,
}

#[derive(Clone, Copy)]
struct RewritePlan {
    block: BlockId,
    term: InstId,
    keep_mask: [bool; 2],
}

impl RangeBranchSimplify {
    pub fn new() -> Self {
        Self { plans: Vec::new() }
    }

    pub fn run(&mut self, func: &mut Function, cfg: &ControlFlowGraph, lpt: &LoopTree) -> bool {
        self.run_with_call_results(func, cfg, lpt, &RangeEnv::default())
    }

    pub(crate) fn run_with_call_results(
        &mut self, func: &mut Function, cfg: &ControlFlowGraph, lpt: &LoopTree,
        call_results: &RangeEnv,
    ) -> bool {
        self.run_with_inputs(func, cfg, lpt, call_results, &RangeEnv::default())
    }

    /// Specialize a function for caller-proved inclusive u32 argument bounds.
    /// The caller must enforce these bounds on EVERY invocation of the resulting
    /// function. This does not establish bounds from a suggested workload.
    /// Invalid argument positions/types/intervals fail before any mutation.
    pub fn run_with_u32_argument_bounds(
        &mut self, func: &mut Function, cfg: &ControlFlowGraph, lpt: &LoopTree,
        bounds: &[(usize, u32, u32)],
    ) -> Result<bool, String> {
        use crate::range_analysis::{RangeFact, UnsignedInterval};
        use sonatina_ir::{Type, U256};
        let mut arguments = RangeEnv::default();
        for &(index, low, high) in bounds {
            let arg = *func.arg_values.get(index).ok_or("bounded argument is absent")?;
            if func.dfg.value_ty(arg) != Type::I32 || low > high {
                return Err("expected an i32 lane and nonempty unsigned interval".into());
            }
            let mut fact = RangeFact::full_for(Type::I32);
            fact.unsigned = UnsignedInterval {lo: U256::from(low), hi: U256::from(high)};
            if arguments.insert(arg, fact).is_some() {
                return Err("duplicate bounded argument".into());
            }
        }
        Ok(self.run_with_inputs(func, cfg, lpt, &RangeEnv::default(), &arguments))
    }

    fn run_with_inputs(
        &mut self, func: &mut Function, cfg: &ControlFlowGraph, lpt: &LoopTree,
        call_results: &RangeEnv, arguments: &RangeEnv,
    ) -> bool {
        if !has_conditional_branch(func) {
            return false;
        }

        self.plans.clear();

        let mut analysis = RangeAnalysis::default();
        analysis.compute_with_inputs(func, cfg, lpt, call_results, arguments);

        let blocks: Vec<_> = func.layout.iter_block().collect();
        for block in blocks {
            if !analysis.is_reachable(block) {
                continue;
            }

            let mut env = analysis.entry_env(block).clone();
            let insts: Vec<_> = func.layout.iter_inst(block).collect();
            for inst in insts {
                if func.dfg.is_phi(inst) {
                    continue;
                }

                if let Some(plan) = plan_branch(func, block, &env, inst) {
                    self.plans.push(plan);
                }

                transfer_inst_with_call_results(func, &mut env, inst, call_results);
            }
        }

        if self.plans.is_empty() {
            return false;
        }

        apply_plans(func, &self.plans)
    }
}

pub(crate) fn has_conditional_branch(func: &Function) -> bool {
    func.layout.iter_block().any(|block| {
        func.layout.last_inst_of(block).is_some_and(|term| {
            func.dfg
                .branch_info(term)
                .is_some_and(|branch| matches!(branch.branch_kind(), BranchKind::Br(_)))
        })
    })
}

#[cfg(test)]
mod invocation_bounds_tests {
    use super::*;
    use crate::domtree::DomTree;
    use sonatina_ir::ir_writer::FuncWriter;

    #[test]
    fn invocation_bounds_remove_only_proved_overflow_edges() {
        for (bounds, removes_trap) in [
            (vec![], false),
            (vec![(0, 0, 255)], true),
            (vec![(0, 0, u32::MAX - 1)], true),
            (vec![(0, 0, u32::MAX)], false),
        ] {
            let parsed = sonatina_parser::parse_module(r#"
target = "shader-unknown-unknown"
func public %entry(v0.i32) -> i32 {
    block0:
        (v1.i32, v2.i1) = uaddo v0 1.i32;
        br v2 block1 block2;
    block1:
        unreachable;
    block2:
        return v1;
}
"#).unwrap();
            let module = parsed.module;
            let reference = module.funcs()[0];
            module.func_store.modify(reference, |function| {
                let mut cfg = ControlFlowGraph::default();
                cfg.compute(function);
                let mut dom = DomTree::default();
                dom.compute(&cfg);
                let mut loops = LoopTree::default();
                loops.compute(&cfg, &dom);
                let mut pass = RangeBranchSimplify::new();
                let before = FuncWriter::new(reference, function).dump_string();
                for invalid in [vec![(1, 0, 1)], vec![(0, 2, 1)], vec![(0, 0, 1), (0, 0, 1)]] {
                    assert!(pass.run_with_u32_argument_bounds(function, &cfg, &loops, &invalid).is_err());
                    assert_eq!(FuncWriter::new(reference, function).dump_string(), before);
                }
                pass.run_with_u32_argument_bounds(function, &cfg, &loops, &bounds).unwrap();
                let text = FuncWriter::new(reference, function).dump_string();
                assert_eq!(!text.contains("unreachable;"), removes_trap, "{bounds:?}: {text}");
                assert!(text.contains("uaddo"), "retain the arithmetic result");
            });
        }
    }
}

fn plan_branch(
    func: &Function,
    block: BlockId,
    env: &RangeEnv,
    term: InstId,
) -> Option<RewritePlan> {
    let branch = func.dfg.branch_info(term)?;
    let BranchKind::Br(br) = branch.branch_kind() else {
        return None;
    };

    let truth = condition_truth_in_env(func, env, *br.cond())?;
    Some(RewritePlan {
        block,
        term,
        keep_mask: if truth { [true, false] } else { [false, true] },
    })
}

fn apply_plans(func: &mut Function, plans: &[RewritePlan]) -> bool {
    let mut editor = CfgEditor::new(func, CleanupMode::Strict);
    let mut changed = false;

    for plan in plans {
        if !editor.func().layout.is_block_inserted(plan.block) {
            continue;
        }

        let Some(term) = editor.func().layout.last_inst_of(plan.block) else {
            continue;
        };
        if term != plan.term {
            continue;
        }

        let is_br = editor
            .func()
            .dfg
            .branch_info(term)
            .is_some_and(|branch| matches!(branch.branch_kind(), BranchKind::Br(_)));
        if !is_br {
            continue;
        }

        changed |= editor.retain_out_edges(plan.block, &plan.keep_mask);
    }

    if changed {
        let unreachable: Vec<_> = {
            let reachable = editor.cfg().reachable_blocks();
            editor
                .func()
                .layout
                .iter_block()
                .filter(|block| !reachable[*block])
                .collect()
        };
        changed |= editor.delete_blocks_unreachable(&unreachable);
    }

    changed
}
