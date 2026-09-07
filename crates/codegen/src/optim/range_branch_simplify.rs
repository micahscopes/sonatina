use sonatina_ir::{BlockId, ControlFlowGraph, Function, InstId, inst::control_flow::BranchKind};

use crate::{
    cfg_edit::{CfgEditor, CleanupMode},
    domtree::DomTree,
    loop_analysis::LoopTree,
    range_analysis::{RangeAnalysis, RangeEnv, condition_truth_in_env},
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

    pub fn run(
        &mut self,
        func: &mut Function,
        cfg: &ControlFlowGraph,
        dom: &DomTree,
        lpt: &LoopTree,
    ) -> bool {
        self.run_proved(func, cfg, dom, lpt, &RangeEnv::default())
    }

    pub(crate) fn run_with_call_results(&mut self, func: &mut Function, cfg: &ControlFlowGraph, lpt: &LoopTree, call_results: &RangeEnv) -> bool {
        let mut dom = DomTree::default();
        dom.compute(cfg);
        self.run_proved(func, cfg, &dom, lpt, call_results)
    }

    fn run_proved(&mut self, func: &mut Function, cfg: &ControlFlowGraph, _dom: &DomTree, lpt: &LoopTree, call_results: &RangeEnv) -> bool {
        if !has_conditional_branch(func) {
            return false;
        }

        self.plans.clear();

        let mut analysis = RangeAnalysis::default();
        analysis.compute_with_call_results(func, cfg, lpt, call_results);

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

                analysis.transfer_inst(func, &mut env, inst);
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
