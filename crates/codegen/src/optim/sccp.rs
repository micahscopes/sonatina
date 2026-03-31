//! This module contains a solver for Sparse Conditional Constant Propagation.
//!
//! The algorithm is based on Mark N. Wegman., Frank Kcnncth Zadeck.: Constant
//! propagation with conditional branches: ACM Transactions on Programming
//! Languages and Systems Volume 13 Issue 2 April 1991 pp 181–210: <https://doi.org/10.1145/103135.103136>

use std::collections::BTreeSet;

use cranelift_entity::SecondaryMap;
use rustc_hash::FxHashSet;
use sonatina_ir::{
    BlockId, ControlFlowGraph, DataFlowGraph, Function, Immediate, InstId, Type, Value, ValueId,
    func_cursor::{CursorLocation, FuncCursor, InstInserter},
    inst::{
        arith, cast,
        control_flow::{BranchInfo, BranchKind},
        downcast,
    },
    interpret::{Action, EvalValue, Interpret, State},
    prelude::*,
};

use super::{
    adce::AdceSolver,
    cfg_cleanup::CfgCleanup,
    sccp_simplify::{SimplifyAction, simplify_inst},
};
use crate::{
    analysis::known_bits::KnownBitsQuery,
    cfg_edit::{CfgEditor, CleanupMode},
};

#[derive(Debug)]
pub struct SccpSolver {
    lattice: SecondaryMap<ValueId, LatticeCell>,
    may_be_undef: SecondaryMap<ValueId, bool>,
    reachable_edges: FxHashSet<FlowEdge>,
    reachable_blocks: SecondaryMap<BlockId, bool>,

    flow_work: Vec<FlowEdge>,
    ssa_work: Vec<ValueId>,

    flow_work_set: FxHashSet<FlowEdge>,
    ssa_work_set: SecondaryMap<ValueId, bool>,
}

impl SccpSolver {
    pub fn new() -> Self {
        Self {
            lattice: SecondaryMap::default(),
            may_be_undef: SecondaryMap::default(),
            reachable_edges: FxHashSet::default(),
            reachable_blocks: SecondaryMap::default(),
            flow_work: Vec::default(),
            ssa_work: Vec::default(),
            flow_work_set: FxHashSet::default(),
            ssa_work_set: SecondaryMap::default(),
        }
    }
    pub fn run(&mut self, func: &mut Function, cfg: &mut ControlFlowGraph) {
        self.clear();

        let cleanup_mode = if cfg!(debug_assertions) {
            CleanupMode::Strict
        } else {
            CleanupMode::RepairWithUndef
        };
        CfgCleanup::new(cleanup_mode).run(func);

        let Some(entry_block) = func.layout.entry_block() else {
            return;
        };

        // Function arguments must be `LatticeCell::Top`
        for arg in &func.arg_values {
            self.lattice[*arg] = LatticeCell::Top;
        }

        let solve_known_bits = KnownBitsQuery::new(func);

        // Evaluate all values in entry block.
        self.reachable_blocks[entry_block] = true;
        self.eval_insts_in(func, entry_block, &solve_known_bits);

        while !(self.flow_work.is_empty() && self.ssa_work.is_empty()) {
            while let Some(edge) = self.flow_work.pop() {
                self.flow_work_set.remove(&edge);
                self.eval_edge(func, edge, &solve_known_bits);
            }

            while let Some(value) = self.ssa_work.pop() {
                self.ssa_work_set[value] = false;
                for &user in func.dfg.users(value) {
                    let user_block = func.layout.inst_block(user);
                    if self.reachable_blocks[user_block] {
                        if func.dfg.is_phi(user) {
                            self.eval_phi(func, user);
                        } else {
                            self.eval_inst(func, user, &solve_known_bits);
                        }
                    }
                }
            }
        }

        #[cfg(debug_assertions)]
        self.assert_no_executable_inst_results_are_bot(func);

        self.remove_unreachable_edges(func, cleanup_mode);
        cfg.compute(func);
        let fold_known_bits = KnownBitsQuery::new(func);
        self.fold_insts(func, cfg, &fold_known_bits);

        CfgCleanup::new(cleanup_mode).run(func);
        cfg.compute(func);

        AdceSolver::new().run(func);
        cfg.compute(func);
    }

    pub fn clear(&mut self) {
        self.lattice.clear();
        self.may_be_undef.clear();
        self.reachable_edges.clear();
        self.reachable_blocks.clear();
        self.flow_work.clear();
        self.ssa_work.clear();
        self.flow_work_set.clear();
        self.ssa_work_set.clear();
    }

    fn push_flow_work(&mut self, edge: FlowEdge) {
        if self.reachable_edges.contains(&edge) || self.flow_work_set.contains(&edge) {
            return;
        }

        self.flow_work.push(edge);
        self.flow_work_set.insert(edge);
    }

    fn push_ssa_work(&mut self, value: ValueId) {
        if self.ssa_work_set[value] {
            return;
        }

        self.ssa_work.push(value);
        self.ssa_work_set[value] = true;
    }

    fn eval_edge(&mut self, func: &mut Function, edge: FlowEdge, known_bits: &KnownBitsQuery) {
        let dest = edge.to;

        if self.reachable_edges.contains(&edge) {
            return;
        }
        self.reachable_edges.insert(edge);

        if self.reachable_blocks[dest] {
            self.eval_phis_in(func, dest);
        } else {
            self.reachable_blocks[dest] = true;
            self.eval_insts_in(func, dest, known_bits);
        }
    }

    fn eval_phis_in(&mut self, func: &Function, block: BlockId) {
        for inst in func.layout.iter_inst(block) {
            if func.dfg.is_phi(inst) {
                self.eval_phi(func, inst);
            }
        }
    }

    fn eval_phi(&mut self, func: &Function, inst: InstId) {
        debug_assert!(self.reachable_blocks[func.layout.inst_block(inst)]);

        let block = func.layout.inst_block(inst);
        let phi = func.dfg.cast_phi(inst).unwrap();

        let mut eval_result = LatticeCell::Bot;
        let mut eval_may_be_undef = false;
        for (val, from) in phi.args() {
            if self.is_reachable(func, *from, block) {
                let v_cell = self.cell_of(func, *val);
                eval_result = eval_result.join(v_cell);
                eval_may_be_undef |= self.may_be_undef_of(func, *val);
            }
        }

        let phi_value = func.dfg.inst_result(inst).unwrap();
        self.set_lattice_cell(phi_value, eval_result);
        self.set_may_be_undef(phi_value, eval_may_be_undef);
    }

    fn eval_insts_in(&mut self, func: &Function, block: BlockId, known_bits: &KnownBitsQuery) {
        for inst in func.layout.iter_inst(block) {
            if func.dfg.is_phi(inst) {
                self.eval_phi(func, inst);
            } else {
                self.eval_inst(func, inst, known_bits);
            }
        }
    }

    fn eval_inst(&mut self, func: &Function, inst_id: InstId, known_bits: &KnownBitsQuery) {
        debug_assert!(!func.dfg.is_phi(inst_id));
        if let Some(bi) = func.dfg.branch_info(inst_id) {
            self.eval_branch(func, inst_id, bi);
            return;
        };

        let inst_results = func.dfg.inst_results(inst_id).to_vec();
        if inst_results.is_empty() {
            return;
        }

        // SCCP here is intraprocedural. Do not attempt to interpret calls even
        // when callee attrs classify them as effect-free.
        if func.dfg.is_call(inst_id) {
            self.set_inst_results_top(&inst_results);
            return;
        }

        let inst = func.dfg.inst(inst_id);
        if func.dfg.side_effect(inst_id).has_effect() {
            self.set_inst_results_top(&inst_results);
            return;
        }

        let simplified =
            simplify_inst(func, &self.lattice, &self.may_be_undef, known_bits, inst_id);
        if simplified.len() != inst_results.len() {
            debug_assert_eq!(
                simplified.len(),
                inst_results.len(),
                "SCCP simplifier returned the wrong number of results for {inst_id:?}"
            );
            self.set_inst_results_top(&inst_results);
            return;
        }

        let mut result_states = vec![None; inst_results.len()];
        let mut all_results_simplified = true;
        for (idx, (&result, action)) in inst_results.iter().zip(simplified).enumerate() {
            match action {
                SimplifyAction::Const(imm) => {
                    let cell = self
                        .normalize_const_imm_for_value(func, result, imm)
                        .map(LatticeCell::Const)
                        .unwrap_or(LatticeCell::Top);
                    result_states[idx] = Some((cell, false));
                }
                SimplifyAction::Copy(src) => {
                    let src_cell =
                        self.normalize_cell_for_value(func, result, self.cell_of(func, src));
                    result_states[idx] = Some((src_cell, self.may_be_undef_of(func, src)));
                }
                SimplifyAction::NoChange => {
                    all_results_simplified = false;
                }
            }
        }

        if all_results_simplified {
            for (&result, state) in inst_results.iter().zip(result_states) {
                let (cell, may_be_undef) =
                    state.expect("all simplified SCCP results must have a state");
                self.set_lattice_cell(result, cell);
                self.set_may_be_undef(result, may_be_undef);
            }
            return;
        }

        let mut operand_may_be_undef = false;
        inst.for_each_value(&mut |value| {
            operand_may_be_undef |= self.may_be_undef_of(func, value);
        });
        let result_may_divide_by_zero = self.is_arith_div_or_rem_by_zero(func, inst_id);

        {
            let mut cell_state = CellState::new(&self.lattice, &func.dfg);
            let value = InstDowncast::map(func.inst_set(), inst, |i: &dyn Interpret| {
                i.interpret(&mut cell_state)
            });

            if let Some(values) = value.as_ref()
                && values.len() != inst_results.len()
            {
                debug_assert_eq!(
                    values.len(),
                    inst_results.len(),
                    "SCCP interpreted the wrong number of results for {inst_id:?}"
                );
                self.set_inst_results_top(&inst_results);
                return;
            }

            for (idx, &result) in inst_results.iter().enumerate() {
                if result_states[idx].is_some() {
                    continue;
                }

                let eval_value = value.as_ref().and_then(|values| values.get(idx));
                let cell = match eval_value {
                    Some(EvalValue::Imm(value)) => self
                        .normalize_const_imm_for_value(func, result, *value)
                        .map(LatticeCell::Const)
                        .unwrap_or(LatticeCell::Top),
                    Some(_) => cell_state.nonconst_result_cell(),
                    None => LatticeCell::Top,
                };
                let may_be_undef = operand_may_be_undef
                    || result_may_divide_by_zero
                    || (eval_value.is_some_and(EvalValue::is_undef) && !cell_state.used_has_top);
                result_states[idx] = Some((cell, may_be_undef));
            }
        }

        for (&result, state) in inst_results.iter().zip(result_states) {
            let (cell, may_be_undef) = state.expect("every SCCP result must have a final state");
            self.set_lattice_cell(result, cell);
            self.set_may_be_undef(result, may_be_undef);
        }
    }

    fn is_arith_div_or_rem_by_zero(&self, func: &Function, inst_id: InstId) -> bool {
        let inst = func.dfg.inst(inst_id);
        let is = func.inst_set();

        let Some(rhs) = downcast::<&arith::Udiv>(is, inst)
            .map(|i| *i.rhs())
            .or_else(|| downcast::<&arith::Sdiv>(is, inst).map(|i| *i.rhs()))
            .or_else(|| downcast::<&arith::Umod>(is, inst).map(|i| *i.rhs()))
            .or_else(|| downcast::<&arith::Smod>(is, inst).map(|i| *i.rhs()))
        else {
            return false;
        };

        self.cell_of(func, rhs)
            .to_imm()
            .is_some_and(|imm| imm.is_zero())
    }

    fn eval_branch(&mut self, func: &Function, inst: InstId, bi: &dyn BranchInfo) {
        match bi.branch_kind() {
            BranchKind::Jump(jump) => {
                self.push_flow_work(FlowEdge::new(inst, *jump.dest()));
            }

            BranchKind::Br(br) => {
                let v_cell = self.cell_of(func, *br.cond());

                match v_cell {
                    LatticeCell::Const(_) if v_cell.is_zero() => {
                        self.push_flow_work(FlowEdge::new(inst, *br.z_dest()));
                    }
                    LatticeCell::Const(_) => {
                        self.push_flow_work(FlowEdge::new(inst, *br.nz_dest()));
                    }
                    LatticeCell::Top => {
                        self.push_flow_work(FlowEdge::new(inst, *br.z_dest()));
                        self.push_flow_work(FlowEdge::new(inst, *br.nz_dest()));
                    }
                    LatticeCell::Bot => {}
                }
            }

            BranchKind::BrTable(brt) => {
                let scrutinee_cell = self.cell_of(func, *brt.scrutinee());

                match scrutinee_cell {
                    LatticeCell::Const(scrutinee) => {
                        for (case_value, dest) in brt.table() {
                            match self.cell_of(func, *case_value) {
                                LatticeCell::Const(case) => {
                                    if case == scrutinee {
                                        self.push_flow_work(FlowEdge::new(inst, *dest));
                                        return;
                                    }
                                }
                                LatticeCell::Top | LatticeCell::Bot => {
                                    self.push_flow_work(FlowEdge::new(inst, *dest));
                                }
                            }
                        }

                        if let Some(default) = brt.default() {
                            self.push_flow_work(FlowEdge::new(inst, *default));
                        }
                    }
                    LatticeCell::Top => {
                        if let Some(default) = brt.default() {
                            self.push_flow_work(FlowEdge::new(inst, *default));
                        }
                        for (_, dest) in brt.table() {
                            self.push_flow_work(FlowEdge::new(inst, *dest));
                        }
                    }
                    LatticeCell::Bot => {}
                }
            }
        }
    }

    /// Remove unreachable edges and blocks.
    fn remove_unreachable_edges(&self, func: &mut Function, mode: CleanupMode) {
        let mut editor = CfgEditor::new(func, mode);

        let blocks: Vec<_> = editor.func().layout.iter_block().collect();
        let unreachable: Vec<_> = blocks
            .iter()
            .copied()
            .filter(|block| !self.reachable_blocks[*block])
            .collect();
        editor.delete_blocks_unreachable(&unreachable);

        let blocks: Vec<_> = editor.func().layout.iter_block().collect();
        for block in blocks {
            if !self.reachable_blocks[block] {
                continue;
            }

            let Some(term) = editor.func().layout.last_inst_of(block) else {
                continue;
            };
            let Some(branch_info) = editor.func().dfg.branch_info(term) else {
                continue;
            };

            for dest in branch_info.dests() {
                if !self.is_reachable_edge(term, dest) {
                    editor.remove_succ(block, dest);
                }
            }
        }
    }

    fn is_reachable_edge(&self, inst: InstId, dest: BlockId) -> bool {
        self.reachable_edges.contains(&FlowEdge::new(inst, dest))
    }

    fn fold_insts(
        &mut self,
        func: &mut Function,
        cfg: &ControlFlowGraph,
        known_bits: &KnownBitsQuery,
    ) {
        let mut rpo: Vec<_> = cfg.post_order().collect();
        rpo.reverse();

        for block in rpo {
            let mut next_inst = func.layout.first_inst_of(block);
            while let Some(inst) = next_inst {
                next_inst = func.layout.next_inst_of(inst);
                self.fold(func, inst, known_bits);
            }
        }
    }

    fn fold(&self, func: &mut Function, inst: InstId, known_bits: &KnownBitsQuery) {
        let inst_results = func.dfg.inst_results(inst).to_vec();
        if inst_results.is_empty() {
            return;
        }

        let mut changed = false;
        if !func.dfg.is_phi(inst) {
            let simplified =
                simplify_inst(func, &self.lattice, &self.may_be_undef, known_bits, inst);
            debug_assert_eq!(
                simplified.len(),
                inst_results.len(),
                "SCCP simplifier returned the wrong number of fold results for {inst:?}"
            );

            for (idx, (&result, action)) in inst_results.iter().zip(simplified).enumerate() {
                match action {
                    SimplifyAction::Const(imm) => {
                        let result_ty = func.dfg.value_ty(result);
                        debug_assert_eq!(
                            imm.ty(),
                            result_ty,
                            "SCCP tried to fold an immediate of a different type: {inst:?}"
                        );
                        if imm.ty() != result_ty {
                            continue;
                        }

                        let new_value = func.dfg.make_imm_value(imm);
                        func.dfg.change_to_alias(result, new_value);
                        changed = true;
                    }
                    SimplifyAction::Copy(src) => {
                        let result_ty = func.dfg.value_ty(result);
                        let src_ty = func.dfg.value_ty(src);
                        if src_ty == result_ty {
                            func.dfg.change_to_alias(result, src);
                            changed = true;
                        } else if idx == 0
                            && inst_results.len() == 1
                            && let Some(bitcast) = func.inst_set().has_bitcast()
                            && func.dfg.ctx.size_of(src_ty).ok()
                                == func.dfg.ctx.size_of(result_ty).ok()
                        {
                            func.dfg.replace_inst(
                                inst,
                                Box::new(cast::Bitcast::new(bitcast, src, result_ty)),
                            );
                            return;
                        }
                    }
                    SimplifyAction::NoChange => {}
                }
            }
        }

        for &result in &inst_results {
            if func.dfg.users_num(result) == 0 {
                continue;
            }

            let Some(imm) = self.lattice[result].to_imm() else {
                continue;
            };
            let result_ty = func.dfg.value_ty(result);
            debug_assert_eq!(
                imm.ty(),
                result_ty,
                "SCCP tried to fold an immediate of a different type: {inst:?}"
            );
            if imm.ty() != result_ty {
                continue;
            }

            let new_value = func.dfg.make_imm_value(imm);
            func.dfg.change_to_alias(result, new_value);
            changed = true;
        }

        if changed && self.inst_results_are_dead(func, inst) {
            InstInserter::at_location(CursorLocation::At(inst)).remove_inst(func);
        } else if !changed && func.dfg.is_phi(inst) {
            self.try_fold_phi(func, inst);
        }
    }

    fn try_fold_phi(&self, func: &mut Function, inst: InstId) {
        let block = func.layout.inst_block(inst);
        let phi_value = func.dfg.inst_result(inst).expect("phi has no result");
        let phi_ty = func.dfg.value_ty(phi_value);


        let reachable_preds: BTreeSet<_> = func
            .dfg
            .cast_phi(inst)
            .unwrap()
            .args()
            .iter()
            .map(|&(_, pred)| pred)
            .filter(|pred| self.is_reachable(func, *pred, block))
            .collect();

        func.dfg.untrack_inst(inst);

        let mut fold_arg = None;
        {
            let phi = func.dfg.cast_phi_mut(inst).unwrap();
            phi.retain(|pred| reachable_preds.contains(&pred));

            let mut seen = BTreeSet::new();
            for &(_, pred) in phi.args() {
                assert!(
                    seen.insert(pred),
                    "phi {inst:?} has duplicate incoming from {pred:?}"
                );
            }

            if phi.args().len() == 1 {
                fold_arg = Some(phi.args()[0].0);
            }
        }

        let fold_arg = if fold_arg.is_none() && func.dfg.cast_phi(inst).unwrap().args().is_empty() {
            Some(func.dfg.make_undef_value(phi_ty))
        } else {
            fold_arg
        };

        // Remove phi function if it has 0 or 1 arguments.
        if let Some(phi_arg) = fold_arg {
            let phi_arg = if phi_arg == phi_value {
                func.dfg.make_undef_value(phi_ty)
            } else {
                phi_arg
            };
            func.dfg.change_to_alias(phi_value, phi_arg);
            InstInserter::at_location(CursorLocation::At(inst)).remove_inst(func);
        } else if let Some(cond_value) =
            try_fold_branch_select_phi(func, inst, phi_value, phi_ty)
        {
            func.dfg.change_to_alias(phi_value, cond_value);
            InstInserter::at_location(CursorLocation::At(inst)).remove_inst(func);
        } else {
            func.dfg.attach_user(inst);
        }
    }

    fn is_reachable(&self, func: &Function, from: BlockId, to: BlockId) -> bool {
        if !func.layout.is_block_inserted(from) || !func.layout.is_block_inserted(to) {
            return false;
        }

        let Some(last_inst) = func.layout.last_inst_of(from) else {
            return false;
        };

        let Some(bi) = func.dfg.branch_info(last_inst) else {
            return false;
        };

        for dest in bi.dests() {
            if dest == to
                && self
                    .reachable_edges
                    .contains(&FlowEdge::new(last_inst, dest))
            {
                return true;
            }
        }
        false
    }

    fn set_lattice_cell(&mut self, value: ValueId, cell: LatticeCell) {
        let old_cell = &self.lattice[value];
        if old_cell != &cell {
            self.lattice[value] = cell;
            self.push_ssa_work(value);
        }
    }

    fn set_may_be_undef(&mut self, value: ValueId, may_be_undef: bool) {
        let old = self.may_be_undef.get(value).copied().unwrap_or_default();
        if old != may_be_undef {
            self.may_be_undef[value] = may_be_undef;
            self.push_ssa_work(value);
        }
    }

    fn set_inst_results_top(&mut self, results: &[ValueId]) {
        for &result in results {
            self.set_lattice_cell(result, LatticeCell::Top);
            self.set_may_be_undef(result, false);
        }
    }

    fn inst_results_are_dead(&self, func: &Function, inst: InstId) -> bool {
        func.dfg
            .inst_results(inst)
            .iter()
            .all(|&result| func.dfg.users_num(result) == 0)
    }

    fn cell_of(&self, func: &Function, value: ValueId) -> LatticeCell {
        match func.dfg.value(value) {
            Value::Immediate { imm, .. } => LatticeCell::Const(*imm),
            Value::Global { .. } | Value::Undef { .. } => LatticeCell::Top,
            Value::Arg { .. } | Value::Inst { .. } => {
                self.lattice.get(value).copied().unwrap_or_default()
            }
        }
    }

    fn may_be_undef_of(&self, func: &Function, value: ValueId) -> bool {
        match func.dfg.value(value) {
            Value::Undef { .. } => true,
            Value::Arg { .. } | Value::Immediate { .. } | Value::Global { .. } => false,
            Value::Inst { .. } => self.may_be_undef.get(value).copied().unwrap_or_default(),
        }
    }

    fn normalize_const_imm_for_value(
        &self,
        func: &Function,
        value: ValueId,
        imm: Immediate,
    ) -> Option<Immediate> {
        let ty = func.dfg.value_ty(value);
        if !ty.is_integral() {
            None
        } else if imm.ty() == ty {
            Some(imm)
        } else {
            Some(Immediate::from_i256(imm.as_i256(), ty))
        }
    }

    fn normalize_cell_for_value(
        &self,
        func: &Function,
        value: ValueId,
        cell: LatticeCell,
    ) -> LatticeCell {
        if let LatticeCell::Const(imm) = cell {
            self.normalize_const_imm_for_value(func, value, imm)
                .map(LatticeCell::Const)
                .unwrap_or(LatticeCell::Top)
        } else {
            cell
        }
    }

    #[cfg(debug_assertions)]
    fn assert_no_executable_inst_results_are_bot(&self, func: &Function) {
        for block in func.layout.iter_block() {
            if !self.reachable_blocks[block] {
                continue;
            }
            for inst in func.layout.iter_inst(block) {
                if func.dfg.is_phi(inst) || func.dfg.branch_info(inst).is_some() {
                    continue;
                }

                let inst_data = func.dfg.inst(inst);
                if inst_data.side_effect().has_effect() {
                    continue;
                }

                let results = func.dfg.inst_results(inst);
                if results.is_empty() {
                    continue;
                }

                let mut operands_all_non_bot = true;
                inst_data.for_each_value(&mut |value| {
                    if self.cell_of(func, value).is_bot() {
                        operands_all_non_bot = false;
                    }
                });

                for &result in results {
                    let result_cell = self.lattice.get(result).copied().unwrap_or_default();
                    if operands_all_non_bot && result_cell.is_bot() {
                        panic!("SCCP produced Bot for executable inst result: {inst:?}");
                    }
                }
            }
        }
    }
}

impl Default for SccpSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct FlowEdge {
    inst: InstId,
    to: BlockId,
}

impl FlowEdge {
    fn new(inst: InstId, to: BlockId) -> Self {
        Self { inst, to }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) enum LatticeCell {
    Top,
    Const(Immediate),
    #[default]
    Bot,
}

impl PartialEq for LatticeCell {
    fn eq(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (Self::Top, Self::Top) | (Self::Bot, Self::Bot) => true,
            (Self::Const(v1), Self::Const(v2)) => v1 == v2,
            _ => false,
        }
    }
}

impl LatticeCell {
    fn to_imm(self) -> Option<Immediate> {
        match self {
            Self::Top | Self::Bot => None,
            Self::Const(imm) => Some(imm),
        }
    }

    fn is_zero(self) -> bool {
        match self {
            Self::Top | Self::Bot => false,
            Self::Const(c) => c.is_zero(),
        }
    }

    #[cfg(debug_assertions)]
    fn is_bot(self) -> bool {
        matches!(self, Self::Bot)
    }

    fn join(self, rhs: Self) -> Self {
        match (self, rhs) {
            (Self::Top, _) | (_, Self::Top) => Self::Top,
            (Self::Const(v1), Self::Const(v2)) => {
                if v1 == v2 {
                    self
                } else {
                    Self::Top
                }
            }
            (Self::Bot, other) | (other, Self::Bot) => other,
        }
    }
}

struct CellState<'a, 'i> {
    map: &'i SecondaryMap<ValueId, LatticeCell>,
    used_has_top: bool,
    used_has_bot: bool,
    dfg: &'a DataFlowGraph,
}
impl<'a, 'i> CellState<'a, 'i> {
    fn new(map: &'i SecondaryMap<ValueId, LatticeCell>, dfg: &'a DataFlowGraph) -> Self {
        Self {
            map,
            used_has_top: false,
            used_has_bot: false,
            dfg,
        }
    }

    fn nonconst_result_cell(&self) -> LatticeCell {
        match (self.used_has_top, self.used_has_bot) {
            (true, _) => LatticeCell::Top,
            (false, true) => LatticeCell::Bot,
            (false, false) => LatticeCell::Top,
        }
    }
}

impl State for CellState<'_, '_> {
    fn lookup_val(&mut self, value: ValueId) -> EvalValue {
        let cell = match self.dfg.value(value) {
            Value::Immediate { imm, .. } => LatticeCell::Const(*imm),
            Value::Global { .. } | Value::Undef { .. } => LatticeCell::Top,
            Value::Arg { .. } | Value::Inst { .. } => {
                self.map.get(value).copied().unwrap_or_default()
            }
        };

        match cell {
            LatticeCell::Const(imm) => EvalValue::Imm(imm),
            LatticeCell::Top => {
                self.used_has_top = true;
                EvalValue::Undef
            }
            LatticeCell::Bot => {
                self.used_has_bot = true;
                EvalValue::Undef
            }
        }
    }

    fn call_func(
        &mut self,
        _func: sonatina_ir::module::FuncRef,
        _args: Vec<EvalValue>,
    ) -> sonatina_ir::interpret::EvalResults {
        panic!("call instuctuion must not be Interpreted")
    }

    fn set_action(&mut self, action: Action) {
        if action != Action::Continue {
            panic!("instruction with side effect must not be interpreted")
        }
    }

    fn prev_block(&mut self) -> BlockId {
        panic!("flow sensitive operation must not be interpreted")
    }

    fn load(&mut self, _addr: EvalValue, _ty: Type) -> EvalValue {
        panic!("instruction with side effect must not be interpreted")
    }

    fn store(&mut self, _addr: EvalValue, _value: EvalValue, _ty: Type) -> EvalValue {
        panic!("instruction with side effect must not be interpreted")
    }

    fn alloca(&mut self, _ty: Type) -> EvalValue {
        panic!("instruction with side effect must not be interpreted")
    }

    fn dfg(&self) -> &DataFlowGraph {
        self.dfg
    }
}

/// Fold a phi of the form `phi(const_0, const_1)` where the two predecessors
/// are reached via a conditional branch on some `cond: I1`.
///
/// Patterns matched:
///   phi(0 from pred_z, 1 from pred_nz)  →  zext(cond, phi_ty)
///   phi(1 from pred_z, 0 from pred_nz)  →  zext(is_zero(cond), phi_ty)
///   phi(0 from pred_nz, 1 from pred_z)  →  zext(is_zero(cond), phi_ty)
///   phi(1 from pred_nz, 0 from pred_z)  →  zext(cond, phi_ty)
///
/// This eliminates branch+phi overhead for Fe's `if cond { 1 } else { 0 }`.
fn try_fold_branch_select_phi(
    func: &mut Function,
    phi_inst: InstId,
    _phi_value: ValueId,
    phi_ty: Type,
) -> Option<ValueId> {
    if !phi_ty.is_integral() {
        return None;
    }

    let phi = func.dfg.cast_phi(phi_inst)?;
    let args = phi.args();
    if args.len() != 2 {
        return None;
    }

    let (val_a, pred_a) = args[0];
    let (val_b, pred_b) = args[1];

    // Both values must be known immediates.
    let imm_a = func.dfg.value_imm(val_a)?;
    let imm_b = func.dfg.value_imm(val_b)?;

    let zero = Immediate::zero(phi_ty);
    let one = Immediate::one(phi_ty);

    // Determine which value is 0 and which is 1.
    let (zero_pred, one_pred) = if imm_a == zero && imm_b == one {
        (pred_a, pred_b)
    } else if imm_a == one && imm_b == zero {
        (pred_b, pred_a)
    } else {
        return None;
    };

    // Both predecessors must be trivial blocks (only phis + jump).
    // Find the common "branch" block that reaches both predecessors.
    let branch_block = find_common_branch_predecessor(func, zero_pred, one_pred)?;

    // The branch block must end with a conditional branch.
    let term = func.layout.last_inst_of(branch_block)?;
    use sonatina_ir::inst::control_flow::Br;
    let br = downcast::<&Br>(func.inst_set(), func.dfg.inst(term))?;
    let cond = *br.cond();
    let nz_dest = *br.nz_dest();
    let z_dest = *br.z_dest();

    // Match the branch targets to our phi predecessors.
    let needs_invert = if nz_dest == one_pred && z_dest == zero_pred {
        // cond=true → 1, cond=false → 0 → result = zext(cond)
        false
    } else if nz_dest == zero_pred && z_dest == one_pred {
        // cond=true → 0, cond=false → 1 → result = zext(is_zero(cond))
        true
    } else {
        return None;
    };

    // Build the replacement value.
    let phi_block = func.layout.inst_block(phi_inst);
    let insert_before = func.layout.first_inst_of(phi_block)?;

    let is = func.inst_set();
    use sonatina_ir::inst::cmp::IsZero;
    let result = if needs_invert {
        // Insert is_zero(cond), then zext
        let Some(iz_is) = is.has_is_zero() else { return None };
        let iz_inst = func.dfg.make_inst(IsZero::new(iz_is, cond));
        let iz_val = func.dfg.make_value(Value::Inst {
            inst: iz_inst,
            result_idx: 0,
            ty: Type::I1,
        });
        func.dfg.attach_result(iz_inst, iz_val);
        func.layout.insert_inst_before(iz_inst, insert_before);

        if phi_ty == Type::I1 {
            iz_val
        } else {
            let Some(zext_is) = is.has_zext() else { return None };
            let zext_inst = func.dfg.make_inst(cast::Zext::new(zext_is, iz_val, phi_ty));
            let zext_val = func.dfg.make_value(Value::Inst {
                inst: zext_inst,
                result_idx: 0,
                ty: phi_ty,
            });
            func.dfg.attach_result(zext_inst, zext_val);
            func.layout.insert_inst_before(zext_inst, insert_before);
            zext_val
        }
    } else if phi_ty == Type::I1 {
        cond
    } else {
        let Some(zext_is) = is.has_zext() else { return None };
        let zext_inst = func.dfg.make_inst(cast::Zext::new(zext_is, cond, phi_ty));
        let zext_val = func.dfg.make_value(Value::Inst {
            inst: zext_inst,
            result_idx: 0,
            ty: phi_ty,
        });
        func.dfg.attach_result(zext_inst, zext_val);
        func.layout.insert_inst_before(zext_inst, insert_before);
        zext_val
    };

    Some(result)
}

/// Find a common predecessor block that ends with a conditional branch
/// targeting the two given blocks (directly or through trivial trampolines).
fn find_common_branch_predecessor(
    func: &Function,
    block_a: BlockId,
    block_b: BlockId,
) -> Option<BlockId> {
    // Simple case: both blocks have exactly one predecessor and it's the same block.
    let pred_a = single_predecessor(func, block_a)?;
    let pred_b = single_predecessor(func, block_b)?;

    if pred_a == pred_b {
        return Some(pred_a);
    }

    None
}

/// Return the single predecessor of a block if it has exactly one.
fn single_predecessor(func: &Function, block: BlockId) -> Option<BlockId> {
    let mut preds = Vec::new();
    for b in func.layout.iter_block() {
        if let Some(last) = func.layout.last_inst_of(b) {
            if let Some(bi) = func.dfg.branch_info(last) {
                for dest in bi.dests() {
                    if dest == block && !preds.contains(&b) {
                        preds.push(b);
                    }
                }
            }
        }
    }
    if preds.len() == 1 {
        Some(preds[0])
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use smallvec::smallvec;
    use sonatina_ir::{
        I256, Linkage, Signature, Type,
        builder::test_util::{test_isa, test_module_builder},
        inst::{
            cast::{Bitcast, IntToPtr, PtrToInt},
            control_flow::Return,
            data::{Gep, Mstore},
            logic,
        },
    };
    use sonatina_parser::parse_module;

    use crate::analysis::known_bits::count_query_news_for_test;

    #[derive(Clone, Copy)]
    struct XorShift64(u64);

    impl XorShift64 {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }

        fn pick<T>(self, values: &[T]) -> usize {
            (self.0 as usize) % values.len()
        }
    }

    #[test]
    fn sccp_smoke_fuzz_no_panics() {
        for seed in 0..8u64 {
            let mb = test_module_builder();
            let sig = Signature::new_single("fuzz", Linkage::Public, &[], Type::I256);
            let func_ref = mb.declare_function(sig).unwrap();

            let mut fb = mb.func_builder::<InstInserter>(func_ref);
            let block0 = fb.append_block();
            fb.switch_to_block(block0);

            let ty = Type::I256;
            let is = fb.func.inst_set();

            let mut values = vec![
                fb.make_imm_value(Immediate::zero(ty)),
                fb.make_imm_value(Immediate::one(ty)),
            ];

            let mut rng = XorShift64(seed.wrapping_add(1));
            for _ in 0..64 {
                let lhs = values[rng.pick(&values)];
                rng.next();
                let rhs = values[rng.pick(&values)];
                let op = rng.next() % 9;

                let value = match op {
                    0 => fb.insert_inst(arith::Add::new_unchecked(is, lhs, rhs), ty),
                    1 => fb.insert_inst(arith::Sub::new_unchecked(is, lhs, rhs), ty),
                    2 => fb.insert_inst(arith::Mul::new_unchecked(is, lhs, rhs), ty),
                    3 => fb.insert_inst(arith::Shl::new_unchecked(is, lhs, rhs), ty),
                    4 => fb.insert_inst(arith::Shr::new_unchecked(is, lhs, rhs), ty),
                    5 => fb.insert_inst(arith::Sar::new_unchecked(is, lhs, rhs), ty),
                    6 => fb.insert_inst(logic::And::new_unchecked(is, lhs, rhs), ty),
                    7 => fb.insert_inst(logic::Or::new_unchecked(is, lhs, rhs), ty),
                    _ => fb.insert_inst(logic::Xor::new_unchecked(is, lhs, rhs), ty),
                };

                values.push(value);
            }

            let ret = *values.last().unwrap();
            fb.insert_inst_no_result(Return::new_unchecked(is, smallvec![ret].into()));
            fb.seal_all();
            fb.finish();

            let mut cfg = ControlFlowGraph::default();
            mb.func_store.modify(func_ref, |func| {
                cfg.compute(func);
                let mut solver = SccpSolver::new();
                solver.run(func, &mut cfg);
                CfgCleanup::new(CleanupMode::Strict).run(func);
            });
        }
    }

    #[test]
    fn sccp_does_not_fold_pointer_constants() {
        let mb = test_module_builder();
        let sig = Signature::new_single("ptr_const", Linkage::Public, &[], Type::I256);
        let func_ref = mb.declare_function(sig).unwrap();

        let mut fb = mb.func_builder::<InstInserter>(func_ref);
        let block0 = fb.append_block();
        fb.switch_to_block(block0);

        let isa = test_isa();
        let is = isa.inst_set();
        let ptr_ty = fb.ptr_type(Type::I8);
        let word = fb.make_imm_value(Immediate::from_i256(I256::from(64u64), Type::I256));
        let ptr = fb.insert_inst(IntToPtr::new(is, word, ptr_ty), ptr_ty);
        let roundtrip = fb.insert_inst(PtrToInt::new(is, ptr, Type::I256), Type::I256);

        fb.insert_inst_no_result(Return::new_unchecked(is, smallvec![roundtrip].into()));
        fb.seal_all();
        fb.finish();

        let mut cfg = ControlFlowGraph::default();
        mb.func_store.modify(func_ref, |func| {
            cfg.compute(func);
            let mut solver = SccpSolver::new();
            solver.run(func, &mut cfg);
            CfgCleanup::new(CleanupMode::Strict).run(func);
        });
    }

    #[test]
    fn sccp_folds_bool_bitcast_with_raw_bits() {
        let mb = test_module_builder();
        let sig = Signature::new_single("bool_bitcast", Linkage::Public, &[], Type::I256);
        let func_ref = mb.declare_function(sig).unwrap();

        let mut fb = mb.func_builder::<InstInserter>(func_ref);
        let block0 = fb.append_block();
        fb.switch_to_block(block0);

        let isa = test_isa();
        let is = isa.inst_set();
        let one = fb.make_imm_value(Immediate::one(Type::I1));
        let widened = fb.insert_inst(Bitcast::new(is, one, Type::I256), Type::I256);
        fb.insert_inst_no_result(Return::new_unchecked(is, smallvec![widened].into()));
        fb.seal_all();
        fb.finish();

        let mut cfg = ControlFlowGraph::default();
        mb.func_store.modify(func_ref, |func| {
            cfg.compute(func);
            let mut solver = SccpSolver::new();
            solver.run(func, &mut cfg);
            CfgCleanup::new(CleanupMode::Strict).run(func);

            let block = func.layout.entry_block().expect("entry block");
            let term = func.layout.last_inst_of(block).expect("return");
            let ret = downcast::<&Return>(func.inst_set(), func.dfg.inst(term)).expect("return");
            let value = ret.args().iter().next().copied().expect("return value");
            assert_eq!(func.dfg.value_imm(value), Some(Immediate::one(Type::I256)));
        });
    }

    #[test]
    fn sccp_reuses_known_bits_queries_across_probes() {
        let parsed = parse_module(
            r#"
target = "evm-ethereum-london"

func public %f(v0.i256) -> i256 {
block0:
    v1.i256 = shr 224.i256 v0;
    v2.i256 = and v1 4294967295.i256;
    v3.i256 = and v2 4294967295.i256;
    return v3;
}
"#,
        )
        .expect("module parses");
        let func_ref = parsed.module.funcs()[0];

        parsed.module.func_store.modify(func_ref, |func| {
            let mut cfg = ControlFlowGraph::default();
            cfg.compute(func);
            let (_, query_news) = count_query_news_for_test(|| {
                let mut solver = SccpSolver::new();
                solver.run(func, &mut cfg);
            });
            assert_eq!(
                query_news, 2,
                "SCCP should build one known-bits snapshot per phase"
            );
        });
    }

    #[test]
    fn sccp_folds_add_zero_with_pointer_operand() {
        let mb = test_module_builder();
        let sig = Signature::new_single("ptr_add_zero", Linkage::Public, &[], Type::I256);
        let func_ref = mb.declare_function(sig).unwrap();

        let mut fb = mb.func_builder::<InstInserter>(func_ref);
        let block0 = fb.append_block();
        fb.switch_to_block(block0);

        let isa = test_isa();
        let is = isa.inst_set();
        let ptr_ty = fb.ptr_type(Type::I8);
        let base_word = fb.make_imm_value(Immediate::from_i256(I256::from(64u64), Type::I256));
        let ptr = fb.insert_inst(IntToPtr::new(is, base_word, ptr_ty), ptr_ty);
        let zero = fb.make_imm_value(Immediate::zero(Type::I256));
        let addr = fb.insert_inst(arith::Add::new_unchecked(is, ptr, zero), Type::I256);
        let one = fb.make_imm_value(Immediate::one(Type::I256));
        fb.insert_inst_no_result(Mstore::new(is, addr, one, Type::I256));
        fb.insert_inst_no_result(Return::new_unchecked(is, smallvec![one].into()));
        fb.seal_all();
        fb.finish();

        let mut cfg = ControlFlowGraph::default();
        mb.func_store.modify(func_ref, |func| {
            cfg.compute(func);
            let mut solver = SccpSolver::new();
            solver.run(func, &mut cfg);
            CfgCleanup::new(CleanupMode::Strict).run(func);

            let mut has_zero_add = false;
            for block in func.layout.iter_block() {
                for inst in func.layout.iter_inst(block) {
                    let Some(add) = downcast::<&arith::Add>(func.inst_set(), func.dfg.inst(inst))
                    else {
                        continue;
                    };
                    let lhs_zero = func
                        .dfg
                        .value_imm(*add.lhs())
                        .is_some_and(Immediate::is_zero);
                    let rhs_zero = func
                        .dfg
                        .value_imm(*add.rhs())
                        .is_some_and(Immediate::is_zero);
                    has_zero_add |= lhs_zero || rhs_zero;
                }
            }

            assert!(!has_zero_add, "SCCP should fold add-with-zero");
        });
    }

    #[test]
    fn sccp_folds_all_zero_gep_chain() {
        let mb = test_module_builder();
        let sig = Signature::new_single("ptr_gep_zero", Linkage::Public, &[], Type::I256);
        let func_ref = mb.declare_function(sig).unwrap();

        let mut fb = mb.func_builder::<InstInserter>(func_ref);
        let block0 = fb.append_block();
        fb.switch_to_block(block0);

        let isa = test_isa();
        let is = isa.inst_set();
        let outer_ty = fb.declare_struct_type("Outer", &[Type::I256, Type::I256], false);
        let ptr_outer_ty = fb.ptr_type(outer_ty);
        let ptr_i256_ty = fb.ptr_type(Type::I256);
        let base_word = fb.make_imm_value(Immediate::from_i256(I256::from(64u64), Type::I256));
        let base_ptr = fb.insert_inst(IntToPtr::new(is, base_word, ptr_outer_ty), ptr_outer_ty);
        let zero = fb.make_imm_value(Immediate::zero(Type::I8));
        let gep = fb.insert_inst(
            Gep::new(
                is.has_gep().expect("inst set has gep"),
                smallvec![base_ptr, zero, zero],
            ),
            ptr_i256_ty,
        );
        let roundtrip = fb.insert_inst(PtrToInt::new(is, gep, Type::I256), Type::I256);
        fb.insert_inst_no_result(Return::new_unchecked(is, smallvec![roundtrip].into()));
        fb.seal_all();
        fb.finish();

        let mut cfg = ControlFlowGraph::default();
        mb.func_store.modify(func_ref, |func| {
            cfg.compute(func);
            let mut solver = SccpSolver::new();
            solver.run(func, &mut cfg);
            CfgCleanup::new(CleanupMode::Strict).run(func);

            let mut has_all_zero_gep = false;
            let mut has_bitcast = false;
            for block in func.layout.iter_block() {
                for inst in func.layout.iter_inst(block) {
                    if let Some(gep) = downcast::<&Gep>(func.inst_set(), func.dfg.inst(inst)) {
                        let values = gep.values();
                        let is_all_zero = values[1..]
                            .iter()
                            .all(|&idx| func.dfg.value_imm(idx).is_some_and(Immediate::is_zero));
                        has_all_zero_gep |= is_all_zero;
                    }
                    if downcast::<&cast::Bitcast>(func.inst_set(), func.dfg.inst(inst)).is_some() {
                        has_bitcast = true;
                    }
                }
            }

            assert!(!has_all_zero_gep, "SCCP should fold all-zero geps");
            assert!(
                has_bitcast,
                "SCCP should preserve typed zero-gep via bitcast"
            );
        });
    }
}
