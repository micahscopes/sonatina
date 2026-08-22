//! Control flow structuring pass for targets that require structured control flow
//! (WASM, SPIR-V). Converts an arbitrary reducible CFG into a nested region tree
//! (Block / Loop / IfThenElse) via a Ramsey-style dominator-tree walk (Norman
//! Ramsey, "Beyond Relooper", ICFP 2022), which the existing DomTree + LoopTree
//! analyses already supply as input.
//!
//! This pass operates on Sonatina IR post-optimization and produces a
//! [`StructuredCfg`] annotation that structured-CF backends consume. It is
//! fail-closed: any shape it cannot classify (irreducible residue, a switch,
//! an ambiguous merge, a loop with no recognizable header exit) returns a named
//! `Err`, never a silently dropped branch.

use std::collections::{HashMap, HashSet};

use bit_set::BitSet;
use sonatina_ir::{
    BlockId, Function, InstDowncast, InstSetBase,
    cfg::ControlFlowGraph,
    inst::control_flow::{Br, BrTable, Jump, Return, Unreachable},
};

use crate::{
    domtree::DomTree,
    loop_analysis::{Loop, LoopTree},
};

/// A structured control flow region.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Region {
    /// A linear block (no nesting).
    Block(BlockId),
    /// A loop with a header and body regions. `body` is the structured form of
    /// the loop's blocks starting at the header's in-loop successor; the header
    /// block itself (phis, exit condition, exit branch) is referenced by
    /// `header` and consumed by the emitter's loop preamble.
    Loop { header: BlockId, body: Vec<Region> },
    /// An edge from a loop body to that loop's canonical fallthrough block.
    /// The source block remains a separate `Block` region when it contains
    /// instructions; this marker preserves the exact predecessor needed for
    /// exit-phi transport before the structured backend emits `break`.
    LoopExit { from: BlockId, target: BlockId },
    /// A conditional edge from a loop body back to that loop's header. Direct
    /// jumps are represented by their source `Block`, but a branch arm needs
    /// an explicit marker so structured backends can emit `continue` and the
    /// exact header-phi transfer without trying to consume the header twice.
    LoopContinue { from: BlockId, target: BlockId },
    /// An if-then-else with a condition (header) block, then-regions,
    /// else-regions, and the join (merge) block, if any. The merge block itself
    /// stays a SIBLING in the enclosing `Vec<Region>`; the arm vectors contain
    /// only arm-dominated regions.
    IfThenElse {
        header: BlockId,
        then_branch: Vec<Region>,
        else_branch: Vec<Region>,
        merge: Option<BlockId>,
    },
}

/// The result of structuring a function's control flow.
#[derive(Debug, Clone)]
pub struct StructuredCfg {
    pub regions: Vec<Region>,
    pub block_order: Vec<BlockId>,
}

/// Dense post-dominator sets for every reachable block. Merge discovery asks
/// the same post-dominance question at every conditional, so computing these
/// sets once avoids rebuilding a full reachability graph and fixed point for
/// each branch.
struct PostDominators {
    blocks: Vec<BlockId>,
    indices: HashMap<BlockId, usize>,
    sets: Vec<BitSet>,
}

impl PostDominators {
    fn compute(cfg: &ControlFlowGraph, blocks: &[BlockId]) -> Self {
        let indices = blocks
            .iter()
            .copied()
            .enumerate()
            .map(|(index, block)| (block, index))
            .collect::<HashMap<_, _>>();
        let successors = blocks
            .iter()
            .map(|block| {
                cfg.succs_of(*block)
                    .filter_map(|successor| indices.get(successor).copied())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let all = (0..blocks.len()).collect::<BitSet>();
        let mut sets = successors
            .iter()
            .enumerate()
            .map(|(index, successors)| {
                if successors.is_empty() {
                    [index].into_iter().collect()
                } else {
                    all.clone()
                }
            })
            .collect::<Vec<_>>();

        loop {
            let mut changed = false;
            for (index, successors) in successors.iter().enumerate() {
                let Some((&first, rest)) = successors.split_first() else {
                    continue;
                };
                let mut intersection = sets[first].clone();
                for successor in rest {
                    intersection.intersect_with(&sets[*successor]);
                }
                intersection.insert(index);
                if intersection != sets[index] {
                    sets[index] = intersection;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        Self {
            blocks: blocks.to_vec(),
            indices,
            sets,
        }
    }
}

/// Compute structured control flow for a function.
///
/// Requires a reducible CFG (which Fe always produces). Returns an error
/// for irreducible control flow or any shape outside the supported closure.
pub fn structurize_function(function: &Function) -> Result<StructuredCfg, String> {
    let mut cfg = ControlFlowGraph::default();
    cfg.compute(function);
    let mut domtree = DomTree::new();
    domtree.compute(&cfg);

    let mut loop_tree = LoopTree::new();
    loop_tree.compute(&cfg, &domtree);

    let rpo = domtree.rpo().to_vec();

    if rpo.is_empty() {
        return Ok(StructuredCfg {
            regions: Vec::new(),
            block_order: Vec::new(),
        });
    }

    let postdom = PostDominators::compute(&cfg, &rpo);
    let s = Structurer {
        function,
        is: function.inst_set(),
        loop_tree: &loop_tree,
        postdom: &postdom,
    };
    let mut active = HashSet::new();
    let mut consumed = HashSet::new();
    let regions = s.build_seq(rpo[0], None, None, &mut active, &mut consumed)?;

    let missing = rpo
        .iter()
        .copied()
        .filter(|block| !consumed.contains(block))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(format!(
            "spirv structurize: reachable blocks were not consumed: {missing:?}"
        ));
    }

    Ok(StructuredCfg {
        block_order: rpo,
        regions,
    })
}

/// A block's classified terminator.
enum Term {
    Jump(BlockId),
    Br(BlockId, BlockId),
    Return,
    /// A block terminated by Sonatina `Unreachable` (the array/memory bounds
    /// trap `wasm_lower.rs` emits for every dynamically-indexed access).
    /// Classified Return-like throughout this pass (excluded from loop SCC
    /// membership, treated as a chain terminator) so a trap arm structures
    /// exactly the way an early-return arm does. Kept as its own variant
    /// (rather than literally reusing `Term::Return`) so the SPIR-V emitter
    /// can tell "real return value" apart from "poison" by re-inspecting the
    /// block's own instructions.
    ///
    /// Guards Codex NO-GO bug 4 (wrong value on unconditional trap): without
    /// this arm, `structurize_function` hard-errors ("unsupported
    /// terminator") on ANY function containing a bounds trap, i.e. every
    /// function with a dynamically-indexed array access, so no array kernel
    /// could reach the SPIR-V emitter at all.
    Unreachable,
    Other,
}

struct Structurer<'a> {
    function: &'a Function,
    is: &'a dyn InstSetBase,
    loop_tree: &'a LoopTree,
    postdom: &'a PostDominators,
}

impl Structurer<'_> {
    fn term(&self, b: BlockId) -> Term {
        for inst_id in self.function.layout.iter_inst(b) {
            let d = self.function.dfg.inst(inst_id);
            if let Some(j) = <&Jump as InstDowncast>::downcast(self.is, d) {
                return Term::Jump(*j.dest());
            }
            if let Some(br) = <&Br as InstDowncast>::downcast(self.is, d) {
                return Term::Br(*br.nz_dest(), *br.z_dest());
            }
            if <&Return as InstDowncast>::downcast(self.is, d).is_some() {
                return Term::Return;
            }
            if <&Unreachable as InstDowncast>::downcast(self.is, d).is_some() {
                return Term::Unreachable;
            }
            if <&BrTable as InstDowncast>::downcast(self.is, d).is_some() {
                return Term::Other;
            }
        }
        Term::Other
    }

    fn returns(&self, b: BlockId) -> bool {
        matches!(self.term(b), Term::Return | Term::Unreachable)
    }

    /// Whether every path from `start` stays outside `lp` and terminates in a
    /// Return. Loop analysis deliberately excludes these blocks from the loop
    /// SCC, but structurally they belong to the early-return arm, not to the
    /// loop's canonical break/fallthrough exit.
    fn is_return_corridor(&self, start: BlockId, lp: Loop) -> bool {
        fn visit(
            s: &Structurer<'_>,
            block: BlockId,
            lp: Loop,
            visiting: &mut HashSet<BlockId>,
            memo: &mut HashMap<BlockId, bool>,
        ) -> bool {
            if let Some(&result) = memo.get(&block) {
                return result;
            }
            if s.in_loop(block, lp) || s.is_canonical_loop_exit(lp, block) {
                return false;
            }
            if !visiting.insert(block) {
                return false;
            }
            let result = match s.term(block) {
                Term::Return | Term::Unreachable => true,
                Term::Jump(target) => visit(s, target, lp, visiting, memo),
                Term::Br(nz, z) => {
                    visit(s, nz, lp, visiting, memo) && visit(s, z, lp, visiting, memo)
                }
                Term::Other => false,
            };
            visiting.remove(&block);
            memo.insert(block, result);
            result
        }

        visit(self, start, lp, &mut HashSet::new(), &mut HashMap::new())
    }

    fn in_loop(&self, b: BlockId, lp: Loop) -> bool {
        self.loop_tree.is_in_loop(b, lp)
    }

    /// A block that is ALWAYS a CFG dead end: its only instruction is
    /// `Unreachable` (no other side effects, no successors, no phi inputs to
    /// preserve). `wasm_lower.rs::trap_block` creates and caches exactly ONE
    /// such block per function and reuses it for EVERY dynamic-index bounds
    /// check and checked-usize-overflow check, so it legitimately has many
    /// predecessors spread across the whole function -- not the "one true
    /// position in the region tree" every other block has. Referencing it
    /// from more than one arm is therefore not a real "multiply owned"
    /// block; the general active/consumed cycle guard in `build_seq` would
    /// otherwise reject the SECOND (and every later) bounds check in any
    /// function with more than one dynamically-indexed access, once
    /// `Unreachable` stops being an unconditional hard error.
    fn is_shared_trap_block(&self, block: BlockId) -> bool {
        matches!(self.term(block), Term::Unreachable)
            && self.function.layout.iter_inst(block).count() == 1
    }

    /// Build the region sequence for the maximal single-entry area entered at
    /// `start`, following structured successors, stopping before `stop` and at
    /// the boundaries of `cur_loop` (its header on a backedge, or a fallthrough
    /// exit that the caller resumes at). Return blocks reached from inside the
    /// loop are INCLUDED here (they become early-return arms).
    fn build_seq(
        &self,
        start: BlockId,
        stop: Option<BlockId>,
        cur_loop: Option<Loop>,
        active: &mut HashSet<BlockId>,
        consumed: &mut HashSet<BlockId>,
    ) -> Result<Vec<Region>, String> {
        let mut regions = Vec::new();
        let mut cur = Some(start);
        let allow_return_corridor = cur_loop
            .is_some_and(|lp| !self.in_loop(start, lp) && self.is_return_corridor(start, lp));

        while let Some(b) = cur {
            if Some(b) == stop {
                break;
            }
            if let Some(lp) = cur_loop {
                // Backedge to our own header ends this chain (a continue edge).
                if b == self.loop_tree.loop_header(lp) && b != start {
                    break;
                }
                // A fallthrough exit (a non-return block outside the loop) is
                // resumed by the caller, not structured inside the loop.
                if !self.in_loop(b, lp) && b != start && !self.returns(b) && !allow_return_corridor
                {
                    break;
                }
            }

            // A shared trap block dead-ends here regardless of how many
            // other predecessors also reach it elsewhere in the function; see
            // `is_shared_trap_block`. `consumed.insert` is idempotent (a
            // HashSet re-insert is a harmless no-op), so this is safe to hit
            // on the first occurrence AND every later one.
            if self.is_shared_trap_block(b) {
                consumed.insert(b);
                regions.push(Region::Block(b));
                cur = None;
                continue;
            }

            if active.contains(&b) || !consumed.insert(b) {
                return Err(format!(
                    "spirv structurize: cyclic or multiply consumed block {b:?} while building \
                     sequence {start:?}..{stop:?}; active={active:?}"
                ));
            }
            active.insert(b);

            // Open a loop when we first reach a header we are not already in.
            if let Some(lp) = self.loop_tree.loop_of_block(b) {
                if self.loop_tree.loop_header(lp) == b && cur_loop != Some(lp) {
                    let body = self.build_loop_body(b, lp, active, consumed)?;
                    regions.push(Region::Loop { header: b, body });
                    if let Some(exit) = self.loop_direct_exit(b, lp) {
                        if self.returns(exit) {
                            consumed.insert(exit);
                        }
                    }
                    cur = self.loop_fallthrough(b, lp)?;
                    active.remove(&b);
                    continue;
                }
            }

            match self.term(b) {
                Term::Jump(t) => {
                    regions.push(Region::Block(b));
                    if let Some(lp) = cur_loop
                        && !self.in_loop(t, lp)
                        && (self.is_canonical_loop_exit(lp, t) || !self.returns(t))
                        && !self.is_return_corridor(t, lp)
                    {
                        self.validate_loop_exit_target(lp, b, t)?;
                        regions.push(Region::LoopExit { from: b, target: t });
                        cur = None;
                    } else {
                        cur = Some(t);
                    }
                }
                Term::Return | Term::Unreachable => {
                    regions.push(Region::Block(b));
                    cur = None;
                }
                Term::Br(nz, z) => {
                    // Reachability-based merge discovery includes an enclosing
                    // arm's stop whenever both successors really converge
                    // there, without mistaking an unrelated outer join for
                    // this conditional's merge.
                    let merge = self.find_merge(b, cur_loop, stop)?;
                    let then_branch =
                        self.build_branch(b, nz, merge, cur_loop, active, consumed)?;
                    let else_branch = self.build_branch(b, z, merge, cur_loop, active, consumed)?;
                    regions.push(Region::IfThenElse {
                        header: b,
                        then_branch,
                        else_branch,
                        merge,
                    });
                    cur = merge;
                }
                Term::Other => {
                    return Err(format!(
                        "spirv structurize: block {b:?} has an unsupported terminator \
                         (switch/BrTable or missing terminator); this push handles \
                         Jump/Br/Return only"
                    ));
                }
            }
            active.remove(&b);
        }

        Ok(regions)
    }

    fn build_branch(
        &self,
        from: BlockId,
        target: BlockId,
        merge: Option<BlockId>,
        cur_loop: Option<Loop>,
        active: &mut HashSet<BlockId>,
        consumed: &mut HashSet<BlockId>,
    ) -> Result<Vec<Region>, String> {
        if Some(target) == merge {
            return Ok(Vec::new());
        }
        if let Some(lp) = cur_loop
            && target == self.loop_tree.loop_header(lp)
        {
            return Ok(vec![Region::LoopContinue { from, target }]);
        }
        if let Some(merge) = merge
            && self.is_transparent_forwarder(target, merge)
        {
            // Empty source-level arms can share the same forwarding block.
            // Duplicate that semantically empty region in each structured arm
            // so its merge-edge phi transport is preserved without granting
            // general multiply-owned blocks.
            consumed.insert(target);
            return Ok(vec![Region::Block(target)]);
        }
        if let Some(lp) = cur_loop
            && !self.in_loop(target, lp)
            && (self.is_canonical_loop_exit(lp, target) || !self.returns(target))
            && !self.is_return_corridor(target, lp)
        {
            if !self.is_canonical_loop_exit(lp, target)
                && let Term::Jump(exit) = self.term(target)
                && self.is_canonical_loop_exit(lp, exit)
            {
                if !consumed.insert(target) {
                    return Err(format!(
                        "spirv structurize: cyclic or multiply consumed block {target:?}"
                    ));
                }
                return Ok(vec![
                    Region::Block(target),
                    Region::LoopExit {
                        from: target,
                        target: exit,
                    },
                ]);
            }
            self.validate_loop_exit_target(lp, from, target)?;
            return Ok(vec![Region::LoopExit { from, target }]);
        }
        self.build_seq(target, merge, cur_loop, active, consumed)
    }

    fn is_canonical_loop_exit(&self, lp: Loop, target: BlockId) -> bool {
        let header = self.loop_tree.loop_header(lp);
        self.loop_direct_exit(header, lp) == Some(target)
    }

    fn is_transparent_forwarder(&self, block: BlockId, target: BlockId) -> bool {
        matches!(self.term(block), Term::Jump(destination) if destination == target)
            && self.function.layout.iter_inst(block).count() == 1
    }

    fn validate_loop_exit_target(
        &self,
        lp: Loop,
        from: BlockId,
        target: BlockId,
    ) -> Result<(), String> {
        let header = self.loop_tree.loop_header(lp);
        let canonical = self.loop_direct_exit(header, lp).ok_or_else(|| {
            format!("spirv structurize: loop {header:?} has no canonical header exit")
        })?;
        if target != canonical {
            return Err(format!(
                "spirv structurize: loop body edge {from:?}->{target:?} targets a \
                 noncanonical exit; expected the header exit {canonical:?}"
            ));
        }
        Ok(())
    }

    /// Structure a loop's body: the region sequence entered at the header's
    /// in-loop successor, stopping at the header (backedge) and at loop exits.
    fn build_loop_body(
        &self,
        header: BlockId,
        lp: Loop,
        active: &mut HashSet<BlockId>,
        consumed: &mut HashSet<BlockId>,
    ) -> Result<Vec<Region>, String> {
        match self.term(header) {
            Term::Br(nz, z) => {
                let nz_in = self.in_loop(nz, lp);
                let z_in = self.in_loop(z, lp);
                let entry = match (nz_in, z_in) {
                    (true, false) => nz,
                    (false, true) => z,
                    (true, true) => {
                        return Err(format!(
                            "spirv structurize: loop header {header:?} branches to two \
                             in-loop blocks (mid-loop-only exit); unsupported in this push"
                        ));
                    }
                    (false, false) => {
                        return Err(format!(
                            "spirv structurize: loop header {header:?} has no in-loop \
                             successor"
                        ));
                    }
                };
                if entry == header {
                    // A header-only loop branches straight back to itself. The
                    // header owns the condition and backedge, so there are no
                    // separate body regions to recurse into.
                    Ok(Vec::new())
                } else {
                    self.build_seq(entry, None, Some(lp), active, consumed)
                }
            }
            Term::Jump(t) => self.build_seq(t, None, Some(lp), active, consumed),
            _ => Err(format!(
                "spirv structurize: loop header {header:?} must end in Jump or Br"
            )),
        }
    }

    fn loop_direct_exit(&self, header: BlockId, lp: Loop) -> Option<BlockId> {
        match self.term(header) {
            Term::Br(nz, z) if self.in_loop(nz, lp) && !self.in_loop(z, lp) => Some(z),
            Term::Br(nz, z) if !self.in_loop(nz, lp) && self.in_loop(z, lp) => Some(nz),
            _ => None,
        }
    }

    /// The block execution resumes at after the loop: the header's out-of-loop
    /// exit target if it continues (non-return); `None` if the exit returns
    /// (the emitter funnels that return value out of the loop).
    fn loop_fallthrough(&self, header: BlockId, lp: Loop) -> Result<Option<BlockId>, String> {
        match self.term(header) {
            Term::Br(nz, z) => {
                let exit = if self.in_loop(nz, lp) { z } else { nz };
                if self.returns(exit) {
                    Ok(None)
                } else {
                    Ok(Some(exit))
                }
            }
            Term::Jump(_) => Ok(None),
            _ => Ok(None),
        }
    }

    /// The merge (join) block for a two-way branch at `header`: the unique
    /// dominator-tree child of `header` that is a join (>= 2 preds), excluding
    /// the enclosing loop header (a backedge join is not an if-merge). `None`
    /// means the arms diverge (each returns/continues/breaks) with no join.
    fn find_merge(
        &self,
        header: BlockId,
        cur_loop: Option<Loop>,
        enclosing_stop: Option<BlockId>,
    ) -> Result<Option<BlockId>, String> {
        let loop_hdr = cur_loop.map(|lp| self.loop_tree.loop_header(lp));
        let Term::Br(nz, z) = self.term(header) else {
            return Ok(None);
        };
        let header_index = self.postdom.indices[&header];
        let strict = self.postdom.sets[header_index]
            .iter()
            .map(|index| self.postdom.blocks[index])
            .filter(|candidate| *candidate != header && Some(*candidate) != loop_hdr)
            .collect::<Vec<_>>();
        let immediate = strict
            .iter()
            .copied()
            .filter(|candidate| {
                let candidate_index = self.postdom.indices[candidate];
                strict.iter().all(|other| {
                    candidate == other
                        || self.postdom.sets[candidate_index].contains(self.postdom.indices[other])
                })
            })
            .collect::<Vec<_>>();
        match immediate.as_slice() {
            [] => {
                // A bounds-checked arm may either trap or continue at the
                // other successor. The continuation is not a strict
                // post-dominator because the trap is terminal, but it is
                // still the structured merge for every nonterminal path.
                // When that continuation is already the enclosing arm's
                // stop, retain it explicitly. Otherwise the continuing arm
                // consumes the stop and its parent consumes it a second time.
                if Some(z) == enclosing_stop && self.returns(nz) {
                    return Ok(Some(z));
                }
                if Some(nz) == enclosing_stop && self.returns(z) {
                    return Ok(Some(nz));
                }
                // Search only within the current loop iteration so a later
                // backedge cannot make the two successors appear mutually
                // reachable.
                let nz_reaches_z = self.reaches_before_loop_header(nz, z, cur_loop);
                let z_reaches_nz = self.reaches_before_loop_header(z, nz, cur_loop);
                match (nz_reaches_z, z_reaches_nz) {
                    (true, false) => {
                        if self.returns(z)
                            && let Some(stop) = enclosing_stop
                            && !self.returns(stop)
                            && self.reaches_before_loop_header(nz, stop, cur_loop)
                        {
                            return Ok(Some(stop));
                        }
                        Ok(Some(z))
                    }
                    (false, true) => {
                        if self.returns(nz)
                            && let Some(stop) = enclosing_stop
                            && !self.returns(stop)
                            && self.reaches_before_loop_header(z, stop, cur_loop)
                        {
                            return Ok(Some(stop));
                        }
                        Ok(Some(nz))
                    }
                    _ => {
                        if let Some(stop) = enclosing_stop
                            && self.reaches_before_loop_header(nz, stop, cur_loop)
                            && self.reaches_before_loop_header(z, stop, cur_loop)
                        {
                            return Ok(Some(stop));
                        }
                        Ok(None)
                    }
                }
            }
            [merge] => Ok(Some(*merge)),
            _ => Err(format!(
                "spirv structurize: {} immediate postdominator candidates for block {header:?} \
                 (irreducible/unsupported control-flow shape)",
                immediate.len()
            )),
        }
    }

    /// Whether `target` is reachable from `start` without completing the
    /// enclosing loop iteration. Terminal return and trap paths simply stop.
    /// This recognizes one-sided continuations without treating a later loop
    /// iteration as evidence that two branch successors merge.
    fn reaches_before_loop_header(
        &self,
        start: BlockId,
        target: BlockId,
        cur_loop: Option<Loop>,
    ) -> bool {
        let loop_header = cur_loop.map(|lp| self.loop_tree.loop_header(lp));
        let mut pending = vec![start];
        let mut visited = HashSet::new();
        while let Some(block) = pending.pop() {
            if block == target {
                return true;
            }
            if Some(block) == loop_header && block != start {
                continue;
            }
            if !visited.insert(block) {
                continue;
            }
            match self.term(block) {
                Term::Jump(next) => pending.push(next),
                Term::Br(nz, z) => {
                    pending.push(nz);
                    pending.push(z);
                }
                Term::Return | Term::Unreachable | Term::Other => {}
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonatina_ir::{
        Linkage, Module, Signature, Type,
        builder::ModuleBuilder,
        func_cursor::InstInserter,
        inst::{
            arith, cmp,
            control_flow::{Br, Jump, Phi, Return, Unreachable},
        },
        isa::{Isa, native::Native},
        module::ModuleCtx,
    };
    use sonatina_triple::{Architecture, OperatingSystem, TargetTriple, Vendor};

    fn native_builder() -> (ModuleBuilder, &'static dyn sonatina_ir::InstSetBase) {
        let isa = Native::new(TargetTriple::new(
            Architecture::X86_64,
            Vendor::Unknown,
            OperatingSystem::Native,
        ));
        let is = isa.inst_set();
        let ctx = ModuleCtx::new(&isa);
        (ModuleBuilder::new(ctx), is)
    }

    fn structurize(module: &Module, fr: sonatina_ir::module::FuncRef) -> StructuredCfg {
        module
            .func_store
            .view(fr, |func| structurize_function(func))
            .unwrap()
    }

    #[test]
    fn structurize_linear_cfg() {
        let (mb, is) = native_builder();
        let sig = Signature::new_unit("linear", Linkage::Public, &[]);
        let func_ref = mb.declare_function(sig).unwrap();
        let mut fb = mb.func_builder::<InstInserter>(func_ref);

        let b0 = fb.append_block();
        let b1 = fb.append_block();
        fb.switch_to_block(b0);
        fb.insert_inst_no_result(Jump::new(is, b1));
        fb.switch_to_block(b1);
        fb.insert_inst_no_result(Return::new_unit(is));
        fb.seal_all();
        fb.finish();

        let module = mb.build();
        let structured = structurize(&module, func_ref);
        assert_eq!(structured.regions.len(), 2);
        assert!(matches!(structured.regions[0], Region::Block(_)));
        assert!(matches!(structured.regions[1], Region::Block(_)));
    }

    #[test]
    fn structurize_simple_loop() {
        let (mb, is) = native_builder();
        let sig = Signature::new_unit("loopy", Linkage::Public, &[Type::I32]);
        let func_ref = mb.declare_function(sig).unwrap();
        let mut fb = mb.func_builder::<InstInserter>(func_ref);

        let entry = fb.append_block();
        let loop_header = fb.append_block();
        let exit = fb.append_block();

        fb.switch_to_block(entry);
        fb.insert_inst_no_result(Jump::new(is, loop_header));

        fb.switch_to_block(loop_header);
        let cond = fb.args()[0];
        fb.insert_inst_no_result(Br::new(is, cond, exit, loop_header));

        fb.switch_to_block(exit);
        fb.insert_inst_no_result(Return::new_unit(is));

        fb.seal_all();
        fb.finish();

        let module = mb.build();
        let structured = structurize(&module, func_ref);

        let has_loop = structured
            .regions
            .iter()
            .any(|r| matches!(r, Region::Loop { .. }));
        assert!(
            has_loop,
            "expected a loop region in {:?}",
            structured.regions
        );
    }

    /// A top-level if/else diamond structurizes into a single `IfThenElse` with
    /// one region per arm and the merge as a following sibling `Block`.
    #[test]
    fn structurize_diamond() {
        let (mb, is) = native_builder();
        let sig = Signature::new_single(
            "diamond",
            Linkage::Public,
            &[Type::I32, Type::I32],
            Type::I32,
        );
        let fr = mb.declare_function(sig).unwrap();
        let mut fb = mb.func_builder::<InstInserter>(fr);
        let entry = fb.append_block();
        let t = fb.append_block();
        let e = fb.append_block();
        let m = fb.append_block();

        fb.switch_to_block(entry);
        let a0 = fb.args()[0];
        let a1 = fb.args()[1];
        let c = fb.insert_inst(cmp::Lt::new(is, a0, a1), Type::I1);
        fb.insert_inst_no_result(Br::new(is, c, t, e));
        fb.switch_to_block(t);
        let x = fb.insert_inst(arith::Add::new(is, a0, a1), Type::I32);
        fb.insert_inst_no_result(Jump::new(is, m));
        fb.switch_to_block(e);
        let y = fb.insert_inst(arith::Sub::new(is, a0, a1), Type::I32);
        fb.insert_inst_no_result(Jump::new(is, m));
        fb.switch_to_block(m);
        let r = fb.insert_inst(Phi::new(is, vec![(x, t), (y, e)]), Type::I32);
        fb.insert_inst_no_result(Return::new_single(is, r));
        fb.seal_all();
        fb.finish();

        let module = mb.build();
        let s = structurize(&module, fr);
        // [IfThenElse{ then:[Block t], else:[Block e], merge:Some(m) }, Block m]
        assert_eq!(s.regions.len(), 2, "regions: {:?}", s.regions);
        match &s.regions[0] {
            Region::IfThenElse {
                then_branch,
                else_branch,
                merge,
                ..
            } => {
                assert_eq!(then_branch.len(), 1);
                assert_eq!(else_branch.len(), 1);
                assert!(merge.is_some());
            }
            other => panic!("expected IfThenElse, got {other:?}"),
        }
        assert!(matches!(s.regions[1], Region::Block(_)));
    }

    /// One outer arm contains a nested diamond while the sibling enters the
    /// nested diamond's convergence directly. The shared block is the outer
    /// merge; it must not be consumed inside the first arm and then visited a
    /// second time from the sibling.
    #[test]
    fn sibling_start_at_nested_convergence_uses_shared_merge() {
        let (mb, is) = native_builder();
        let sig = Signature::new_unit("shared_merge", Linkage::Public, &[Type::I1, Type::I1]);
        let fr = mb.declare_function(sig).unwrap();
        let mut fb = mb.func_builder::<InstInserter>(fr);
        let entry = fb.append_block();
        let nested = fb.append_block();
        let nested_then = fb.append_block();
        let nested_else = fb.append_block();
        let shared = fb.append_block();
        let exit = fb.append_block();

        fb.switch_to_block(entry);
        fb.insert_inst_no_result(Br::new(is, fb.args()[0], nested, shared));
        fb.switch_to_block(nested);
        fb.insert_inst_no_result(Br::new(is, fb.args()[1], nested_then, nested_else));
        fb.switch_to_block(nested_then);
        fb.insert_inst_no_result(Jump::new(is, shared));
        fb.switch_to_block(nested_else);
        fb.insert_inst_no_result(Jump::new(is, shared));
        fb.switch_to_block(shared);
        fb.insert_inst_no_result(Jump::new(is, exit));
        fb.switch_to_block(exit);
        fb.insert_inst_no_result(Return::new_unit(is));
        fb.seal_all();
        fb.finish();

        let module = mb.build();
        let structured = structurize(&module, fr);
        match &structured.regions[0] {
            Region::IfThenElse { merge, .. } => assert_eq!(*merge, Some(shared)),
            other => panic!("expected outer IfThenElse, got {other:?}"),
        }
        assert!(matches!(structured.regions[1], Region::Block(block) if block == shared));
    }

    /// Two nested selections may share the same empty else-arm forwarding
    /// block while their actual postdominating merge is one block later.
    #[test]
    fn nested_and_outer_arms_may_share_transparent_forwarder() {
        let (mb, is) = native_builder();
        let sig = Signature::new_unit("shared_forwarder", Linkage::Public, &[Type::I1, Type::I1]);
        let fr = mb.declare_function(sig).unwrap();
        let mut fb = mb.func_builder::<InstInserter>(fr);
        let outer = fb.append_block();
        let nested = fb.append_block();
        let body = fb.append_block();
        let shared_forwarder = fb.append_block();
        let merge = fb.append_block();

        fb.switch_to_block(outer);
        fb.insert_inst_no_result(Br::new(is, fb.args()[0], nested, shared_forwarder));
        fb.switch_to_block(nested);
        fb.insert_inst_no_result(Br::new(is, fb.args()[1], body, shared_forwarder));
        fb.switch_to_block(body);
        fb.insert_inst_no_result(Jump::new(is, merge));
        fb.switch_to_block(shared_forwarder);
        fb.insert_inst_no_result(Jump::new(is, merge));
        fb.switch_to_block(merge);
        fb.insert_inst_no_result(Return::new_unit(is));
        fb.seal_all();
        fb.finish();

        let module = mb.build();
        let structured = structurize(&module, fr);
        match &structured.regions[0] {
            Region::IfThenElse {
                then_branch,
                else_branch,
                merge: selected,
                ..
            } => {
                assert_eq!(*selected, Some(merge));
                assert!(
                    matches!(else_branch.as_slice(), [Region::Block(block)] if *block == shared_forwarder)
                );
                assert!(matches!(
                    then_branch.as_slice(),
                    [Region::IfThenElse { merge: Some(inner_merge), .. }]
                        if *inner_merge == merge
                ));
            }
            other => panic!("expected outer IfThenElse, got {other:?}"),
        }
    }

    /// An if-WITHOUT-else (triangle) structurizes with an empty arm.
    #[test]
    fn structurize_triangle() {
        let (mb, is) = native_builder();
        let sig = Signature::new_single(
            "triangle",
            Linkage::Public,
            &[Type::I32, Type::I32],
            Type::I32,
        );
        let fr = mb.declare_function(sig).unwrap();
        let mut fb = mb.func_builder::<InstInserter>(fr);
        let entry = fb.append_block();
        let t = fb.append_block();
        let m = fb.append_block();

        fb.switch_to_block(entry);
        let a0 = fb.args()[0];
        let a1 = fb.args()[1];
        let c = fb.insert_inst(cmp::Lt::new(is, a0, a1), Type::I1);
        fb.insert_inst_no_result(Br::new(is, c, t, m));
        fb.switch_to_block(t);
        let x = fb.insert_inst(arith::Add::new(is, a0, a1), Type::I32);
        fb.insert_inst_no_result(Jump::new(is, m));
        fb.switch_to_block(m);
        let r = fb.insert_inst(Phi::new(is, vec![(a0, entry), (x, t)]), Type::I32);
        fb.insert_inst_no_result(Return::new_single(is, r));
        fb.seal_all();
        fb.finish();

        let module = mb.build();
        let s = structurize(&module, fr);
        match &s.regions[0] {
            Region::IfThenElse {
                then_branch,
                else_branch,
                merge,
                ..
            } => {
                assert_eq!(then_branch.len(), 1, "then arm should hold block t");
                assert_eq!(else_branch.len(), 0, "empty else (z_dest == merge)");
                assert!(merge.is_some());
            }
            other => panic!("expected IfThenElse, got {other:?}"),
        }
    }

    /// An if INSIDE a loop body structurizes as a Loop whose body holds an
    /// IfThenElse (the mandelbrot-escape shape: continue arm + early-return arm).
    #[test]
    fn structurize_if_in_loop() {
        let (mb, is) = native_builder();
        let sig = Signature::new_single("if_in_loop", Linkage::Public, &[Type::I32], Type::I32);
        let fr = mb.declare_function(sig).unwrap();
        let mut fb = mb.func_builder::<InstInserter>(fr);
        let entry = fb.append_block();
        let lh = fb.append_block();
        let lb = fb.append_block();
        let cont = fb.append_block();
        let esc = fb.append_block();
        let exit = fb.append_block();

        fb.switch_to_block(entry);
        let n = fb.args()[0];
        let zero = fb.make_imm_value(0i32);
        fb.insert_inst_no_result(Jump::new(is, lh));
        fb.switch_to_block(lh);
        let i = fb.insert_inst(Phi::new(is, vec![(zero, entry)]), Type::I32);
        let max = fb.make_imm_value(50i32);
        let c = fb.insert_inst(cmp::Lt::new(is, i, max), Type::I1);
        fb.insert_inst_no_result(Br::new(is, c, lb, exit));
        fb.switch_to_block(lb);
        let ec = fb.insert_inst(cmp::Lt::new(is, i, n), Type::I1);
        fb.insert_inst_no_result(Br::new(is, ec, cont, esc));
        fb.switch_to_block(cont);
        let one = fb.make_imm_value(1i32);
        let i2 = fb.insert_inst(arith::Add::new(is, i, one), Type::I32);
        fb.append_phi_arg(i, i2, cont);
        fb.insert_inst_no_result(Jump::new(is, lh));
        fb.switch_to_block(esc);
        fb.insert_inst_no_result(Return::new_single(is, i));
        fb.switch_to_block(exit);
        fb.insert_inst_no_result(Return::new_single(is, max));
        fb.seal_all();
        fb.finish();

        let module = mb.build();
        let s = structurize(&module, fr);
        // Top level: Block(entry), Loop{lh, body:[IfThenElse{lb,...}]}
        let loop_region = s
            .regions
            .iter()
            .find_map(|r| match r {
                Region::Loop { header, body } => Some((header, body)),
                _ => None,
            })
            .expect("expected a Loop region");
        assert!(
            loop_region
                .1
                .iter()
                .any(|r| matches!(r, Region::IfThenElse { .. })),
            "loop body should contain an IfThenElse, got {:?}",
            loop_region.1
        );
    }

    /// A loop-latch branch may either continue or enter a terminal trap arm.
    /// The header is already owned by the enclosing `Loop`, so the continue
    /// edge must be a marker rather than a second recursive region sequence.
    #[test]
    fn structurize_conditional_loop_continue_with_trap() {
        let (mb, is) = native_builder();
        let sig = Signature::new_unit("conditional_continue", Linkage::Public, &[Type::I1]);
        let fr = mb.declare_function(sig).unwrap();
        let mut fb = mb.func_builder::<InstInserter>(fr);
        let entry = fb.append_block();
        let header = fb.append_block();
        let latch = fb.append_block();
        let trap = fb.append_block();
        let exit = fb.append_block();

        fb.switch_to_block(entry);
        let cond = fb.args()[0];
        fb.insert_inst_no_result(Jump::new(is, header));
        fb.switch_to_block(header);
        fb.insert_inst_no_result(Br::new(is, cond, latch, exit));
        fb.switch_to_block(latch);
        fb.insert_inst_no_result(Br::new(is, cond, trap, header));
        fb.switch_to_block(trap);
        fb.insert_inst_no_result(Unreachable::new(is));
        fb.switch_to_block(exit);
        fb.insert_inst_no_result(Return::new_unit(is));
        fb.seal_all();
        fb.finish();

        let module = mb.build();
        let structured = structurize(&module, fr);
        let loop_body = structured
            .regions
            .iter()
            .find_map(|region| match region {
                Region::Loop { body, .. } => Some(body),
                _ => None,
            })
            .expect("expected loop region");
        assert!(loop_body.iter().any(|region| matches!(
            region,
            Region::IfThenElse { then_branch, else_branch, .. }
                if then_branch.iter().chain(else_branch).any(|arm| matches!(
                    arm,
                    Region::LoopContinue { from, target }
                        if *from == latch && *target == header
                ))
        )));
    }

    /// A nested guard inside a loop may trap or fall through to the other arm's
    /// direct successor. The fallthrough is the merge for every nonterminal
    /// path even though the terminal trap prevents strict post-dominance.
    #[test]
    fn structurize_loop_guard_with_trap_and_one_sided_continuation() {
        let (mb, is) = native_builder();
        let sig = Signature::new_unit("guarded_loop", Linkage::Public, &[Type::I1]);
        let fr = mb.declare_function(sig).unwrap();
        let mut fb = mb.func_builder::<InstInserter>(fr);
        let entry = fb.append_block();
        let header = fb.append_block();
        let branch = fb.append_block();
        let guarded = fb.append_block();
        let merge = fb.append_block();
        let trap = fb.append_block();
        let exit = fb.append_block();

        fb.switch_to_block(entry);
        let cond = fb.args()[0];
        fb.insert_inst_no_result(Jump::new(is, header));
        fb.switch_to_block(header);
        fb.insert_inst_no_result(Br::new(is, cond, branch, exit));
        fb.switch_to_block(branch);
        fb.insert_inst_no_result(Br::new(is, cond, guarded, merge));
        fb.switch_to_block(guarded);
        fb.insert_inst_no_result(Br::new(is, cond, trap, merge));
        fb.switch_to_block(merge);
        fb.insert_inst_no_result(Jump::new(is, header));
        fb.switch_to_block(trap);
        fb.insert_inst_no_result(Unreachable::new(is));
        fb.switch_to_block(exit);
        fb.insert_inst_no_result(Return::new_unit(is));
        fb.seal_all();
        fb.finish();

        let module = mb.build();
        let structured = structurize(&module, fr);
        let loop_body = structured
            .regions
            .iter()
            .find_map(|region| match region {
                Region::Loop { body, .. } => Some(body),
                _ => None,
            })
            .expect("expected loop region");
        let outer_branch = loop_body
            .iter()
            .find_map(|region| match region {
                Region::IfThenElse {
                    then_branch,
                    merge: Some(found),
                    ..
                } if *found == merge => Some(then_branch),
                _ => None,
            })
            .expect("expected one-sided continuation merge");
        assert!(outer_branch.iter().any(|region| matches!(
            region,
            Region::IfThenElse { merge: Some(found), .. } if *found == merge
        )));
    }

    /// A nested checked branch can have a closer loop latch even when both the
    /// latch and an enclosing guard may reach the same terminal trap. The
    /// latch is the inner merge; selecting the enclosing trap consumes that
    /// latch once per arm and produces an invalid multiply-owned region.
    #[test]
    fn structurize_prefers_inner_latch_over_enclosing_trap_stop() {
        let (mb, is) = native_builder();
        let sig = Signature::new_unit("nested_checked_latch", Linkage::Public, &[Type::I1]);
        let fr = mb.declare_function(sig).unwrap();
        let mut fb = mb.func_builder::<InstInserter>(fr);
        let entry = fb.append_block();
        let header = fb.append_block();
        let guard_one = fb.append_block();
        let guard_two = fb.append_block();
        let branch = fb.append_block();
        let body = fb.append_block();
        let latch = fb.append_block();
        let first_trap = fb.append_block();
        let shared_trap = fb.append_block();
        let exit = fb.append_block();

        fb.switch_to_block(entry);
        let cond = fb.args()[0];
        fb.insert_inst_no_result(Jump::new(is, header));
        fb.switch_to_block(header);
        fb.insert_inst_no_result(Br::new(is, cond, guard_one, exit));
        fb.switch_to_block(guard_one);
        fb.insert_inst_no_result(Br::new(is, cond, first_trap, guard_two));
        fb.switch_to_block(guard_two);
        fb.insert_inst_no_result(Br::new(is, cond, shared_trap, branch));
        fb.switch_to_block(branch);
        fb.insert_inst_no_result(Br::new(is, cond, body, latch));
        fb.switch_to_block(body);
        fb.insert_inst_no_result(Jump::new(is, latch));
        fb.switch_to_block(latch);
        fb.insert_inst_no_result(Br::new(is, cond, shared_trap, header));
        fb.switch_to_block(first_trap);
        fb.insert_inst_no_result(Unreachable::new(is));
        fb.switch_to_block(shared_trap);
        fb.insert_inst_no_result(Unreachable::new(is));
        fb.switch_to_block(exit);
        fb.insert_inst_no_result(Return::new_unit(is));
        fb.seal_all();
        fb.finish();

        fn contains_merge(regions: &[Region], wanted: BlockId) -> bool {
            regions.iter().any(|region| match region {
                Region::IfThenElse {
                    then_branch,
                    else_branch,
                    merge,
                    ..
                } => {
                    *merge == Some(wanted)
                        || contains_merge(then_branch, wanted)
                        || contains_merge(else_branch, wanted)
                }
                Region::Loop { body, .. } => contains_merge(body, wanted),
                _ => false,
            })
        }

        let module = mb.build();
        let structured = structurize(&module, fr);
        assert!(
            contains_merge(&structured.regions, latch),
            "expected the inner branch to merge at its latch: {:?}",
            structured.regions
        );
    }

    /// Once an enclosing selection has established a nonterminal latch, a
    /// nested checked arm must retain that latch as its stop. A shared trap is
    /// reachable from the continuing arm too, but it is not that arm's merge.
    #[test]
    fn structurize_nested_trap_arm_retains_enclosing_latch() {
        let (mb, is) = native_builder();
        let sig = Signature::new_unit("nested_trap_latch", Linkage::Public, &[Type::I1]);
        let fr = mb.declare_function(sig).unwrap();
        let mut fb = mb.func_builder::<InstInserter>(fr);
        let entry = fb.append_block();
        let header = fb.append_block();
        let decision = fb.append_block();
        let guarded = fb.append_block();
        let nested = fb.append_block();
        let path = fb.append_block();
        let latch = fb.append_block();
        let trap = fb.append_block();
        let exit = fb.append_block();

        fb.switch_to_block(entry);
        let cond = fb.args()[0];
        fb.insert_inst_no_result(Jump::new(is, header));
        fb.switch_to_block(header);
        fb.insert_inst_no_result(Br::new(is, cond, decision, exit));
        fb.switch_to_block(decision);
        fb.insert_inst_no_result(Br::new(is, cond, guarded, latch));
        fb.switch_to_block(guarded);
        fb.insert_inst_no_result(Br::new(is, cond, nested, trap));
        fb.switch_to_block(nested);
        fb.insert_inst_no_result(Br::new(is, cond, path, trap));
        fb.switch_to_block(path);
        fb.insert_inst_no_result(Jump::new(is, latch));
        fb.switch_to_block(latch);
        fb.insert_inst_no_result(Br::new(is, cond, trap, header));
        fb.switch_to_block(trap);
        fb.insert_inst_no_result(Unreachable::new(is));
        fb.switch_to_block(exit);
        fb.insert_inst_no_result(Return::new_unit(is));
        fb.seal_all();
        fb.finish();

        let module = mb.build();
        let structured = structurize(&module, fr);
        let loop_body = structured
            .regions
            .iter()
            .find_map(|region| match region {
                Region::Loop { body, .. } => Some(body),
                _ => None,
            })
            .expect("expected loop region");
        assert!(loop_body.iter().any(|region| matches!(
            region,
            Region::IfThenElse { merge: Some(found), .. } if *found == latch
        )));
    }

    /// A 2-deep nested loop structurizes as Loop-in-Loop.
    #[test]
    fn structurize_nested_loop() {
        let (mb, is) = native_builder();
        let sig = Signature::new_single("nested", Linkage::Public, &[], Type::I32);
        let fr = mb.declare_function(sig).unwrap();
        let mut fb = mb.func_builder::<InstInserter>(fr);
        let entry = fb.append_block();
        let oh = fb.append_block();
        let ih = fb.append_block();
        let ib = fb.append_block();
        let idone = fb.append_block();
        let oexit = fb.append_block();

        fb.switch_to_block(entry);
        let zero = fb.make_imm_value(0i32);
        let eight = fb.make_imm_value(8i32);
        fb.insert_inst_no_result(Jump::new(is, oh));

        fb.switch_to_block(oh);
        let i = fb.insert_inst(Phi::new(is, vec![(zero, entry)]), Type::I32);
        let s_outer = fb.insert_inst(Phi::new(is, vec![(zero, entry)]), Type::I32);
        let oc = fb.insert_inst(cmp::Lt::new(is, i, eight), Type::I1);
        fb.insert_inst_no_result(Br::new(is, oc, ih, oexit));

        fb.switch_to_block(ih);
        let j = fb.insert_inst(Phi::new(is, vec![(zero, oh)]), Type::I32);
        let s_inner = fb.insert_inst(Phi::new(is, vec![(s_outer, oh)]), Type::I32);
        let ic = fb.insert_inst(cmp::Lt::new(is, j, eight), Type::I1);
        fb.insert_inst_no_result(Br::new(is, ic, ib, idone));

        fb.switch_to_block(ib);
        let one = fb.make_imm_value(1i32);
        let s1 = fb.insert_inst(arith::Add::new(is, s_inner, one), Type::I32);
        let j1 = fb.insert_inst(arith::Add::new(is, j, one), Type::I32);
        fb.append_phi_arg(j, j1, ib);
        fb.append_phi_arg(s_inner, s1, ib);
        fb.insert_inst_no_result(Jump::new(is, ih));

        fb.switch_to_block(idone);
        let i1 = fb.insert_inst(arith::Add::new(is, i, one), Type::I32);
        fb.append_phi_arg(i, i1, idone);
        fb.append_phi_arg(s_outer, s_inner, idone);
        fb.insert_inst_no_result(Jump::new(is, oh));

        fb.switch_to_block(oexit);
        fb.insert_inst_no_result(Return::new_single(is, s_outer));
        fb.seal_all();
        fb.finish();

        let module = mb.build();
        let s = structurize(&module, fr);
        let outer_body = s
            .regions
            .iter()
            .find_map(|r| match r {
                Region::Loop { body, .. } => Some(body),
                _ => None,
            })
            .expect("expected an outer Loop");
        assert!(
            outer_body.iter().any(|r| matches!(r, Region::Loop { .. })),
            "outer loop body should nest an inner Loop, got {:?}",
            outer_body
        );
    }

    #[test]
    fn irreducible_multi_entry_cycle_fails_closed() {
        let (mb, is) = native_builder();
        let sig = Signature::new_unit("irreducible", Linkage::Public, &[]);
        let fr = mb.declare_function(sig).unwrap();
        let mut fb = mb.func_builder::<InstInserter>(fr);
        let entry = fb.append_block();
        let a = fb.append_block();
        let b = fb.append_block();
        let c = fb.append_block();
        let d = fb.append_block();
        let exit = fb.append_block();

        fb.switch_to_block(entry);
        let cond = fb.make_imm_value(true);
        fb.insert_inst_no_result(Br::new(is, cond, a, b));
        fb.switch_to_block(a);
        fb.insert_inst_no_result(Jump::new(is, c));
        fb.switch_to_block(b);
        fb.insert_inst_no_result(Jump::new(is, d));
        fb.switch_to_block(c);
        fb.insert_inst_no_result(Br::new(is, cond, exit, d));
        fb.switch_to_block(d);
        fb.insert_inst_no_result(Jump::new(is, c));
        fb.switch_to_block(exit);
        fb.insert_inst_no_result(Return::new_unit(is));
        fb.seal_all();
        fb.finish();

        let module = mb.build();
        let result = module
            .func_store
            .view(fr, |func| structurize_function(func));
        let err = result.expect_err("irreducible cycle must fail closed");
        assert!(
            err.contains("cyclic")
                || err.contains("consumed")
                || err.contains("reachable")
                || err.contains("unstructured/unsupported")
                || err.contains("irreducible/unsupported"),
            "expected a named fail-closed structurizer error, got: {err}"
        );
    }

    #[test]
    fn loop_early_return_corridor_is_not_misclassified_as_break() {
        let (mb, is) = native_builder();
        let sig = Signature::new_unit("two_loop_exits", Linkage::Public, &[]);
        let fr = mb.declare_function(sig).unwrap();
        let mut fb = mb.func_builder::<InstInserter>(fr);
        let entry = fb.append_block();
        let header = fb.append_block();
        let body = fb.append_block();
        let latch = fb.append_block();
        let canonical_exit = fb.append_block();
        let other_exit = fb.append_block();
        let terminal = fb.append_block();

        fb.switch_to_block(entry);
        let cond = fb.make_imm_value(true);
        fb.insert_inst_no_result(Jump::new(is, header));
        fb.switch_to_block(header);
        fb.insert_inst_no_result(Br::new(is, cond, body, canonical_exit));
        fb.switch_to_block(body);
        fb.insert_inst_no_result(Br::new(is, cond, other_exit, latch));
        fb.switch_to_block(latch);
        fb.insert_inst_no_result(Jump::new(is, header));
        fb.switch_to_block(canonical_exit);
        fb.insert_inst_no_result(Return::new_unit(is));
        fb.switch_to_block(other_exit);
        fb.insert_inst_no_result(Jump::new(is, terminal));
        fb.switch_to_block(terminal);
        fb.insert_inst_no_result(Return::new_unit(is));
        fb.seal_all();
        fb.finish();

        let module = mb.build();
        let structured = module
            .func_store
            .view(fr, |func| structurize_function(func))
            .expect("a return-only corridor should remain inside the loop arm");
        let loop_body = structured
            .regions
            .iter()
            .find_map(|region| match region {
                Region::Loop { body, .. } => Some(body),
                _ => None,
            })
            .expect("expected loop region");
        assert!(
            loop_body.iter().any(|region| matches!(
                region,
                Region::IfThenElse { then_branch, else_branch, .. }
                    if then_branch
                        .iter()
                        .chain(else_branch)
                        .filter(|r| matches!(r, Region::Block(_)))
                        .count()
                        >= 2
            )),
            "expected the two-block return corridor in the loop arm, got: {loop_body:?}"
        );
    }

    #[test]
    fn nonreturning_noncanonical_loop_exit_fails_closed_by_name() {
        let (mb, is) = native_builder();
        let sig = Signature::new_unit("noncanonical_loop_exit", Linkage::Public, &[]);
        let fr = mb.declare_function(sig).unwrap();
        let mut fb = mb.func_builder::<InstInserter>(fr);
        let entry = fb.append_block();
        let header = fb.append_block();
        let body = fb.append_block();
        let latch = fb.append_block();
        let canonical_exit = fb.append_block();
        let other_exit = fb.append_block();

        fb.switch_to_block(entry);
        let cond = fb.make_imm_value(true);
        fb.insert_inst_no_result(Jump::new(is, header));
        fb.switch_to_block(header);
        fb.insert_inst_no_result(Br::new(is, cond, body, canonical_exit));
        fb.switch_to_block(body);
        fb.insert_inst_no_result(Br::new(is, cond, other_exit, latch));
        fb.switch_to_block(latch);
        fb.insert_inst_no_result(Jump::new(is, header));
        fb.switch_to_block(other_exit);
        fb.insert_inst_no_result(Br::new(is, cond, canonical_exit, canonical_exit));
        fb.switch_to_block(canonical_exit);
        fb.insert_inst_no_result(Return::new_unit(is));
        fb.seal_all();
        fb.finish();

        let module = mb.build();
        let result = module
            .func_store
            .view(fr, |func| structurize_function(func));
        let error = result.expect_err("a noncanonical break must fail closed");
        assert!(
            error.contains("noncanonical exit") && error.contains("expected the header exit"),
            "expected a named noncanonical-exit error, got: {error}"
        );
    }
}
