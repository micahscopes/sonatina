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

use std::collections::{HashMap, HashSet, VecDeque};

use bit_set::BitSet;
use sonatina_ir::{
    BlockId, Function, InstDowncast, InstSetBase,
    cfg::ControlFlowGraph,
    inst::control_flow::{Br, BrTable, Jump, Phi, Return, Unreachable},
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
    /// `header` and consumed by the emitter's loop preamble. A single
    /// exhaustion-only forwarding block, recognized by `forwarded_loop_exit`,
    /// is likewise consumed on that header's exit edge rather than as a sibling.
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

/// Compact structural measurements for one reconstructed region tree.
///
/// `block_occurrences` counts every block body position in the tree, including
/// loop and conditional headers. `duplicated_block_occurrences` therefore
/// measures the exact CFG-to-tree cloning pressure introduced by structuring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StructuredCfgStats {
    pub region_nodes: usize,
    pub reachable_blocks: usize,
    pub referenced_blocks: usize,
    pub block_occurrences: usize,
    pub duplicated_block_occurrences: usize,
    pub loops: usize,
    pub conditionals: usize,
    pub loop_exits: usize,
    pub loop_continues: usize,
}

impl StructuredCfg {
    pub fn stats(&self) -> StructuredCfgStats {
        fn visit(
            regions: &[Region],
            stats: &mut StructuredCfgStats,
            blocks: &mut HashSet<BlockId>,
        ) {
            for region in regions {
                stats.region_nodes += 1;
                match region {
                    Region::Block(block) => {
                        stats.block_occurrences += 1;
                        blocks.insert(*block);
                    }
                    Region::Loop { header, body } => {
                        stats.loops += 1;
                        stats.block_occurrences += 1;
                        blocks.insert(*header);
                        visit(body, stats, blocks);
                    }
                    Region::IfThenElse {
                        header,
                        then_branch,
                        else_branch,
                        ..
                    } => {
                        stats.conditionals += 1;
                        stats.block_occurrences += 1;
                        blocks.insert(*header);
                        visit(then_branch, stats, blocks);
                        visit(else_branch, stats, blocks);
                    }
                    Region::LoopExit { .. } => stats.loop_exits += 1,
                    Region::LoopContinue { .. } => stats.loop_continues += 1,
                }
            }
        }

        let mut stats = StructuredCfgStats {
            region_nodes: 0,
            reachable_blocks: self.block_order.len(),
            referenced_blocks: 0,
            block_occurrences: 0,
            duplicated_block_occurrences: 0,
            loops: 0,
            conditionals: 0,
            loop_exits: 0,
            loop_continues: 0,
        };
        let mut blocks = HashSet::new();
        visit(&self.regions, &mut stats, &mut blocks);
        stats.referenced_blocks = blocks.len();
        stats.duplicated_block_occurrences = stats
            .block_occurrences
            .saturating_sub(stats.referenced_blocks);
        stats
    }
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
        cfg: &cfg,
        is: function.inst_set(),
        loop_tree: &loop_tree,
        postdom: &postdom,
    };
    let mut active = HashSet::new();
    let mut consumed = HashSet::new();
    let regions = s.build_seq(rpo[0], None, None, false, &mut active, &mut consumed)?;

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

/// An exhaustion-only block can construct a fallback before joining an early
/// loop exit. Realize that block on the header's exit edge, not unconditionally
/// after the loop. Only a single-block corridor with exactly one incoming edge
/// is admitted here; more general multi-exit loops retain their diagnostics.
pub(crate) fn forwarded_loop_exit(
    function: &Function,
    header: BlockId,
    exit: BlockId,
    in_loop: impl Fn(BlockId) -> bool,
) -> Option<BlockId> {
    let is = function.inst_set();
    let join = function.layout.iter_inst(exit).find_map(|iid| {
        <&Jump as InstDowncast>::downcast(is, function.dfg.inst(iid)).map(|j| *j.dest())
    })?;
    if in_loop(join) { return None; }
    let mut cfg = ControlFlowGraph::default();
    cfg.compute(function);
    let mut predecessors = cfg.preds_of(exit);
    if predecessors.next().copied() != Some(header) || predecessors.next().is_some() {
        return None;
    }
    cfg.preds_of(join).any(|pred| *pred != header && in_loop(*pred)).then_some(join)
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
    cfg: &'a ControlFlowGraph,
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

    /// Whether every control-flow exit from `start` reaches the loop's
    /// canonical fallthrough block without returning to the outer header. A
    /// corridor may contain a compiler-recognized nested loop, so a reducible
    /// backedge is valid as long as at least one path leaves that loop and all
    /// such exits continue to the canonical fallthrough. These blocks belong
    /// inside the structured break arm because they compute values consumed by
    /// exit phis.
    fn is_loop_exit_corridor(&self, start: BlockId, lp: Loop) -> bool {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        enum Reach {
            Invalid,
            BackedgeOnly,
            Canonical,
        }

        fn combine(left: Reach, right: Reach) -> Reach {
            match (left, right) {
                (Reach::Invalid, _) | (_, Reach::Invalid) => Reach::Invalid,
                (Reach::Canonical, _) | (_, Reach::Canonical) => Reach::Canonical,
                (Reach::BackedgeOnly, Reach::BackedgeOnly) => Reach::BackedgeOnly,
            }
        }

        fn visit(
            s: &Structurer<'_>,
            block: BlockId,
            lp: Loop,
            canonical: BlockId,
            visiting: &mut HashSet<BlockId>,
            memo: &mut HashMap<BlockId, Reach>,
        ) -> Reach {
            if block == canonical {
                return Reach::Canonical;
            }
            // Loop-tree membership is too coarse here: an exit-arm tail may be
            // assigned to the outer loop even though it cannot reach the
            // backedge. Reaching the header itself is the semantic continue
            // boundary that a break corridor must never cross.
            if block == s.loop_tree.loop_header(lp) {
                return Reach::Invalid;
            }
            if let Some(&result) = memo.get(&block) {
                return result;
            }
            if visiting.contains(&block) {
                return match s.loop_tree.loop_of_block(block) {
                    Some(nested)
                        if nested != lp && s.loop_tree.loop_header(nested) == block =>
                    {
                        Reach::BackedgeOnly
                    }
                    _ => Reach::Invalid,
                };
            }
            visiting.insert(block);
            let result = match s.term(block) {
                Term::Jump(target) => visit(s, target, lp, canonical, visiting, memo),
                Term::Br(nz, z) => combine(
                    visit(s, nz, lp, canonical, visiting, memo),
                    visit(s, z, lp, canonical, visiting, memo),
                ),
                Term::Return | Term::Unreachable | Term::Other => Reach::Invalid,
            };
            visiting.remove(&block);
            memo.insert(block, result);
            if std::env::var_os("SONATINA_STRUCTURIZE_TRACE").is_some() {
                eprintln!(
                    "[spirv structurize] corridor block={block:?}, owner={:?}, result={result:?}",
                    s.loop_tree.loop_of_block(block),
                );
            }
            result
        }

        let header = self.loop_tree.loop_header(lp);
        let Some(canonical) = self.loop_direct_exit(header, lp) else {
            return false;
        };
        let result = start != canonical
            && visit(
                self,
                start,
                lp,
                canonical,
                &mut HashSet::new(),
                &mut HashMap::new(),
            ) == Reach::Canonical;
        if std::env::var_os("SONATINA_STRUCTURIZE_TRACE").is_some() {
            eprintln!(
                "[spirv structurize] corridor start={start:?}, outer={lp:?}, canonical={canonical:?}, result={result}"
            );
        }
        result
    }

    fn in_loop(&self, b: BlockId, lp: Loop) -> bool {
        self.loop_tree.is_in_loop(b, lp)
    }

    /// A block that is always a CFG dead end and contains only its terminal.
    ///
    /// `wasm_lower.rs::trap_block` caches one `Unreachable` block per function,
    /// and optimized unit-returning functions likewise commonly funnel many
    /// arms into one bare `Return`. Neither block owns values, side effects, or
    /// phi transport. A structured target may therefore reference it at each
    /// tree position without violating semantics. The general consumed-block
    /// guard must remain strict for every non-bare terminal.
    fn is_shared_bare_terminal(&self, block: BlockId) -> bool {
        matches!(self.term(block), Term::Return | Term::Unreachable)
            && self.function.layout.iter_inst(block).count() == 1
    }

    /// A shared, phi-free block directly before a bare terminal is likewise a
    /// safe tree leaf. Its side effects execute once on whichever mutually
    /// exclusive path reaches that leaf, and it owns no edge-dependent values.
    fn is_shared_terminal_forwarder(&self, block: BlockId) -> bool {
        let Term::Jump(target) = self.term(block) else {
            return false;
        };
        self.is_shared_bare_terminal(target)
            && self.function.layout.iter_inst(block).all(|inst_id| {
                <&Phi as InstDowncast>::downcast(self.is, self.function.dfg.inst(inst_id)).is_none()
            })
    }

    /// A phi-free decision block may be referenced from several mutually
    /// exclusive CFG paths. Structured targets are trees, so each reference
    /// needs its own region node even though all nodes retain the original
    /// block identity. Phi-bearing decisions remain single-owner because their
    /// incoming edge values cannot be cloned without predecessor rewriting.
    fn is_shared_decision_block(&self, block: BlockId) -> bool {
        matches!(self.term(block), Term::Br(_, _))
            && self.function.layout.iter_inst(block).all(|inst_id| {
                <&Phi as InstDowncast>::downcast(self.is, self.function.dfg.inst(inst_id)).is_none()
            })
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
        clone_shared_subtree: bool,
        active: &mut HashSet<BlockId>,
        consumed: &mut HashSet<BlockId>,
    ) -> Result<Vec<Region>, String> {
        let mut regions = Vec::new();
        let mut cur = Some(start);
        let mut cloning = clone_shared_subtree;
        let allow_return_corridor = cur_loop
            .is_some_and(|lp| !self.in_loop(start, lp) && self.is_return_corridor(start, lp));
        let allow_loop_exit_corridor = cur_loop
            .is_some_and(|lp| !self.in_loop(start, lp) && self.is_loop_exit_corridor(start, lp));

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
                if !self.in_loop(b, lp)
                    && b != start
                    && !self.returns(b)
                    && !allow_return_corridor
                    && !allow_loop_exit_corridor
                {
                    break;
                }
            }

            // A shared bare terminal dead-ends here regardless of how many
            // predecessors reach it; see `is_shared_bare_terminal`.
            if self.is_shared_bare_terminal(b) {
                consumed.insert(b);
                regions.push(Region::Block(b));
                cur = None;
                continue;
            }

            // A nonempty forwarder directly into this arm's merge can also be
            // reached after structuring a nested loop fallthrough, rather than
            // as the branch's immediate target. As with `build_branch`'s
            // direct-merge case, mutually exclusive tree positions may each
            // reference the original block. Phi transport remains owned by the
            // merge after it, and an active block is never cloned.
            if !active.contains(&b)
                && consumed.contains(&b)
                && stop.is_some_and(|merge| self.is_direct_merge_arm(b, merge))
            {
                regions.push(Region::Block(b));
                cur = None;
                continue;
            }
            // Nested selections may share a nonempty backedge arm without
            // sharing their other continuation. Such an arm is terminal for
            // this iteration, not a join that every sibling must reach.
            // Reuse its original block identity at mutually exclusive tree
            // positions, just as for direct merge arms above. The emitter
            // transports incoming phis on the original predecessor edges,
            // then emits this block's exact header-phi edge and Continue.
            // Never clone an active block or a backedge to a different loop.
            if !active.contains(&b)
                && consumed.contains(&b)
                && cur_loop.is_some_and(|lp| {
                    self.in_loop(b, lp)
                        && self.is_direct_merge_arm(b, self.loop_tree.loop_header(lp))
                })
            {
                regions.push(Region::Block(b));
                cur = None;
                continue;
            }
            if !active.contains(&b) && consumed.contains(&b) && self.is_shared_terminal_forwarder(b)
            {
                let Term::Jump(terminal) = self.term(b) else {
                    unreachable!("terminal forwarder classification changed")
                };
                consumed.insert(terminal);
                regions.push(Region::Block(b));
                regions.push(Region::Block(terminal));
                cur = None;
                continue;
            }

            let clone_shared_decision =
                !active.contains(&b) && consumed.contains(&b) && self.is_shared_decision_block(b);
            let clone_existing =
                !active.contains(&b) && consumed.contains(&b) && (cloning || clone_shared_decision);
            if active.contains(&b) || (!clone_existing && !consumed.insert(b)) {
                let predecessors = self.cfg.preds_of(b).copied().collect::<Vec<_>>();
                let successors = self.cfg.succs_of(b).copied().collect::<Vec<_>>();
                let owner_loop = self.loop_tree.loop_of_block(b);
                let owner_header = owner_loop.map(|lp| self.loop_tree.loop_header(lp));
                let predecessor_headers = predecessors
                    .iter()
                    .map(|pred| {
                        self.loop_tree
                            .loop_of_block(*pred)
                            .map(|lp| self.loop_tree.loop_header(lp))
                    })
                    .collect::<Vec<_>>();
                return Err(format!(
                    "spirv structurize: cyclic or multiply consumed block {b:?} while building \
                     sequence {start:?}..{stop:?}; active={active:?}; \
                     predecessors={predecessors:?}; successors={successors:?}; \
                     owner_loop={owner_loop:?}; owner_header={owner_header:?}; \
                     predecessor_headers={predecessor_headers:?}"
                ));
            }
            cloning |= clone_shared_decision;
            active.insert(b);

            // Open a loop when we first reach a header we are not already in.
            if let Some(lp) = self.loop_tree.loop_of_block(b) {
                if self.loop_tree.loop_header(lp) == b && cur_loop != Some(lp) {
                    let body = self.build_loop_body(b, lp, cloning, active, consumed)?;
                    regions.push(Region::Loop { header: b, body });
                    if let Some(exit) = self.loop_direct_exit(b, lp) {
                        if self.returns(exit) {
                            consumed.insert(exit);
                        } else if let Some(join) = forwarded_loop_exit(self.function, b, exit,
                            |block| self.in_loop(block, lp))
                        {
                            // The emitter owns this block on the header's
                            // exhaustion edge, just as it owns a returning exit.
                            consumed.insert(exit);
                            if self.returns(join) { consumed.insert(join); }
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
                        && (self.is_canonical_loop_exit(lp, t)
                            || (!allow_loop_exit_corridor
                                && !self.in_loop(t, lp)
                                && !self.returns(t)
                                && !self.is_return_corridor(t, lp)))
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
                    let then_branch = self
                        .build_branch(b, nz, merge, cur_loop, cloning, active, consumed)
                        .map_err(|error| {
                            format!(
                                "{error}; while structuring nonzero arm {b:?}->{nz:?}, \
                                 selected_merge={merge:?}, enclosing_stop={stop:?}, \
                                 current_loop={cur_loop:?}"
                            )
                        })?;
                    let else_branch = self
                        .build_branch(b, z, merge, cur_loop, cloning, active, consumed)
                        .map_err(|error| {
                            format!(
                                "{error}; while structuring zero arm {b:?}->{z:?}, \
                                 selected_merge={merge:?}, enclosing_stop={stop:?}, \
                                 current_loop={cur_loop:?}"
                            )
                        })?;
                    regions.push(Region::IfThenElse {
                        header: b,
                        then_branch,
                        else_branch,
                        merge,
                    });
                    cur = if let (Some(lp), Some(merge)) = (cur_loop, merge)
                        && self.is_canonical_loop_exit(lp, merge)
                    {
                        None
                    } else {
                        merge
                    };
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
        clone_shared_subtree: bool,
        active: &mut HashSet<BlockId>,
        consumed: &mut HashSet<BlockId>,
    ) -> Result<Vec<Region>, String> {
        if let Some(lp) = cur_loop
            && target == self.loop_tree.loop_header(lp)
        {
            return Ok(vec![Region::LoopContinue { from, target }]);
        }
        if let Some(lp) = cur_loop
            && self.is_canonical_loop_exit(lp, target)
        {
            return Ok(vec![Region::LoopExit { from, target }]);
        }
        if Some(target) == merge {
            return Ok(Vec::new());
        }
        if let Some(merge) = merge
            && self.is_direct_merge_arm(target, merge)
        {
            // A reducible CFG may share one direct-to-merge block between
            // nested, mutually exclusive selections. A structured target is
            // a tree, so clone that arm into each tree position. The block's
            // original identity remains intact for exact merge-phi transport,
            // while each emitted copy has its own branch-local value map.
            consumed.insert(target);
            return Ok(vec![Region::Block(target)]);
        }
        if let Some(lp) = cur_loop
            && !self.in_loop(target, lp)
            && self.is_loop_exit_corridor(target, lp)
        {
            return self.build_seq(
                target,
                merge,
                cur_loop,
                clone_shared_subtree,
                active,
                consumed,
            );
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
        self.build_seq(
            target,
            merge,
            cur_loop,
            clone_shared_subtree,
            active,
            consumed,
        )
    }

    fn is_canonical_loop_exit(&self, lp: Loop, target: BlockId) -> bool {
        let header = self.loop_tree.loop_header(lp);
        self.loop_direct_exit(header, lp) == Some(target)
    }

    fn is_direct_merge_arm(&self, block: BlockId, target: BlockId) -> bool {
        matches!(self.term(block), Term::Jump(destination) if destination == target)
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
        clone_shared_subtree: bool,
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
                    self.build_seq(
                        entry,
                        None,
                        Some(lp),
                        clone_shared_subtree,
                        active,
                        consumed,
                    )
                }
            }
            Term::Jump(t) => {
                self.build_seq(t, None, Some(lp), clone_shared_subtree, active, consumed)
            }
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

    /// The block execution resumes at after the loop: the header's exit, or
    /// the shared join after an exhaustion-only forwarder. `None` for a
    /// returning exit (the emitter funnels its return value out of the loop).
    fn loop_fallthrough(&self, header: BlockId, lp: Loop) -> Result<Option<BlockId>, String> {
        match self.term(header) {
            Term::Br(nz, z) => {
                let direct_exit = if self.in_loop(nz, lp) { z } else { nz };
                let exit = forwarded_loop_exit(self.function, header, direct_exit,
                    |block| self.in_loop(block, lp)).unwrap_or(direct_exit);
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
        if std::env::var_os("SONATINA_STRUCTURIZE_TRACE").is_some() {
            eprintln!(
                "[spirv structurize] merge header={header:?}, successors=({nz:?}, {z:?}), enclosing_stop={enclosing_stop:?}, immediate={immediate:?}"
            );
        }
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
                // A nested bounds guard can reach the enclosing merge through
                // a short live corridor while its other arm traps. Do this
                // check before testing whether the live arm can eventually
                // reach the shared trap through that merge. Mere downstream
                // reachability of the trap must not erase the closer merge.
                if let Some(stop) = enclosing_stop
                    && self.is_local_merge_candidate(stop, cur_loop)
                {
                    if self.returns(z) && self.all_nonterminal_paths_reach(nz, stop, cur_loop) {
                        return Ok(Some(stop));
                    }
                    if self.returns(nz) && self.all_nonterminal_paths_reach(z, stop, cur_loop) {
                        return Ok(Some(stop));
                    }
                }
                // A terminal arm does not need to reach the continuation for
                // that continuation to be the structured merge. Keep an
                // enclosing loop header or exit as an explicit continue/break
                // edge, but otherwise resume at the live successor.
                if self.returns(nz)
                    && !self.reaches_before_loop_header(z, nz, cur_loop)
                    && self.is_local_merge_candidate(z, cur_loop)
                {
                    return Ok(Some(z));
                }
                if self.returns(z)
                    && !self.reaches_before_loop_header(nz, z, cur_loop)
                    && self.is_local_merge_candidate(nz, cur_loop)
                {
                    return Ok(Some(nz));
                }
                // Search only within the current loop iteration so a later
                // backedge cannot make the two successors appear mutually
                // reachable.
                let nz_reaches_z = self.reaches_before_loop_header(nz, z, cur_loop);
                let z_reaches_nz = self.reaches_before_loop_header(z, nz, cur_loop);
                match (nz_reaches_z, z_reaches_nz) {
                    (true, false) if self.all_nonterminal_paths_reach(nz, z, cur_loop) => {
                        if self.returns(z)
                            && let Some(stop) = enclosing_stop
                            && self.is_local_merge_candidate(stop, cur_loop)
                            && self.all_nonterminal_paths_reach(nz, stop, cur_loop)
                        {
                            return Ok(Some(stop));
                        }
                        Ok(Some(z))
                    }
                    (false, true) if self.all_nonterminal_paths_reach(z, nz, cur_loop) => {
                        if self.returns(nz)
                            && let Some(stop) = enclosing_stop
                            && self.is_local_merge_candidate(stop, cur_loop)
                            && self.all_nonterminal_paths_reach(z, stop, cur_loop)
                        {
                            return Ok(Some(stop));
                        }
                        Ok(Some(nz))
                    }
                    _ => {
                        // Prefer the closest live continuation over the
                        // enclosing arm's stop. A nested trap or early return
                        // removes the local join from strict post-dominance,
                        // but does not make an outer continuation the semantic
                        // merge of the remaining live paths. Choosing the
                        // outer stop here would structure the local join once
                        // inside each sibling arm. Phi-bearing joins cannot be
                        // cloned, and their edge transport must remain owned by
                        // one region.
                        if let Some(merge) = self.nearest_nonterminal_merge(nz, z, cur_loop) {
                            return Ok(Some(merge));
                        }
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

    fn is_local_merge_candidate(&self, block: BlockId, cur_loop: Option<Loop>) -> bool {
        // A bare return or trap is a terminal leaf, but a terminal block that
        // also owns a phi or other instructions is a real convergence point.
        // Match lowering commonly produces exactly that shape: live arms join
        // through a result phi and return, while the default arm traps.
        let owns_join = !self.returns(block) || !self.is_shared_bare_terminal(block);
        let Some(lp) = cur_loop else {
            return owns_join;
        };
        block != self.loop_tree.loop_header(lp) && self.in_loop(block, lp) && owns_join
    }

    /// Find the closest block reached by every nonterminal path from both
    /// successors. This complements strict post-dominance when early returns
    /// or traps make the ordinary post-dominator set empty. In particular, a
    /// nested loop header can be the common continuation of several guarded
    /// arms even though terminal siblings never reach it.
    fn nearest_nonterminal_merge(
        &self,
        left: BlockId,
        right: BlockId,
        cur_loop: Option<Loop>,
    ) -> Option<BlockId> {
        let left_distances = self.reachable_distances_before_loop_header(left, cur_loop);
        let right_distances = self.reachable_distances_before_loop_header(right, cur_loop);
        let mut candidates = left_distances
            .iter()
            .filter_map(|(candidate, left_distance)| {
                let right_distance = right_distances.get(candidate)?;
                if !self.is_local_merge_candidate(*candidate, cur_loop) {
                    return None;
                }
                Some((
                    (*left_distance).max(*right_distance),
                    *left_distance + *right_distance,
                    self.postdom.indices[candidate],
                    *candidate,
                ))
            })
            .collect::<Vec<_>>();
        candidates.sort_unstable_by_key(|candidate| (candidate.0, candidate.1, candidate.2));
        candidates
            .into_iter()
            .map(|candidate| candidate.3)
            .find(|candidate| {
                self.all_nonterminal_paths_reach(left, *candidate, cur_loop)
                    && self.all_nonterminal_paths_reach(right, *candidate, cur_loop)
            })
    }

    fn reachable_distances_before_loop_header(
        &self,
        start: BlockId,
        cur_loop: Option<Loop>,
    ) -> HashMap<BlockId, usize> {
        let loop_header = cur_loop.map(|lp| self.loop_tree.loop_header(lp));
        let mut distances = HashMap::new();
        let mut pending = VecDeque::from([(start, 0usize)]);
        while let Some((block, distance)) = pending.pop_front() {
            if distances.contains_key(&block) {
                continue;
            }
            distances.insert(block, distance);
            if Some(block) == loop_header && block != start {
                continue;
            }
            match self.term(block) {
                Term::Jump(next) => pending.push_back((next, distance + 1)),
                Term::Br(nz, z) => {
                    pending.push_back((nz, distance + 1));
                    pending.push_back((z, distance + 1));
                }
                Term::Return | Term::Unreachable | Term::Other => {}
            }
        }
        distances
    }

    fn all_nonterminal_paths_reach(
        &self,
        start: BlockId,
        target: BlockId,
        cur_loop: Option<Loop>,
    ) -> bool {
        fn visit(
            s: &Structurer<'_>,
            block: BlockId,
            start: BlockId,
            target: BlockId,
            loop_header: Option<BlockId>,
            target_downstream: &HashSet<BlockId>,
            visiting: &mut HashSet<BlockId>,
            memo: &mut HashMap<BlockId, bool>,
        ) -> bool {
            if block == target {
                return true;
            }
            if s.returns(block) {
                return true;
            }
            if block != start && target_downstream.contains(&block) {
                // This path skipped `target` and rejoined computation that is
                // also downstream of it. It is a live bypass, not an unrelated
                // terminal corridor, so `target` cannot be this selection's
                // merge.
                return false;
            }
            if let Some(result) = memo.get(&block) {
                return *result;
            }
            if Some(block) == loop_header {
                return false;
            }
            if !visiting.insert(block) {
                // A backedge inside a nested SCC does not escape the proposed
                // continuation. Treat the cycle as provisionally valid while
                // the originating visit checks every actual exit. An exit
                // that can avoid `target` still returns false through its own
                // branch, while a nonterminating cycle needs no merge.
                return true;
            }
            let result = match s.term(block) {
                Term::Return | Term::Unreachable => true,
                Term::Jump(next) => visit(
                    s,
                    next,
                    start,
                    target,
                    loop_header,
                    target_downstream,
                    visiting,
                    memo,
                ),
                Term::Br(nz, z) => {
                    visit(
                        s,
                        nz,
                        start,
                        target,
                        loop_header,
                        target_downstream,
                        visiting,
                        memo,
                    ) && visit(
                        s,
                        z,
                        start,
                        target,
                        loop_header,
                        target_downstream,
                        visiting,
                        memo,
                    )
                }
                Term::Other => false,
            };
            visiting.remove(&block);
            memo.insert(block, result);
            result
        }

        let target_downstream = self
            .reachable_distances_before_loop_header(target, cur_loop)
            .into_keys()
            .collect::<HashSet<_>>();
        visit(
            self,
            start,
            start,
            target,
            cur_loop.map(|lp| self.loop_tree.loop_header(lp)),
            &target_downstream,
            &mut HashSet::new(),
            &mut HashMap::new(),
        )
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
        let stats = structured.stats();
        assert_eq!(stats.reachable_blocks, 2);
        assert_eq!(stats.referenced_blocks, 2);
        assert_eq!(stats.block_occurrences, 2);
        assert_eq!(stats.duplicated_block_occurrences, 0);
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

    /// Bounds checks inside one loop body may share a trap while two live
    /// paths join at a phi-bearing latch. The latch is the inner branch merge,
    /// even though one of its successors is the enclosing loop header.
    #[test]
    fn guarded_loop_paths_merge_at_phi_latch() {
        let (mb, is) = native_builder();
        let sig = Signature::new_unit(
            "guarded_loop_phi_latch",
            Linkage::Public,
            &[Type::I1, Type::I1, Type::I1, Type::I1, Type::I1],
        );
        let fr = mb.declare_function(sig).unwrap();
        let mut fb = mb.func_builder::<InstInserter>(fr);

        let entry = fb.append_block();
        let header = fb.append_block();
        let first_guard = fb.append_block();
        let second_guard = fb.append_block();
        let choose_update = fb.append_block();
        let update_guard = fb.append_block();
        let update = fb.append_block();
        let latch = fb.append_block();
        let exit = fb.append_block();
        let trap = fb.append_block();

        let zero = fb.make_imm_value(0i32);
        let one = fb.make_imm_value(1i32);
        let conditions = fb.args().to_vec();

        fb.switch_to_block(entry);
        fb.insert_inst_no_result(Jump::new(is, header));
        fb.switch_to_block(header);
        fb.insert_inst_no_result(Br::new(is, conditions[0], first_guard, exit));
        fb.switch_to_block(first_guard);
        fb.insert_inst_no_result(Br::new(is, conditions[1], second_guard, trap));
        fb.switch_to_block(second_guard);
        fb.insert_inst_no_result(Br::new(is, conditions[2], choose_update, trap));
        fb.switch_to_block(choose_update);
        fb.insert_inst_no_result(Br::new(is, conditions[3], update_guard, latch));
        fb.switch_to_block(update_guard);
        fb.insert_inst_no_result(Br::new(is, conditions[4], update, trap));
        fb.switch_to_block(update);
        let updated = fb.insert_inst(arith::Add::new(is, one, one), Type::I32);
        fb.insert_inst_no_result(Jump::new(is, latch));
        fb.switch_to_block(latch);
        let carried = fb.insert_inst(
            Phi::new(is, vec![(zero, choose_update), (updated, update)]),
            Type::I32,
        );
        let keep_going = fb.insert_inst(cmp::Lt::new(is, zero, carried), Type::I1);
        fb.insert_inst_no_result(Br::new(is, keep_going, trap, header));
        fb.switch_to_block(exit);
        fb.insert_inst_no_result(Return::new_unit(is));
        fb.switch_to_block(trap);
        fb.insert_inst_no_result(Unreachable::new(is));
        fb.seal_all();
        fb.finish();

        let module = mb.build();
        let structured = structurize(&module, fr);
        assert!(structured.block_order.contains(&latch));
        fn owned_occurrences(regions: &[Region], target: BlockId) -> usize {
            regions
                .iter()
                .map(|region| match region {
                    Region::Block(block) => usize::from(*block == target),
                    Region::IfThenElse {
                        header,
                        then_branch,
                        else_branch,
                        ..
                    } => {
                        usize::from(*header == target)
                            + owned_occurrences(then_branch, target)
                            + owned_occurrences(else_branch, target)
                    }
                    Region::Loop { header, body } => {
                        usize::from(*header == target) + owned_occurrences(body, target)
                    }
                    Region::LoopExit { .. } | Region::LoopContinue { .. } => 0,
                })
                .sum()
        }
        assert_eq!(
            owned_occurrences(&structured.regions, latch),
            1,
            "the phi-bearing latch must have one owner: {:?}",
            structured.regions,
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

    /// Bounds-checked branches in a unit-returning kernel may converge on one
    /// bare return while their failure arms share one bare trap. Both terminal
    /// blocks are safe leaves at each structured tree position.
    #[test]
    fn nested_branches_may_share_bare_return_and_trap() {
        let (mb, is) = native_builder();
        let sig = Signature::new_unit(
            "shared_bare_terminals",
            Linkage::Public,
            &[Type::I1, Type::I1, Type::I1],
        );
        let fr = mb.declare_function(sig).unwrap();
        let mut fb = mb.func_builder::<InstInserter>(fr);
        let entry = fb.append_block();
        let left = fb.append_block();
        let right = fb.append_block();
        let left_live = fb.append_block();
        let right_live = fb.append_block();
        let return_block = fb.append_block();
        let trap_block = fb.append_block();

        fb.switch_to_block(entry);
        fb.insert_inst_no_result(Br::new(is, fb.args()[0], left, right));
        fb.switch_to_block(left);
        fb.insert_inst_no_result(Br::new(is, fb.args()[1], left_live, trap_block));
        fb.switch_to_block(right);
        fb.insert_inst_no_result(Br::new(is, fb.args()[2], right_live, trap_block));
        fb.switch_to_block(left_live);
        fb.insert_inst_no_result(Jump::new(is, return_block));
        fb.switch_to_block(right_live);
        fb.insert_inst_no_result(Jump::new(is, return_block));
        fb.switch_to_block(return_block);
        fb.insert_inst_no_result(Return::new_unit(is));
        fb.switch_to_block(trap_block);
        fb.insert_inst_no_result(Unreachable::new(is));
        fb.seal_all();
        fb.finish();

        let module = mb.build();
        let structured = structurize(&module, fr);
        assert!(structured.block_order.contains(&return_block));
        assert!(structured.block_order.contains(&trap_block));
    }

    /// A lowered match may send every valid arm to one phi-bearing return
    /// while its default arm traps. The return block is the owned merge of the
    /// live arms, not a clonable terminal leaf.
    #[test]
    fn match_arms_merge_at_phi_return_while_default_traps() {
        let (mb, is) = native_builder();
        let sig = Signature::new_single(
            "match_with_trapping_default",
            Linkage::Public,
            &[Type::I32],
            Type::I32,
        );
        let fr = mb.declare_function(sig).unwrap();
        let mut fb = mb.func_builder::<InstInserter>(fr);
        let entry = fb.append_block();
        let check_one = fb.append_block();
        let check_two = fb.append_block();
        let check_three = fb.append_block();
        let arm_zero = fb.append_block();
        let arm_one = fb.append_block();
        let arm_two = fb.append_block();
        let arm_three = fb.append_block();
        let merge = fb.append_block();
        let trap = fb.append_block();

        let selector = fb.args()[0];
        let zero = fb.make_imm_value(0i32);
        let one = fb.make_imm_value(1i32);
        let two = fb.make_imm_value(2i32);
        let three = fb.make_imm_value(3i32);
        let fallback = fb.make_imm_value(99i32);

        fb.switch_to_block(entry);
        let is_zero = fb.insert_inst(cmp::Eq::new(is, selector, zero), Type::I1);
        fb.insert_inst_no_result(Br::new(is, is_zero, arm_zero, check_one));
        fb.switch_to_block(check_one);
        let is_one = fb.insert_inst(cmp::Eq::new(is, selector, one), Type::I1);
        fb.insert_inst_no_result(Br::new(is, is_one, arm_one, check_two));
        fb.switch_to_block(check_two);
        let is_two = fb.insert_inst(cmp::Eq::new(is, selector, two), Type::I1);
        fb.insert_inst_no_result(Br::new(is, is_two, arm_two, check_three));
        fb.switch_to_block(check_three);
        let is_three = fb.insert_inst(cmp::Eq::new(is, selector, three), Type::I1);
        fb.insert_inst_no_result(Br::new(is, is_three, arm_three, trap));

        for arm in [arm_zero, arm_one, arm_two, arm_three] {
            fb.switch_to_block(arm);
            fb.insert_inst_no_result(Jump::new(is, merge));
        }
        fb.switch_to_block(merge);
        let result = fb.insert_inst(
            Phi::new(
                is,
                vec![
                    (zero, arm_zero),
                    (one, arm_one),
                    (two, arm_two),
                    (fallback, arm_three),
                ],
            ),
            Type::I32,
        );
        fb.insert_inst_no_result(Return::new_single(is, result));
        fb.switch_to_block(trap);
        fb.insert_inst_no_result(Unreachable::new(is));
        fb.seal_all();
        fb.finish();

        let module = mb.build();
        let structured = structurize(&module, fr);
        assert!(matches!(
            structured.regions.last(),
            Some(Region::Block(block)) if *block == merge
        ));
        assert_eq!(
            structured.stats().block_occurrences,
            structured.stats().referenced_blocks,
            "the phi-bearing merge must have one owner: {:?}",
            structured.regions,
        );
    }

    /// CFG cleanup removes the otherwise empty arm blocks from a lowered
    /// match, leaving each successful comparison to branch directly to the
    /// phi-bearing return. The return remains the one owned merge of all live
    /// arms even though it is also an immediate successor of every decision.
    #[test]
    fn optimized_match_arms_merge_directly_at_phi_return_while_default_traps() {
        let (mb, is) = native_builder();
        let sig = Signature::new_single(
            "optimized_match_with_trapping_default",
            Linkage::Public,
            &[Type::I32],
            Type::I32,
        );
        let fr = mb.declare_function(sig).unwrap();
        let mut fb = mb.func_builder::<InstInserter>(fr);
        let entry = fb.append_block();
        let merge = fb.append_block();
        let trap = fb.append_block();
        let check_one = fb.append_block();
        let check_two = fb.append_block();
        let check_three = fb.append_block();

        let selector = fb.args()[0];
        let zero = fb.make_imm_value(0i32);
        let one = fb.make_imm_value(1i32);
        let two = fb.make_imm_value(2i32);
        let three = fb.make_imm_value(3i32);
        let fallback = fb.make_imm_value(99i32);

        fb.switch_to_block(entry);
        let is_zero = fb.insert_inst(cmp::Eq::new(is, selector, zero), Type::I1);
        fb.insert_inst_no_result(Br::new(is, is_zero, merge, check_one));
        fb.switch_to_block(check_one);
        let is_one = fb.insert_inst(cmp::Eq::new(is, selector, one), Type::I1);
        fb.insert_inst_no_result(Br::new(is, is_one, merge, check_two));
        fb.switch_to_block(check_two);
        let is_two = fb.insert_inst(cmp::Eq::new(is, selector, two), Type::I1);
        fb.insert_inst_no_result(Br::new(is, is_two, merge, check_three));
        fb.switch_to_block(check_three);
        let is_three = fb.insert_inst(cmp::Eq::new(is, selector, three), Type::I1);
        fb.insert_inst_no_result(Br::new(is, is_three, merge, trap));
        fb.switch_to_block(merge);
        let result = fb.insert_inst(
            Phi::new(
                is,
                vec![
                    (zero, entry),
                    (one, check_one),
                    (two, check_two),
                    (fallback, check_three),
                ],
            ),
            Type::I32,
        );
        fb.insert_inst_no_result(Return::new_single(is, result));
        fb.switch_to_block(trap);
        fb.insert_inst_no_result(Unreachable::new(is));
        fb.seal_all();
        fb.finish();

        let module = mb.build();
        let structured = structurize(&module, fr);
        assert!(matches!(
            structured.regions.last(),
            Some(Region::Block(block)) if *block == merge
        ));
        assert_eq!(
            structured.stats().block_occurrences,
            structured.stats().referenced_blocks,
            "the direct phi-bearing merge must have one owner: {:?}",
            structured.regions,
        );
    }

    /// A lowered match arm may contain nested validation guards that share the
    /// default trap. The trap is not an inner merge: every live path from the
    /// guarded arm still belongs to the enclosing phi-bearing return merge.
    #[test]
    fn guarded_match_arm_retains_enclosing_phi_return_merge() {
        let (mb, is) = native_builder();
        let sig = Signature::new_single(
            "guarded_match_with_trapping_default",
            Linkage::Public,
            &[Type::I32, Type::I1, Type::I1],
            Type::I32,
        );
        let fr = mb.declare_function(sig).unwrap();
        let mut fb = mb.func_builder::<InstInserter>(fr);
        let entry = fb.append_block();
        let first_arm = fb.append_block();
        let fallback = fb.append_block();
        let first_guard = fb.append_block();
        let second_guard = fb.append_block();
        let second_arm = fb.append_block();
        let merge = fb.append_block();
        let trap = fb.append_block();

        let selector = fb.args()[0];
        let first_ok = fb.args()[1];
        let second_ok = fb.args()[2];
        let zero = fb.make_imm_value(0i32);
        let one = fb.make_imm_value(1i32);

        fb.switch_to_block(entry);
        let is_zero = fb.insert_inst(cmp::Eq::new(is, selector, zero), Type::I1);
        fb.insert_inst_no_result(Br::new(is, is_zero, first_arm, fallback));
        fb.switch_to_block(first_arm);
        fb.insert_inst_no_result(Jump::new(is, merge));
        fb.switch_to_block(fallback);
        fb.insert_inst_no_result(Br::new(is, first_ok, first_guard, trap));
        fb.switch_to_block(first_guard);
        fb.insert_inst_no_result(Br::new(is, second_ok, second_guard, trap));
        fb.switch_to_block(second_guard);
        fb.insert_inst_no_result(Jump::new(is, second_arm));
        fb.switch_to_block(second_arm);
        fb.insert_inst_no_result(Jump::new(is, merge));
        fb.switch_to_block(merge);
        let result = fb.insert_inst(
            Phi::new(is, vec![(zero, first_arm), (one, second_arm)]),
            Type::I32,
        );
        fb.insert_inst_no_result(Return::new_single(is, result));
        fb.switch_to_block(trap);
        fb.insert_inst_no_result(Unreachable::new(is));
        fb.seal_all();
        fb.finish();

        fn block_occurrences(regions: &[Region], wanted: BlockId) -> usize {
            regions
                .iter()
                .map(|region| match region {
                    Region::Block(block) if *block == wanted => 1,
                    Region::IfThenElse {
                        then_branch,
                        else_branch,
                        ..
                    } => {
                        block_occurrences(then_branch, wanted)
                            + block_occurrences(else_branch, wanted)
                    }
                    Region::Loop { body, .. } => block_occurrences(body, wanted),
                    _ => 0,
                })
                .sum()
        }

        let module = mb.build();
        let structured = structurize(&module, fr);
        assert!(matches!(
            structured.regions.last(),
            Some(Region::Block(block)) if *block == merge
        ));
        assert_eq!(
            block_occurrences(&structured.regions, merge),
            1,
            "the enclosing phi-bearing merge must have one owner: {:?}",
            structured.regions,
        );
    }

    /// Nested terminal arms may share one phi-free cleanup block before its
    /// bare return while a sibling terminates independently.
    #[test]
    fn nested_terminal_arms_may_share_cleanup_forwarder() {
        let (mb, is) = native_builder();
        let sig = Signature::new_unit(
            "shared_terminal_cleanup",
            Linkage::Public,
            &[Type::I1, Type::I1, Type::I32],
        );
        let fr = mb.declare_function(sig).unwrap();
        let mut fb = mb.func_builder::<InstInserter>(fr);
        let outer = fb.append_block();
        let nested = fb.append_block();
        let body = fb.append_block();
        let cleanup = fb.append_block();
        let return_block = fb.append_block();
        let other_return = fb.append_block();

        let outer_cond = fb.args()[0];
        let nested_cond = fb.args()[1];
        let value = fb.args()[2];
        fb.switch_to_block(outer);
        fb.insert_inst_no_result(Br::new(is, outer_cond, nested, cleanup));
        fb.switch_to_block(nested);
        fb.insert_inst_no_result(Br::new(is, nested_cond, body, cleanup));
        fb.switch_to_block(body);
        fb.insert_inst_no_result(Jump::new(is, other_return));
        fb.switch_to_block(cleanup);
        fb.insert_inst(arith::Add::new(is, value, value), Type::I32);
        fb.insert_inst_no_result(Jump::new(is, return_block));
        fb.switch_to_block(return_block);
        fb.insert_inst_no_result(Return::new_unit(is));
        fb.switch_to_block(other_return);
        fb.insert_inst_no_result(Return::new_unit(is));
        fb.seal_all();
        fb.finish();

        let module = mb.build();
        let structured = structurize(&module, fr);
        assert!(structured.block_order.contains(&cleanup));
        assert!(structured.block_order.contains(&return_block));

        fn count_cleanup_terminal_pairs(
            regions: &[Region],
            cleanup: BlockId,
            terminal: BlockId,
        ) -> usize {
            let local = regions
                .windows(2)
                .filter(|pair| {
                    matches!(pair[0], Region::Block(block) if block == cleanup)
                        && matches!(pair[1], Region::Block(block) if block == terminal)
                })
                .count();
            local
                + regions
                    .iter()
                    .map(|region| match region {
                        Region::IfThenElse {
                            then_branch,
                            else_branch,
                            ..
                        } => {
                            count_cleanup_terminal_pairs(then_branch, cleanup, terminal)
                                + count_cleanup_terminal_pairs(else_branch, cleanup, terminal)
                        }
                        Region::Loop { body, .. } => {
                            count_cleanup_terminal_pairs(body, cleanup, terminal)
                        }
                        Region::Block(_)
                        | Region::LoopExit { .. }
                        | Region::LoopContinue { .. } => 0,
                    })
                    .sum::<usize>()
        }

        assert_eq!(
            count_cleanup_terminal_pairs(&structured.regions, cleanup, return_block),
            2,
        );
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

    /// Two nested selections may share a nonempty false arm before their
    /// common merge. Structured targets represent the selections as a tree,
    /// so the shared arm must be cloned into both mutually exclusive paths.
    #[test]
    fn nested_and_outer_arms_may_share_nonempty_forwarder() {
        let (mb, is) = native_builder();
        let sig = Signature::new_single(
            "shared_nonempty_forwarder",
            Linkage::Public,
            &[Type::I1, Type::I32],
            Type::I32,
        );
        let fr = mb.declare_function(sig).unwrap();
        let mut fb = mb.func_builder::<InstInserter>(fr);
        let outer = fb.append_block();
        let nested = fb.append_block();
        let body = fb.append_block();
        let shared_forwarder = fb.append_block();
        let merge = fb.append_block();

        let cond = fb.args()[0];
        let value = fb.args()[1];
        fb.switch_to_block(outer);
        fb.insert_inst_no_result(Br::new(is, cond, nested, shared_forwarder));
        fb.switch_to_block(nested);
        fb.insert_inst_no_result(Br::new(is, cond, body, shared_forwarder));
        fb.switch_to_block(body);
        let body_value = fb.insert_inst(arith::Add::new(is, value, value), Type::I32);
        fb.insert_inst_no_result(Jump::new(is, merge));
        fb.switch_to_block(shared_forwarder);
        let shared_value = fb.insert_inst(arith::Sub::new(is, value, value), Type::I32);
        fb.insert_inst_no_result(Jump::new(is, merge));
        fb.switch_to_block(merge);
        let result = fb.insert_inst(
            Phi::new(
                is,
                vec![(body_value, body), (shared_value, shared_forwarder)],
            ),
            Type::I32,
        );
        fb.insert_inst_no_result(Return::new_single(is, result));
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
                    [Region::IfThenElse {
                        else_branch: inner_else,
                        merge: Some(inner_merge),
                        ..
                    }] if *inner_merge == merge
                        && matches!(inner_else.as_slice(), [Region::Block(block)] if *block == shared_forwarder)
                ));
            }
            other => panic!("expected outer IfThenElse, got {other:?}"),
        }
    }

    /// Two nested selections may enter the same phi-free decision corridor.
    /// The corridor and its leaves are cloned into both mutually exclusive
    /// tree positions, while their common merge remains a sibling.
    #[test]
    fn nested_and_outer_arms_may_share_decision_corridor() {
        let (mb, is) = native_builder();
        let sig = Signature::new_unit(
            "shared_decision_corridor",
            Linkage::Public,
            &[Type::I1, Type::I1, Type::I1, Type::I32],
        );
        let fr = mb.declare_function(sig).unwrap();
        let mut fb = mb.func_builder::<InstInserter>(fr);
        let outer = fb.append_block();
        let nested = fb.append_block();
        let body = fb.append_block();
        let shared_decision = fb.append_block();
        let left = fb.append_block();
        let right = fb.append_block();
        let corridor_merge = fb.append_block();
        let corridor_live = fb.append_block();
        let merge = fb.append_block();
        let trap = fb.append_block();

        let outer_cond = fb.args()[0];
        let nested_cond = fb.args()[1];
        let shared_cond = fb.args()[2];
        let value = fb.args()[3];
        fb.switch_to_block(outer);
        fb.insert_inst_no_result(Br::new(is, outer_cond, nested, shared_decision));
        fb.switch_to_block(nested);
        fb.insert_inst_no_result(Br::new(is, nested_cond, body, shared_decision));
        fb.switch_to_block(body);
        fb.insert_inst(arith::Add::new(is, value, value), Type::I32);
        fb.insert_inst_no_result(Jump::new(is, merge));
        fb.switch_to_block(shared_decision);
        fb.insert_inst_no_result(Br::new(is, shared_cond, left, right));
        fb.switch_to_block(left);
        let left_value = fb.insert_inst(arith::Add::new(is, value, value), Type::I32);
        fb.insert_inst_no_result(Jump::new(is, corridor_merge));
        fb.switch_to_block(right);
        let right_value = fb.insert_inst(arith::Sub::new(is, value, value), Type::I32);
        fb.insert_inst_no_result(Jump::new(is, corridor_merge));
        fb.switch_to_block(corridor_merge);
        let selected_value = fb.insert_inst(
            Phi::new(is, vec![(left_value, left), (right_value, right)]),
            Type::I32,
        );
        let remains_live = fb.insert_inst(cmp::Lt::new(is, selected_value, value), Type::I1);
        fb.insert_inst_no_result(Br::new(is, remains_live, corridor_live, trap));
        fb.switch_to_block(corridor_live);
        fb.insert_inst_no_result(Jump::new(is, merge));
        fb.switch_to_block(merge);
        fb.insert_inst_no_result(Return::new_unit(is));
        fb.switch_to_block(trap);
        fb.insert_inst_no_result(Unreachable::new(is));
        fb.seal_all();
        fb.finish();

        let module = mb.build();
        let structured = structurize(&module, fr);
        fn count_headers(regions: &[Region], target: BlockId) -> usize {
            regions
                .iter()
                .map(|region| match region {
                    Region::IfThenElse {
                        header,
                        then_branch,
                        else_branch,
                        ..
                    } => {
                        usize::from(*header == target)
                            + count_headers(then_branch, target)
                            + count_headers(else_branch, target)
                    }
                    Region::Loop { body, .. } => count_headers(body, target),
                    Region::Block(_) | Region::LoopExit { .. } | Region::LoopContinue { .. } => 0,
                })
                .sum()
        }

        assert_eq!(count_headers(&structured.regions, shared_decision), 2);
        assert_eq!(count_headers(&structured.regions, corridor_merge), 2);
        let stats = structured.stats();
        assert!(stats.block_occurrences > stats.referenced_blocks);
        assert!(stats.duplicated_block_occurrences >= 2);
    }

    /// Mutually exclusive loops may fall through to the same nonempty block
    /// before an enclosing merge. The forwarder is cloned at each tree
    /// position just like a directly selected shared arm.
    #[test]
    fn nested_loop_fallthroughs_may_share_nonempty_forwarder() {
        let (mb, is) = native_builder();
        let sig = Signature::new_unit(
            "shared_loop_forwarder",
            Linkage::Public,
            &[Type::I1, Type::I1, Type::I1, Type::I1, Type::I32],
        );
        let fr = mb.declare_function(sig).unwrap();
        let mut fb = mb.func_builder::<InstInserter>(fr);
        let outer = fb.append_block();
        let nested = fb.append_block();
        let left_guard = fb.append_block();
        let right_guard = fb.append_block();
        let left_loop = fb.append_block();
        let right_loop = fb.append_block();
        let shared_forwarder = fb.append_block();
        let merge = fb.append_block();
        let trap = fb.append_block();
        let outer_cond = fb.args()[0];
        let nested_cond = fb.args()[1];
        let left_cond = fb.args()[2];
        let right_cond = fb.args()[3];
        let value = fb.args()[4];

        fb.switch_to_block(outer);
        fb.insert_inst_no_result(Br::new(is, outer_cond, nested, merge));
        fb.switch_to_block(nested);
        fb.insert_inst_no_result(Br::new(is, nested_cond, left_guard, right_guard));
        fb.switch_to_block(left_guard);
        fb.insert_inst_no_result(Br::new(is, left_cond, left_loop, trap));
        fb.switch_to_block(right_guard);
        fb.insert_inst_no_result(Br::new(is, right_cond, right_loop, trap));
        fb.switch_to_block(left_loop);
        fb.insert_inst_no_result(Br::new(is, left_cond, left_loop, shared_forwarder));
        fb.switch_to_block(right_loop);
        fb.insert_inst_no_result(Br::new(is, right_cond, right_loop, shared_forwarder));
        fb.switch_to_block(shared_forwarder);
        fb.insert_inst(arith::Add::new(is, value, value), Type::I32);
        fb.insert_inst_no_result(Jump::new(is, merge));
        fb.switch_to_block(merge);
        fb.insert_inst_no_result(Return::new_unit(is));
        fb.switch_to_block(trap);
        fb.insert_inst_no_result(Unreachable::new(is));
        fb.seal_all();
        fb.finish();

        let module = mb.build();
        let structured = structurize(&module, fr);
        assert!(structured.block_order.contains(&shared_forwarder));
        assert!(structured.block_order.contains(&merge));
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

    /// Guarded sibling paths can terminate or converge on the same nested
    /// loop header. Terminal paths do not strictly post-dominate, but the
    /// header is still the unique continuation of every live path and must be
    /// emitted once as a sibling region.
    #[test]
    fn structurize_guarded_siblings_merge_at_nested_loop_header() {
        let (mb, is) = native_builder();
        let sig = Signature::new_unit("guarded_nested_loop", Linkage::Public, &[Type::I1]);
        let fr = mb.declare_function(sig).unwrap();
        let mut fb = mb.func_builder::<InstInserter>(fr);
        let entry = fb.append_block();
        let left_guard = fb.append_block();
        let right_guard = fb.append_block();
        let left_path = fb.append_block();
        let right_path = fb.append_block();
        let loop_header = fb.append_block();
        let loop_body = fb.append_block();
        let trap = fb.append_block();
        let exit = fb.append_block();

        fb.switch_to_block(entry);
        let cond = fb.args()[0];
        fb.insert_inst_no_result(Br::new(is, cond, left_guard, right_guard));
        fb.switch_to_block(left_guard);
        fb.insert_inst_no_result(Br::new(is, cond, trap, left_path));
        fb.switch_to_block(right_guard);
        fb.insert_inst_no_result(Br::new(is, cond, trap, right_path));
        fb.switch_to_block(left_path);
        fb.insert_inst_no_result(Jump::new(is, loop_header));
        fb.switch_to_block(right_path);
        fb.insert_inst_no_result(Jump::new(is, loop_header));
        fb.switch_to_block(loop_header);
        fb.insert_inst_no_result(Br::new(is, cond, loop_body, exit));
        fb.switch_to_block(loop_body);
        fb.insert_inst_no_result(Jump::new(is, loop_header));
        fb.switch_to_block(trap);
        fb.insert_inst_no_result(Unreachable::new(is));
        fb.switch_to_block(exit);
        fb.insert_inst_no_result(Return::new_unit(is));
        fb.seal_all();
        fb.finish();

        let module = mb.build();
        let structured = structurize(&module, fr);
        assert_eq!(
            structured
                .regions
                .iter()
                .filter(|region| matches!(
                    region,
                    Region::Loop { header, .. } if *header == loop_header
                ))
                .count(),
            1,
            "nested loop header must be emitted exactly once: {:?}",
            structured.regions,
        );
    }

    /// A guarded path may cross a complete nested loop before it reaches the
    /// loop header shared with its sibling. Backedges inside that corridor do
    /// not escape the continuation and must not invalidate the shared merge.
    #[test]
    fn structurize_guarded_nested_loop_corridor_reaches_shared_loop() {
        let (mb, is) = native_builder();
        let sig = Signature::new_unit("guarded_loop_corridor", Linkage::Public, &[Type::I1]);
        let fr = mb.declare_function(sig).unwrap();
        let mut fb = mb.func_builder::<InstInserter>(fr);
        let entry = fb.append_block();
        let direct = fb.append_block();
        let guarded = fb.append_block();
        let inner_header = fb.append_block();
        let inner_body = fb.append_block();
        let inner_exit = fb.append_block();
        let shared_header = fb.append_block();
        let shared_body = fb.append_block();
        let trap = fb.append_block();
        let exit = fb.append_block();

        fb.switch_to_block(entry);
        let cond = fb.args()[0];
        fb.insert_inst_no_result(Br::new(is, cond, direct, guarded));
        fb.switch_to_block(direct);
        fb.insert_inst_no_result(Jump::new(is, shared_header));
        fb.switch_to_block(guarded);
        fb.insert_inst_no_result(Br::new(is, cond, trap, inner_header));
        fb.switch_to_block(inner_header);
        fb.insert_inst_no_result(Br::new(is, cond, inner_body, inner_exit));
        fb.switch_to_block(inner_body);
        fb.insert_inst_no_result(Jump::new(is, inner_header));
        fb.switch_to_block(inner_exit);
        fb.insert_inst_no_result(Jump::new(is, shared_header));
        fb.switch_to_block(shared_header);
        fb.insert_inst_no_result(Br::new(is, cond, shared_body, exit));
        fb.switch_to_block(shared_body);
        fb.insert_inst_no_result(Jump::new(is, shared_header));
        fb.switch_to_block(trap);
        fb.insert_inst_no_result(Unreachable::new(is));
        fb.switch_to_block(exit);
        fb.insert_inst_no_result(Return::new_unit(is));
        fb.seal_all();
        fb.finish();

        let module = mb.build();
        let structured = structurize(&module, fr);
        assert_eq!(
            structured
                .regions
                .iter()
                .filter(|region| matches!(
                    region,
                    Region::Loop { header, .. } if *header == shared_header
                ))
                .count(),
            1,
            "shared loop header must be emitted exactly once: {:?}",
            structured.regions,
        );
    }

    /// A candidate forwarder is not a merge when one live path bypasses it and
    /// rejoins the candidate's downstream continuation. An unrelated terminal
    /// arm removes strict post-dominance, so the terminal-aware search must
    /// still choose the downstream join rather than the skipped forwarder.
    #[test]
    fn structurize_rejects_bypassed_forwarder_merge() {
        let (mb, is) = native_builder();
        let sig = Signature::new_unit("bypassed_forwarder", Linkage::Public, &[Type::I1]);
        let fr = mb.declare_function(sig).unwrap();
        let mut fb = mb.func_builder::<InstInserter>(fr);
        let entry = fb.append_block();
        let guarded = fb.append_block();
        let decision = fb.append_block();
        let forwarder = fb.append_block();
        let merge = fb.append_block();
        let exit = fb.append_block();
        let trap = fb.append_block();

        fb.switch_to_block(entry);
        let cond = fb.args()[0];
        fb.insert_inst_no_result(Br::new(is, cond, guarded, forwarder));
        fb.switch_to_block(guarded);
        fb.insert_inst_no_result(Br::new(is, cond, trap, decision));
        fb.switch_to_block(decision);
        fb.insert_inst_no_result(Br::new(is, cond, forwarder, merge));
        fb.switch_to_block(forwarder);
        fb.insert_inst_no_result(Jump::new(is, merge));
        fb.switch_to_block(merge);
        fb.insert_inst_no_result(Jump::new(is, exit));
        fb.switch_to_block(exit);
        fb.insert_inst_no_result(Return::new_unit(is));
        fb.switch_to_block(trap);
        fb.insert_inst_no_result(Unreachable::new(is));
        fb.seal_all();
        fb.finish();

        let module = mb.build();
        let structured = structurize(&module, fr);
        assert!(matches!(
            structured.regions.first(),
            Some(Region::IfThenElse { merge: Some(found), .. }) if *found == merge
        ));
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
    fn conditional_loop_exit_corridor_reaches_canonical_break() {
        let (mb, is) = native_builder();
        let sig = Signature::new_single(
            "conditional_loop_exit_corridor",
            Linkage::Public,
            &[],
            Type::I32,
        );
        let fr = mb.declare_function(sig).unwrap();
        let mut fb = mb.func_builder::<InstInserter>(fr);
        let entry = fb.append_block();
        let header = fb.append_block();
        let body = fb.append_block();
        let latch = fb.append_block();
        let canonical_exit = fb.append_block();
        let corridor_decision = fb.append_block();
        let corridor_value = fb.append_block();
        let corridor_merge = fb.append_block();

        fb.switch_to_block(entry);
        let cond = fb.make_imm_value(true);
        let header_value = fb.make_imm_value(7i32);
        fb.insert_inst_no_result(Jump::new(is, header));
        fb.switch_to_block(header);
        fb.insert_inst_no_result(Br::new(is, cond, body, canonical_exit));
        fb.switch_to_block(body);
        fb.insert_inst_no_result(Br::new(is, cond, corridor_decision, latch));
        fb.switch_to_block(latch);
        fb.insert_inst_no_result(Jump::new(is, header));
        fb.switch_to_block(corridor_decision);
        let direct_value = fb.make_imm_value(11i32);
        fb.insert_inst_no_result(Br::new(is, cond, corridor_value, corridor_merge));
        fb.switch_to_block(corridor_value);
        let arm_value = fb.make_imm_value(22i32);
        fb.insert_inst_no_result(Jump::new(is, corridor_merge));
        fb.switch_to_block(corridor_merge);
        let selected = fb.insert_inst(
            Phi::new(
                is,
                vec![
                    (direct_value, corridor_decision),
                    (arm_value, corridor_value),
                ],
            ),
            Type::I32,
        );
        fb.switch_to_block(canonical_exit);
        let result = fb.insert_inst(
            Phi::new(
                is,
                vec![(header_value, header), (selected, corridor_merge)],
            ),
            Type::I32,
        );
        fb.insert_inst_no_result(Return::new_single(is, result));
        fb.switch_to_block(corridor_merge);
        fb.insert_inst_no_result(Jump::new(is, canonical_exit));
        fb.seal_all();
        fb.finish();

        let module = mb.build();
        let structured = module
            .func_store
            .view(fr, |func| structurize_function(func))
            .expect("an outside-SCC corridor to the canonical break should structure");

        fn contains_exit(regions: &[Region], from: BlockId, target: BlockId) -> bool {
            regions.iter().any(|region| match region {
                Region::LoopExit {
                    from: actual_from,
                    target: actual_target,
                } => *actual_from == from && *actual_target == target,
                Region::Loop { body, .. } => contains_exit(body, from, target),
                Region::IfThenElse {
                    then_branch,
                    else_branch,
                    ..
                } => {
                    contains_exit(then_branch, from, target)
                        || contains_exit(else_branch, from, target)
                }
                Region::Block(_) | Region::LoopContinue { .. } => false,
            })
        }
        assert!(
            contains_exit(&structured.regions, corridor_merge, canonical_exit),
            "the conditional phi corridor must retain the exact exit predecessor: {:?}",
            structured.regions,
        );
    }

    #[test]
    fn mixed_break_and_return_corridor_fails_closed() {
        let (mb, is) = native_builder();
        let sig = Signature::new_unit("mixed_break_return", Linkage::Public, &[]);
        let fr = mb.declare_function(sig).unwrap();
        let mut fb = mb.func_builder::<InstInserter>(fr);
        let entry = fb.append_block();
        let header = fb.append_block();
        let body = fb.append_block();
        let latch = fb.append_block();
        let canonical_exit = fb.append_block();
        let mixed_exit = fb.append_block();
        let early_return = fb.append_block();

        fb.switch_to_block(entry);
        let cond = fb.make_imm_value(true);
        fb.insert_inst_no_result(Jump::new(is, header));
        fb.switch_to_block(header);
        fb.insert_inst_no_result(Br::new(is, cond, body, canonical_exit));
        fb.switch_to_block(body);
        fb.insert_inst_no_result(Br::new(is, cond, mixed_exit, latch));
        fb.switch_to_block(latch);
        fb.insert_inst_no_result(Jump::new(is, header));
        fb.switch_to_block(mixed_exit);
        fb.insert_inst_no_result(Br::new(is, cond, canonical_exit, early_return));
        fb.switch_to_block(canonical_exit);
        fb.insert_inst_no_result(Return::new_unit(is));
        fb.switch_to_block(early_return);
        fb.insert_inst_no_result(Return::new_unit(is));
        fb.seal_all();
        fb.finish();

        let module = mb.build();
        let error = module
            .func_store
            .view(fr, |func| structurize_function(func))
            .expect_err("a mixed break/return corridor must remain fail-closed");
        assert!(
            error.contains("noncanonical exit") && error.contains("expected the header exit"),
            "expected a named noncanonical-exit error, got: {error}",
        );
    }

    #[test]
    fn nested_loop_exit_corridor_reaches_outer_canonical_break() {
        let (mb, is) = native_builder();
        let sig = Signature::new_single(
            "nested_loop_exit_corridor",
            Linkage::Public,
            &[],
            Type::I32,
        );
        let fr = mb.declare_function(sig).unwrap();
        let mut fb = mb.func_builder::<InstInserter>(fr);
        let entry = fb.append_block();
        let outer_header = fb.append_block();
        let outer_body = fb.append_block();
        let outer_latch = fb.append_block();
        let canonical_exit = fb.append_block();
        let corridor_entry = fb.append_block();
        let inner_header = fb.append_block();
        let inner_body = fb.append_block();
        let corridor_exit = fb.append_block();

        fb.switch_to_block(entry);
        let cond = fb.make_imm_value(true);
        let zero = fb.make_imm_value(0i32);
        let two = fb.make_imm_value(2i32);
        let header_value = fb.make_imm_value(7i32);
        fb.insert_inst_no_result(Jump::new(is, outer_header));
        fb.switch_to_block(outer_header);
        fb.insert_inst_no_result(Br::new(is, cond, outer_body, canonical_exit));
        fb.switch_to_block(outer_body);
        fb.insert_inst_no_result(Br::new(is, cond, corridor_entry, outer_latch));
        fb.switch_to_block(outer_latch);
        fb.insert_inst_no_result(Jump::new(is, outer_header));
        fb.switch_to_block(corridor_entry);
        fb.insert_inst_no_result(Jump::new(is, inner_header));
        fb.switch_to_block(inner_header);
        let i = fb.insert_inst(Phi::new(is, vec![(zero, corridor_entry)]), Type::I32);
        let inner_cond = fb.insert_inst(cmp::Lt::new(is, i, two), Type::I1);
        fb.insert_inst_no_result(Br::new(is, inner_cond, inner_body, corridor_exit));
        fb.switch_to_block(inner_body);
        let one = fb.make_imm_value(1i32);
        let next_i = fb.insert_inst(arith::Add::new(is, i, one), Type::I32);
        fb.append_phi_arg(i, next_i, inner_body);
        fb.insert_inst_no_result(Jump::new(is, inner_header));
        fb.switch_to_block(corridor_exit);
        let selected = fb.insert_inst(arith::Add::new(is, i, one), Type::I32);
        fb.insert_inst_no_result(Jump::new(is, canonical_exit));
        fb.switch_to_block(canonical_exit);
        let result = fb.insert_inst(
            Phi::new(
                is,
                vec![(header_value, outer_header), (selected, corridor_exit)],
            ),
            Type::I32,
        );
        fb.insert_inst_no_result(Return::new_single(is, result));
        fb.seal_all();
        fb.finish();

        let module = mb.build();
        let structured = module
            .func_store
            .view(fr, |func| structurize_function(func))
            .expect("a reducible nested loop may compute an outer break value");

        fn contains_loop_and_exit(
            regions: &[Region],
            inner_header: BlockId,
            from: BlockId,
            target: BlockId,
        ) -> (bool, bool) {
            regions.iter().fold((false, false), |found, region| {
                let nested = match region {
                    Region::Loop { header, body } => {
                        let child = contains_loop_and_exit(body, inner_header, from, target);
                        (*header == inner_header || child.0, child.1)
                    }
                    Region::IfThenElse {
                        then_branch,
                        else_branch,
                        ..
                    } => {
                        let then_found =
                            contains_loop_and_exit(then_branch, inner_header, from, target);
                        let else_found =
                            contains_loop_and_exit(else_branch, inner_header, from, target);
                        (then_found.0 || else_found.0, then_found.1 || else_found.1)
                    }
                    Region::LoopExit {
                        from: actual_from,
                        target: actual_target,
                    } => (false, *actual_from == from && *actual_target == target),
                    Region::Block(_) | Region::LoopContinue { .. } => (false, false),
                };
                (found.0 || nested.0, found.1 || nested.1)
            })
        }

        let found = contains_loop_and_exit(
            &structured.regions,
            inner_header,
            corridor_exit,
            canonical_exit,
        );
        assert!(
            found == (true, true),
            "expected the nested loop and exact outer break edge: {:?}",
            structured.regions,
        );
    }
}
