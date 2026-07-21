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

use std::collections::HashSet;

use sonatina_ir::{
    BlockId, Function, InstDowncast, InstSetBase,
    cfg::ControlFlowGraph,
    inst::control_flow::{Br, BrTable, Jump, Return},
};

use crate::{
    domtree::{DomTree, DominatorTreeTraversable},
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

    let mut dom_children = DominatorTreeTraversable::default();
    dom_children.compute(&domtree);

    let rpo = domtree.rpo().to_vec();

    if rpo.is_empty() {
        return Ok(StructuredCfg {
            regions: Vec::new(),
            block_order: Vec::new(),
        });
    }

    let s = Structurer {
        function,
        is: function.inst_set(),
        cfg: &cfg,
        dom_children: &dom_children,
        loop_tree: &loop_tree,
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
    Other,
}

struct Structurer<'a> {
    function: &'a Function,
    is: &'a dyn InstSetBase,
    cfg: &'a ControlFlowGraph,
    dom_children: &'a DominatorTreeTraversable,
    loop_tree: &'a LoopTree,
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
            if <&BrTable as InstDowncast>::downcast(self.is, d).is_some() {
                return Term::Other;
            }
        }
        Term::Other
    }

    fn returns(&self, b: BlockId) -> bool {
        matches!(self.term(b), Term::Return)
    }

    fn in_loop(&self, b: BlockId, lp: Loop) -> bool {
        self.loop_tree.is_in_loop(b, lp)
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
                if !self.in_loop(b, lp) && b != start && !self.returns(b) {
                    break;
                }
            }

            if active.contains(&b) || !consumed.insert(b) {
                return Err(format!(
                    "spirv structurize: cyclic or multiply consumed block {b:?}"
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
                    cur = Some(t);
                }
                Term::Return => {
                    regions.push(Region::Block(b));
                    cur = None;
                }
                Term::Br(nz, z) => {
                    let merge = self.find_merge(b, cur_loop)?;
                    let then_branch = if Some(nz) == merge {
                        Vec::new()
                    } else {
                        self.build_seq(nz, merge, cur_loop, active, consumed)?
                    };
                    let else_branch = if Some(z) == merge {
                        Vec::new()
                    } else {
                        self.build_seq(z, merge, cur_loop, active, consumed)?
                    };
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
    ) -> Result<Option<BlockId>, String> {
        let loop_hdr = cur_loop.map(|lp| self.loop_tree.loop_header(lp));
        let joins: Vec<BlockId> = self
            .dom_children
            .children_of(header)
            .iter()
            .copied()
            .filter(|&c| self.cfg.pred_num_of(c) >= 2 && Some(c) != loop_hdr)
            .collect();
        match joins.len() {
            0 => Ok(None),
            1 => Ok(Some(joins[0])),
            _ => Err(format!(
                "spirv structurize: {} merge candidates dominated by block {header:?} \
                 (unstructured/unsupported control-flow shape)",
                joins.len()
            )),
        }
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
            control_flow::{Br, Jump, Phi, Return},
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
                || err.contains("unstructured/unsupported"),
            "expected a named fail-closed structurizer error, got: {err}"
        );
    }
}
