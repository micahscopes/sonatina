use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use sonatina_ir::{
    BlockId, ControlFlowGraph, Function, Immediate, InstId, Type, Value, ValueId,
    inst::{
        arith::Add,
        cmp::{Eq as CmpEq, Ge, Gt, Le, Lt, Ne, Sge, Sgt, Sle, Slt},
        control_flow::{Br, Phi},
        downcast,
    },
    visitor::VisitorMut,
};

use crate::{
    domtree::DomTree,
    loop_analysis::{Loop, LoopTree},
};

/// Maximum number of instructions in the fully unrolled body.
/// Must be large enough for crypto primitives (e.g. Poseidon: 57 rounds × ~8 insts = 456).
const MAX_UNROLLED_INSTS: usize = 1024;

/// Minimum trip count to consider unrolling. Loops smaller than this are
/// better served by loop strength reduction.
const MIN_TRIP_COUNT: usize = 16;

/// Maximum number of simulation steps when computing trip count.
const MAX_TRIP_COUNT: usize = 512;

/// A counted-loop descriptor extracted from IR pattern matching.
#[allow(dead_code)]
struct CountedLoop {
    /// The header block of the loop (contains the phi + comparison + branch).
    header: BlockId,
    /// The phi instruction that defines the induction variable.
    iv_phi_inst: InstId,
    /// The induction variable (result of the phi).
    iv: ValueId,
    /// Initial value of the induction variable (from the preheader).
    init: Immediate,
    /// The constant step added to the IV each iteration.
    step: Immediate,
    /// The limit constant used in the exit comparison.
    limit: Immediate,
    /// The type of the induction variable.
    iv_ty: Type,
    /// The block that the loop exits to (outside the loop).
    exit_block: BlockId,
    /// The block entered when the loop body should execute.
    body_entry: BlockId,
    /// The preheader block (single predecessor of header from outside the loop).
    preheader: BlockId,
    /// Blocks in the loop body in RPO order (header first).
    loop_blocks: Vec<BlockId>,
    /// The computed trip count.
    trip_count: usize,
    /// The comparison function used to decide the exit condition.
    /// `cmp_fn(iv_value, limit)` returns true when the loop should *continue*.
    cmp_fn: fn(Immediate, Immediate) -> bool,
}

pub struct LoopUnrollSolver {
    _private: (),
}

impl LoopUnrollSolver {
    pub fn new() -> Self {
        Self { _private: () }
    }

    pub fn run(
        &mut self,
        func: &mut Function,
        cfg: &mut ControlFlowGraph,
        lpt: &mut LoopTree,
    ) {
        cfg.compute(func);
        let mut domtree = DomTree::new();
        domtree.compute(cfg);
        lpt.compute(cfg, &domtree);

        // Collect loops to unroll. Process innermost first (reverse iteration
        // gives inner-before-outer since LoopTree stores outer loops before inner).
        let loops: Vec<Loop> = lpt.loops().collect();
        for lp in loops.into_iter().rev() {
            // Skip loops with nested children.
            if has_nested_loops(lpt, lp) {
                continue;
            }

            if let Some(counted) = analyze_counted_loop(func, cfg, lpt, lp) {
                unroll_loop(func, &counted);
                // Recompute analyses after mutation.
                cfg.compute(func);
                domtree.compute(cfg);
                lpt.compute(cfg, &domtree);
            }
        }
    }
}

impl Default for LoopUnrollSolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Returns true if `lp` has any child (nested) loops.
fn has_nested_loops(lpt: &LoopTree, lp: Loop) -> bool {
    for other in lpt.loops() {
        if other != lp && lpt.parent_loop(other) == Some(lp) {
            return true;
        }
    }
    false
}

/// Pattern-match IR to detect a simple counted loop.
///
/// Expected pattern:
/// ```text
/// preheader:
///     jump header
///
/// header:
///     v_ind = phi (v_init, preheader) (v_next, latch)
///     v_cmp = <cmp> v_ind, v_limit
///     br v_cmp, body, exit          // nz_dest = body, z_dest = exit
///
/// ...body blocks...
///
/// latch:
///     v_next = add v_ind, v_step
///     jump header
/// ```
fn analyze_counted_loop(
    func: &Function,
    cfg: &ControlFlowGraph,
    lpt: &LoopTree,
    lp: Loop,
) -> Option<CountedLoop> {
    let header = lpt.loop_header(lp);
    let is = func.inst_set();

    // Collect loop blocks in post-order, then reverse for RPO.
    let mut loop_blocks: Vec<BlockId> = lpt.iter_blocks_post_order(cfg, lp).collect();
    loop_blocks.reverse();

    // Find the unique preheader: a single predecessor of header that is not in the loop.
    let preds: Vec<BlockId> = cfg
        .preds_of(header)
        .copied()
        .filter(|b| !lpt.is_in_loop(*b, lp))
        .collect();
    if preds.len() != 1 {
        return None;
    }
    let preheader = preds[0];

    // Find the unique latch: a single predecessor of header that IS in the loop.
    let latches: Vec<BlockId> = cfg
        .preds_of(header)
        .copied()
        .filter(|b| lpt.is_in_loop(*b, lp))
        .collect();
    if latches.len() != 1 {
        return None;
    }
    let _latch = latches[0];

    // Find the phi in the header that serves as the induction variable.
    // We need a phi with exactly two incoming edges: one from preheader, one from latch.
    let mut iv_info = None;
    for inst in func.layout.iter_inst(header) {
        let Some(phi) = downcast::<&Phi>(is, func.dfg.inst(inst)) else {
            continue;
        };
        let args = phi.args();
        if args.len() != 2 {
            continue;
        }
        // Identify which arg comes from preheader and which from latch.
        let (init_val, _init_block, next_val, _next_block) =
            if args[0].1 == preheader && lpt.is_in_loop(args[1].1, lp) {
                (args[0].0, args[0].1, args[1].0, args[1].1)
            } else if args[1].1 == preheader && lpt.is_in_loop(args[0].1, lp) {
                (args[1].0, args[1].1, args[0].0, args[0].1)
            } else {
                continue;
            };

        // The init value must be an immediate constant.
        let Some(init_imm) = func.dfg.value_imm(init_val) else {
            continue;
        };

        // The next value must be defined by an Add instruction of the form:
        //   v_next = add v_ind, v_step
        // where v_step is a constant.
        let iv_result = func.dfg.inst_results(inst);
        if iv_result.is_empty() {
            continue;
        }
        let iv = iv_result[0];
        let iv_ty = func.dfg.value_ty(iv);

        let Some(next_inst) = func.dfg.value_inst(next_val) else {
            continue;
        };
        let Some(add) = downcast::<&Add>(is, func.dfg.inst(next_inst)) else {
            continue;
        };

        // One operand of the add should be the IV, the other a constant step.
        let step_val = if *add.lhs() == iv {
            *add.rhs()
        } else if *add.rhs() == iv {
            *add.lhs()
        } else {
            continue;
        };

        let Some(step_imm) = func.dfg.value_imm(step_val) else {
            continue;
        };

        // Reject zero step (infinite loop).
        if step_imm.is_zero() {
            continue;
        }

        iv_info = Some((inst, iv, init_imm, step_imm, iv_ty));
        break;
    }

    let (iv_phi_inst, iv, init, step, iv_ty) = iv_info?;

    // Find the comparison + branch in the header.
    // The terminator of the header should be a Br instruction.
    let term = func.layout.last_inst_of(header)?;
    let br = downcast::<&Br>(is, func.dfg.inst(term))?;
    let cond_val = *br.cond();
    let nz_dest = *br.nz_dest();
    let z_dest = *br.z_dest();

    // One dest is in the loop (body entry), the other is the exit.
    let in_loop_nz = lpt.is_in_loop(nz_dest, lp);
    let in_loop_z = lpt.is_in_loop(z_dest, lp);

    // Exactly one should be in the loop.
    if in_loop_nz == in_loop_z {
        return None;
    }

    let (body_entry, exit_block, cond_true_continues) = if in_loop_nz {
        (nz_dest, z_dest, true)
    } else {
        (z_dest, nz_dest, false)
    };

    // The condition must come from a comparison of the IV against a constant.
    let cond_inst = func.dfg.value_inst(cond_val)?;
    let cond_data = func.dfg.inst(cond_inst);

    // Extract both the comparison function and the limit constant.
    let (cmp_fn, limit) =
        extract_comparison(is, cond_data, iv, cond_true_continues, func)?;

    // Simulate the trip count.
    let trip_count = simulate_trip_count(init, step, limit, cmp_fn)?;

    // Skip tiny loops — strength reduction handles those better.
    // Exception: zero-trip loops are always worth eliminating.
    if trip_count > 0 && trip_count < MIN_TRIP_COUNT {
        return None;
    }

    // Budget check: trip_count * body_inst_count <= MAX_UNROLLED_INSTS.
    let body_inst_count = count_body_insts(func, &loop_blocks);
    if trip_count * body_inst_count > MAX_UNROLLED_INSTS {
        return None;
    }

    Some(CountedLoop {
        header,
        iv_phi_inst,
        iv,
        init,
        step,
        limit,
        iv_ty,
        exit_block,
        body_entry,
        preheader,
        loop_blocks,
        trip_count,
        cmp_fn,
    })
}

/// Extract the comparison function and the limit constant from a comparison instruction.
///
/// Returns `(cmp_fn, limit)` where `cmp_fn(iv_value, limit)` returns true when
/// the loop should *continue* (body is taken).
fn extract_comparison(
    is: &dyn sonatina_ir::InstSetBase,
    cond_data: &dyn sonatina_ir::Inst,
    iv: ValueId,
    cond_true_continues: bool,
    func: &Function,
) -> Option<(fn(Immediate, Immediate) -> bool, Immediate)> {
    // Try Lt: iv < limit
    if let Some(lt) = downcast::<&Lt>(is, cond_data) {
        if *lt.lhs() == iv {
            let limit = func.dfg.value_imm(*lt.rhs())?;
            let cmp_fn: fn(Immediate, Immediate) -> bool = if cond_true_continues {
                |a, b| !a.lt(b).is_zero()
            } else {
                |a, b| a.lt(b).is_zero()
            };
            return Some((cmp_fn, limit));
        }
        if *lt.rhs() == iv {
            let limit = func.dfg.value_imm(*lt.lhs())?;
            let cmp_fn: fn(Immediate, Immediate) -> bool = if cond_true_continues {
                |a, b| !b.lt(a).is_zero()
            } else {
                |a, b| b.lt(a).is_zero()
            };
            return Some((cmp_fn, limit));
        }
    }

    // Try Slt: signed iv < limit
    if let Some(slt) = downcast::<&Slt>(is, cond_data) {
        if *slt.lhs() == iv {
            let limit = func.dfg.value_imm(*slt.rhs())?;
            let cmp_fn: fn(Immediate, Immediate) -> bool = if cond_true_continues {
                |a, b| !a.slt(b).is_zero()
            } else {
                |a, b| a.slt(b).is_zero()
            };
            return Some((cmp_fn, limit));
        }
        if *slt.rhs() == iv {
            let limit = func.dfg.value_imm(*slt.lhs())?;
            let cmp_fn: fn(Immediate, Immediate) -> bool = if cond_true_continues {
                |a, b| !b.slt(a).is_zero()
            } else {
                |a, b| b.slt(a).is_zero()
            };
            return Some((cmp_fn, limit));
        }
    }

    // Try Le: iv <= limit
    if let Some(le) = downcast::<&Le>(is, cond_data) {
        if *le.lhs() == iv {
            let limit = func.dfg.value_imm(*le.rhs())?;
            let cmp_fn: fn(Immediate, Immediate) -> bool = if cond_true_continues {
                |a, b| !a.le(b).is_zero()
            } else {
                |a, b| a.le(b).is_zero()
            };
            return Some((cmp_fn, limit));
        }
    }

    // Try Sle: signed iv <= limit
    if let Some(sle) = downcast::<&Sle>(is, cond_data) {
        if *sle.lhs() == iv {
            let limit = func.dfg.value_imm(*sle.rhs())?;
            let cmp_fn: fn(Immediate, Immediate) -> bool = if cond_true_continues {
                |a, b| !a.sle(b).is_zero()
            } else {
                |a, b| a.sle(b).is_zero()
            };
            return Some((cmp_fn, limit));
        }
    }

    // Try Gt: iv > limit
    if let Some(gt) = downcast::<&Gt>(is, cond_data) {
        if *gt.lhs() == iv {
            let limit = func.dfg.value_imm(*gt.rhs())?;
            let cmp_fn: fn(Immediate, Immediate) -> bool = if cond_true_continues {
                |a, b| !a.gt(b).is_zero()
            } else {
                |a, b| a.gt(b).is_zero()
            };
            return Some((cmp_fn, limit));
        }
    }

    // Try Sgt
    if let Some(sgt) = downcast::<&Sgt>(is, cond_data) {
        if *sgt.lhs() == iv {
            let limit = func.dfg.value_imm(*sgt.rhs())?;
            let cmp_fn: fn(Immediate, Immediate) -> bool = if cond_true_continues {
                |a, b| !a.sgt(b).is_zero()
            } else {
                |a, b| a.sgt(b).is_zero()
            };
            return Some((cmp_fn, limit));
        }
    }

    // Try Ge
    if let Some(ge) = downcast::<&Ge>(is, cond_data) {
        if *ge.lhs() == iv {
            let limit = func.dfg.value_imm(*ge.rhs())?;
            let cmp_fn: fn(Immediate, Immediate) -> bool = if cond_true_continues {
                |a, b| !a.ge(b).is_zero()
            } else {
                |a, b| a.ge(b).is_zero()
            };
            return Some((cmp_fn, limit));
        }
    }

    // Try Sge
    if let Some(sge) = downcast::<&Sge>(is, cond_data) {
        if *sge.lhs() == iv {
            let limit = func.dfg.value_imm(*sge.rhs())?;
            let cmp_fn: fn(Immediate, Immediate) -> bool = if cond_true_continues {
                |a, b| !a.sge(b).is_zero()
            } else {
                |a, b| a.sge(b).is_zero()
            };
            return Some((cmp_fn, limit));
        }
    }

    // Try Eq: loop while iv == limit (rare, but handle it)
    if let Some(eq) = downcast::<&CmpEq>(is, cond_data) {
        if *eq.lhs() == iv {
            let limit = func.dfg.value_imm(*eq.rhs())?;
            let cmp_fn: fn(Immediate, Immediate) -> bool = if cond_true_continues {
                |a, b| !a.imm_eq(b).is_zero()
            } else {
                |a, b| a.imm_eq(b).is_zero()
            };
            return Some((cmp_fn, limit));
        }
    }

    // Try Ne: loop while iv != limit
    if let Some(ne) = downcast::<&Ne>(is, cond_data) {
        if *ne.lhs() == iv {
            let limit = func.dfg.value_imm(*ne.rhs())?;
            let cmp_fn: fn(Immediate, Immediate) -> bool = if cond_true_continues {
                |a, b| !a.imm_ne(b).is_zero()
            } else {
                |a, b| a.imm_ne(b).is_zero()
            };
            return Some((cmp_fn, limit));
        }
    }

    None
}

/// Simulate the induction variable to compute trip count.
///
/// Starting from `init`, we add `step` each iteration and check `cmp_fn(iv, limit)`.
/// The function returns the number of iterations where `cmp_fn` returns true.
fn simulate_trip_count(
    init: Immediate,
    step: Immediate,
    limit: Immediate,
    cmp_fn: fn(Immediate, Immediate) -> bool,
) -> Option<usize> {
    let mut iv = init;
    let mut count = 0usize;

    loop {
        if !cmp_fn(iv, limit) {
            return Some(count);
        }
        count += 1;
        if count > MAX_TRIP_COUNT {
            return None;
        }
        iv = iv + step;
    }
}

/// Count instructions in the loop body (all blocks).
fn count_body_insts(func: &Function, loop_blocks: &[BlockId]) -> usize {
    let mut count = 0;
    for &block in loop_blocks {
        for _inst in func.layout.iter_inst(block) {
            count += 1;
        }
    }
    count
}

/// Perform the actual unrolling transformation.
///
/// For a loop with `trip_count` iterations, we:
/// 1. Create N copies of the loop body blocks (one per iteration).
/// 2. In each copy, replace the IV with the known constant `init + i * step`.
/// 3. Wire: preheader -> copy_0 entry, copy_i last -> copy_{i+1} entry, copy_N last -> exit.
/// 4. Fix up any phis in the exit block that referenced values from the loop.
fn unroll_loop(func: &mut Function, counted: &CountedLoop) {
    if counted.trip_count == 0 {
        // Zero trips: just wire preheader directly to exit and clean up.
        unroll_zero_trip(func, counted);
        return;
    }

    let header = counted.header;
    let preheader = counted.preheader;
    let exit_block = counted.exit_block;
    let body_entry = counted.body_entry;

    // Collect the body blocks (everything except the header).
    // The header contains the IV phi, the comparison, and the branch — these are
    // replaced by the unrolled copies. The body blocks are the blocks between
    // body_entry and the latch (inclusive).
    let body_blocks: Vec<BlockId> = counted
        .loop_blocks
        .iter()
        .copied()
        .filter(|b| *b != header)
        .collect();

    // If the body is empty (header is the only block and branches to itself),
    // there are no body instructions to clone. Just wire preheader to exit.
    if body_blocks.is_empty() {
        unroll_zero_trip(func, counted);
        return;
    }

    // Collect instructions per body block (excluding terminators of latch blocks
    // that jump back to header — those become jumps to the next copy's entry).
    // Also collect other header phis (non-IV) that need to be resolved.

    // First, identify values defined in the header (phis) that body blocks use.
    // For each header phi, we need to know the preheader value and the latch value.
    let header_phis = collect_header_phis(func, header, preheader, &counted.loop_blocks);

    // Track which values are defined inside the loop.
    let mut loop_defined = rustc_hash::FxHashSet::default();
    for &block in &counted.loop_blocks {
        for inst in func.layout.iter_inst(block) {
            for &result in func.dfg.inst_results(inst) {
                loop_defined.insert(result);
            }
        }
    }

    // For each iteration, create cloned blocks and wire them.
    let mut prev_iteration_values: FxHashMap<ValueId, ValueId> = FxHashMap::default();
    let mut first_entry_block = None;
    let mut insert_after = preheader;

    // Map from exit-phi's loop-defined incoming value to the last iteration's version.
    // We'll update this after each iteration.
    let mut last_iter_value_map: FxHashMap<ValueId, ValueId> = FxHashMap::default();

    for iter_idx in 0..counted.trip_count {
        let iv_const = compute_iv_at(counted.init, counted.step, iter_idx);

        // block_map: old body block -> new cloned block
        let mut block_map: FxHashMap<BlockId, BlockId> = FxHashMap::default();

        // Create new blocks for this iteration's body.
        for &old_block in &body_blocks {
            let new_block = func.dfg.make_block();
            func.layout.insert_block_after(new_block, insert_after);
            insert_after = new_block;
            block_map.insert(old_block, new_block);
        }

        if iter_idx == 0 {
            first_entry_block = block_map.get(&body_entry).copied();
        }

        // value_map: old value -> new value for this iteration
        let mut value_map: FxHashMap<ValueId, ValueId> = FxHashMap::default();

        // Map the IV to a constant for this iteration.
        let iv_value = func.dfg.make_imm_value(iv_const);
        value_map.insert(counted.iv, iv_value);

        // Map other header phi results to their appropriate values.
        for phi_info in &header_phis {
            if phi_info.result == counted.iv {
                continue; // Already handled above.
            }
            let mapped = if iter_idx == 0 {
                // First iteration: use the preheader value.
                phi_info.init_val
            } else {
                // Subsequent iterations: use the previous iteration's latch value.
                *prev_iteration_values
                    .get(&phi_info.latch_val)
                    .unwrap_or(&phi_info.latch_val)
            };
            value_map.insert(phi_info.result, mapped);
        }

        // Clone instructions from each body block into the new blocks.
        for &old_block in &body_blocks {
            let new_block = block_map[&old_block];
            let insts: Vec<InstId> = func.layout.iter_inst(old_block).collect();

            for &old_inst in &insts {
                let is_phi = func.dfg.is_phi(old_inst);
                let is_term = func.dfg.is_terminator(old_inst);

                if is_phi {
                    // For phis in body blocks, resolve them:
                    // If the phi has an arg from a block within the loop, remap it.
                    // If it has an arg from outside the loop, use that value directly.
                    let phi = downcast::<&Phi>(func.inst_set(), func.dfg.inst(old_inst)).unwrap();
                    let old_results = func.dfg.inst_results(old_inst);
                    if old_results.is_empty() {
                        continue;
                    }
                    let old_result = old_results[0];
                    let old_ty = func.dfg.value_ty(old_result);

                    // Rebuild the phi with remapped blocks and values.
                    let mut new_args: Vec<(ValueId, BlockId)> = Vec::new();
                    for &(val, block) in phi.args() {
                        let new_block_ref = if block == header {
                            // If coming from header, in the unrolled version this
                            // comes from "before" this copy — skip it for the first
                            // iteration or remap.
                            continue;
                        } else if let Some(&mapped_block) = block_map.get(&block) {
                            mapped_block
                        } else {
                            // Block from outside the loop or from a previous copy.
                            // For iter 0, use the value as-is. For later iters,
                            // this shouldn't happen in a well-formed single-latch loop.
                            block
                        };
                        let new_val = remap_value(func, &value_map, val);
                        new_args.push((new_val, new_block_ref));
                    }

                    if new_args.len() == 1 {
                        // Single incoming: no phi needed, alias directly.
                        value_map.insert(old_result, new_args[0].0);
                    } else if new_args.is_empty() {
                        // No args (e.g., all came from header): use the value from
                        // the previous iteration or from header phi init.
                        let fallback = remap_value(func, &value_map, old_result);
                        value_map.insert(old_result, fallback);
                    } else {
                        // Multiple incoming: create a new phi.
                        let new_phi = func.dfg.make_phi(new_args);
                        let new_inst = func.dfg.make_inst(new_phi);
                        func.layout.append_inst(new_inst, new_block);
                        let new_result = func.dfg.make_value(Value::Inst {
                            inst: new_inst,
                            result_idx: 0,
                            ty: old_ty,
                        });
                        func.dfg.attach_result(new_inst, new_result);
                        value_map.insert(old_result, new_result);
                    }
                    continue;
                }

                if is_term {
                    // Check if this terminator jumps back to the header (latch edge).
                    let branch_info = func.dfg.branch_info(old_inst);
                    let jumps_to_header = branch_info
                        .map(|bi| bi.dests().contains(&header))
                        .unwrap_or(false);

                    if jumps_to_header {
                        if iter_idx < counted.trip_count - 1 {
                            // Not the last iteration: this will be wired to the next
                            // copy's entry in the next iteration. For now, emit a
                            // placeholder jump. We'll fix it after creating all copies.
                            // Actually, we can just emit a jump to exit_block and fix
                            // it later, but it's simpler to handle this per-iteration.
                            //
                            // For a simple latch that does `jump header`, we replace
                            // with `jump next_entry`. But we don't have next_entry yet.
                            // Skip the terminator; we'll add it after the next iter's
                            // blocks are created.
                        } else {
                            // Last iteration: jump to exit.
                            let jump = func.dfg.make_jump(exit_block);
                            let jump_inst = func.dfg.make_inst(jump);
                            func.layout.append_inst(jump_inst, new_block);
                        }
                        continue;
                    }

                    // Non-latch terminator (e.g., branch within the body).
                    // Clone it with block and value remapping.
                    clone_and_remap_inst(func, old_inst, new_block, &value_map, &block_map);
                    continue;
                }

                // Regular (non-phi, non-terminator) instruction: clone with remapping.
                let old_results: SmallVec<[ValueId; 2]> =
                    func.dfg.inst_results(old_inst).iter().copied().collect();
                let _result_tys: SmallVec<[Type; 2]> = old_results
                    .iter()
                    .map(|&v| func.dfg.value_ty(v))
                    .collect();

                let new_inst_id =
                    clone_and_remap_inst(func, old_inst, new_block, &value_map, &block_map);

                // Map old results to new results.
                let new_results: SmallVec<[ValueId; 2]> =
                    func.dfg.inst_results(new_inst_id).iter().copied().collect();
                for (i, &old_result) in old_results.iter().enumerate() {
                    if i < new_results.len() {
                        value_map.insert(old_result, new_results[i]);
                    }
                }
            }
        }

        // Save the value map so the next iteration can reference values from this one.
        prev_iteration_values = value_map.clone();

        // Update last_iter_value_map for exit phi fixup.
        for (&old_val, &new_val) in &value_map {
            last_iter_value_map.insert(old_val, new_val);
        }
    }

    // Fix header phi exit values: when the loop exits, the header phis should
    // reflect the state AFTER the last iteration's body executed (i.e., the
    // values that would flow into a hypothetical iteration `trip_count`).
    // During the iteration loop, `last_iter_value_map[phi.result]` was set to
    // the value at the START of the last iteration. We need to update it to
    // the latch value from the last iteration (the value produced by the body).
    for phi_info in &header_phis {
        if phi_info.result == counted.iv {
            // For the IV, the exit value is init + trip_count * step.
            let exit_iv = compute_iv_at(counted.init, counted.step, counted.trip_count);
            let exit_iv_val = func.dfg.make_imm_value(exit_iv);
            last_iter_value_map.insert(phi_info.result, exit_iv_val);
            continue;
        }
        // For non-IV phis, use the last iteration's clone of the latch value.
        if let Some(&last_latch_val) = prev_iteration_values.get(&phi_info.latch_val) {
            last_iter_value_map.insert(phi_info.result, last_latch_val);
        }
    }

    // Now wire the latch terminators between iterations.
    // We need to go back and add jump instructions from each iteration's latch
    // to the next iteration's body entry.
    wire_iteration_latches(func, counted, &body_blocks, first_entry_block);

    // Wire the preheader to the first copy's entry.
    if let Some(first_entry) = first_entry_block {
        let preheader_term = func.layout.last_inst_of(preheader).unwrap();
        func.dfg.rewrite_branch_dest(preheader_term, header, first_entry);
    }

    // Fix up phis in the exit block.
    fix_exit_phis(func, counted, &last_iter_value_map);

    // The original loop blocks are now unreachable. Before CfgCleanup removes
    // them, redirect all uses of values defined in the loop to their
    // last-iteration equivalents so no dangling references remain.
    //
    // We need to alias every value defined in the original loop blocks.
    // The last_iter_value_map covers instruction results we cloned.
    // For header phi results, the IV is mapped to a constant, and other
    // phis' latch values should also be in the map.
    for &block in &counted.loop_blocks {
        for inst in func.layout.iter_inst(block) {
            let results: SmallVec<[ValueId; 2]> =
                func.dfg.inst_results(inst).iter().copied().collect();
            for old_val in results {
                if let Some(&new_val) = last_iter_value_map.get(&old_val) {
                    if old_val != new_val {
                        func.dfg.change_to_alias(old_val, new_val);
                    }
                }
            }
        }
    }
}

/// Handle zero-trip unrolling: wire preheader directly to exit, fix exit phis.
fn unroll_zero_trip(func: &mut Function, counted: &CountedLoop) {
    let preheader = counted.preheader;
    let header = counted.header;
    let exit_block = counted.exit_block;

    // Rewrite preheader's terminator to jump to exit instead of header.
    let preheader_term = func.layout.last_inst_of(preheader).unwrap();
    func.dfg.rewrite_branch_dest(preheader_term, header, exit_block);

    // Fix exit block phis: replace the header's incoming edge with preheader.
    // For zero trips, the exit phi's value from the loop should use the init values.
    let header_phis = collect_header_phis(func, header, preheader, &counted.loop_blocks);
    let mut init_map: FxHashMap<ValueId, ValueId> = FxHashMap::default();
    for phi_info in &header_phis {
        init_map.insert(phi_info.result, phi_info.init_val);
    }

    // Rewrite exit phis.
    for inst in func.layout.iter_inst(exit_block) {
        if !func.dfg.is_phi(inst) {
            continue;
        }
        let phi = func.dfg.cast_phi_mut(inst).unwrap();
        // Find args that come from loop blocks and replace with preheader edge.
        let mut new_args: Vec<(ValueId, BlockId)> = Vec::new();
        let mut had_loop_edge = false;
        for &(val, block) in phi.args() {
            if counted.loop_blocks.contains(&block) || block == header {
                if !had_loop_edge {
                    // Replace with preheader edge, using the init value.
                    let remapped = init_map.get(&val).copied().unwrap_or(val);
                    new_args.push((remapped, preheader));
                    had_loop_edge = true;
                }
            } else {
                new_args.push((val, block));
            }
        }

        func.dfg.untrack_inst(inst);
        let phi = func.dfg.cast_phi_mut(inst).unwrap();
        *phi.args_mut() = new_args;
        func.dfg.attach_user(inst);
    }
}

/// Info about a phi in the loop header.
struct HeaderPhiInfo {
    /// The phi instruction.
    _inst: InstId,
    /// The result value of the phi.
    result: ValueId,
    /// The value coming from the preheader (init).
    init_val: ValueId,
    /// The value coming from the latch (updated value).
    latch_val: ValueId,
}

/// Collect all phis in the header and their init/latch values.
fn collect_header_phis(
    func: &Function,
    header: BlockId,
    preheader: BlockId,
    loop_blocks: &[BlockId],
) -> Vec<HeaderPhiInfo> {
    let is = func.inst_set();
    let mut result = Vec::new();

    for inst in func.layout.iter_inst(header) {
        let Some(phi) = downcast::<&Phi>(is, func.dfg.inst(inst)) else {
            continue;
        };
        let args = phi.args();
        if args.len() != 2 {
            continue;
        }
        let results = func.dfg.inst_results(inst);
        if results.is_empty() {
            continue;
        }

        let (init_val, latch_val) =
            if args[0].1 == preheader && loop_blocks.contains(&args[1].1) {
                (args[0].0, args[1].0)
            } else if args[1].1 == preheader && loop_blocks.contains(&args[0].1) {
                (args[1].0, args[0].0)
            } else {
                continue;
            };

        result.push(HeaderPhiInfo {
            _inst: inst,
            result: results[0],
            init_val,
            latch_val,
        });
    }

    result
}

/// Compute the IV value at a given iteration index.
fn compute_iv_at(init: Immediate, step: Immediate, iter_idx: usize) -> Immediate {
    let mut val = init;
    for _ in 0..iter_idx {
        val = val + step;
    }
    val
}

/// Remap a value through the value_map, or return it unchanged if not mapped.
fn remap_value(
    _func: &Function,
    value_map: &FxHashMap<ValueId, ValueId>,
    val: ValueId,
) -> ValueId {
    value_map.get(&val).copied().unwrap_or(val)
}

/// Clone an instruction, remap its operands, and append it to `target_block`.
/// Returns the new instruction ID.
fn clone_and_remap_inst(
    func: &mut Function,
    old_inst: InstId,
    target_block: BlockId,
    value_map: &FxHashMap<ValueId, ValueId>,
    block_map: &FxHashMap<BlockId, BlockId>,
) -> InstId {
    let mut cloned = func.dfg.clone_inst(old_inst);

    // Remap values and blocks.
    let value_map_clone = value_map.clone();
    let block_map_clone = block_map.clone();
    let mut remapper = OperandRemapper {
        value_map: &value_map_clone,
        block_map: &block_map_clone,
    };
    cloned.accept_mut(&mut remapper);

    let old_results = func.dfg.inst_results(old_inst);
    let result_tys: SmallVec<[Type; 2]> = old_results
        .iter()
        .map(|&v| func.dfg.value_ty(v))
        .collect();

    let new_inst_id = func.dfg.make_inst_dyn(cloned);
    func.layout.append_inst(new_inst_id, target_block);

    // Create result values for the cloned instruction.
    for (idx, ty) in result_tys.iter().enumerate() {
        let new_val = func.dfg.make_value(Value::Inst {
            inst: new_inst_id,
            result_idx: idx.try_into().unwrap(),
            ty: *ty,
        });
        func.dfg.append_result(new_inst_id, new_val);
    }

    new_inst_id
}

/// A visitor that remaps value and block references in a cloned instruction.
struct OperandRemapper<'a> {
    value_map: &'a FxHashMap<ValueId, ValueId>,
    block_map: &'a FxHashMap<BlockId, BlockId>,
}

impl VisitorMut for OperandRemapper<'_> {
    fn visit_value_id(&mut self, value: &mut ValueId) {
        if let Some(&mapped) = self.value_map.get(value) {
            *value = mapped;
        }
    }

    fn visit_block_id(&mut self, block: &mut BlockId) {
        if let Some(&mapped) = self.block_map.get(block) {
            *block = mapped;
        }
    }
}

/// After creating all iteration copies, go back and wire each iteration's latch
/// to the next iteration's body entry block.
///
/// During the main unrolling loop, latch blocks that jump to the header were left
/// without terminators (except the last iteration which jumps to exit). Now we
/// add the missing jumps.
fn wire_iteration_latches(
    func: &mut Function,
    counted: &CountedLoop,
    body_blocks: &[BlockId],
    first_entry_block: Option<BlockId>,
) {
    if counted.trip_count <= 1 {
        return; // Only one iteration, already wired to exit.
    }

    let first_entry = match first_entry_block {
        Some(b) => b,
        None => return,
    };

    // We need to find the cloned blocks for each iteration. They were inserted
    // sequentially after the preheader. Walk the block list to find them.
    let blocks_per_iter = body_blocks.len();

    // Collect all the cloned blocks in order (they should appear after preheader).
    let mut cloned_blocks: Vec<BlockId> = Vec::new();
    let mut found_first = false;
    for block in func.layout.iter_block() {
        if block == first_entry {
            found_first = true;
        }
        if found_first {
            cloned_blocks.push(block);
            if cloned_blocks.len() == blocks_per_iter * counted.trip_count {
                break;
            }
        }
    }

    if cloned_blocks.len() != blocks_per_iter * counted.trip_count {
        return; // Something went wrong, bail.
    }

    // For each iteration except the last, find the latch block (last block of the
    // iteration that should have a terminator jumping to the next iteration's entry).
    for iter_idx in 0..counted.trip_count - 1 {
        let iter_start = iter_idx * blocks_per_iter;
        let next_iter_start = (iter_idx + 1) * blocks_per_iter;
        let next_entry = cloned_blocks[next_iter_start];

        // Find the latch block in this iteration: it's the block that should have
        // a jump to the header but was left without a terminator.
        for i in iter_start..iter_start + blocks_per_iter {
            let block = cloned_blocks[i];
            let has_term = func.layout.last_inst_of(block).map_or(false, |inst| {
                func.dfg.is_terminator(inst)
            });
            if !has_term {
                // This is the latch block — add a jump to the next iteration's entry.
                let jump = func.dfg.make_jump(next_entry);
                let jump_inst = func.dfg.make_inst(jump);
                func.layout.append_inst(jump_inst, block);
            }
        }
    }
}

/// Fix up phi instructions in the exit block that reference loop-internal values.
fn fix_exit_phis(
    func: &mut Function,
    counted: &CountedLoop,
    last_iter_value_map: &FxHashMap<ValueId, ValueId>,
) {
    let exit_block = counted.exit_block;
    let header = counted.header;
    let preheader = counted.preheader;

    // Build a complete mapping from any loop-defined value to its last-iteration
    // equivalent. The last_iter_value_map covers cloned instruction results.
    // We also need to map header phi latch values: if a header phi has
    // latch_val -> some value in the loop, we need to find what that maps to.
    let header_phis = collect_header_phis(func, header, preheader, &counted.loop_blocks);
    let mut complete_map = last_iter_value_map.clone();

    // For each header phi, its latch_val (the value flowing back from the loop)
    // should map to the same thing the phi result maps to in the last iteration.
    // But the phi result in the last iteration IS the IV constant or the prev
    // iteration's latch value — which is already in the map.
    for phi_info in &header_phis {
        if !complete_map.contains_key(&phi_info.latch_val) {
            // The latch_val for the last iteration = the mapped result of the phi
            // at the last iteration.
            if let Some(&mapped) = complete_map.get(&phi_info.result) {
                complete_map.insert(phi_info.latch_val, mapped);
            }
        }
    }

    // Find the block that jumps to exit_block from the unrolled code.
    let mut last_iter_exit_src = None;
    for block in func.layout.iter_block() {
        if counted.loop_blocks.contains(&block) {
            continue;
        }
        if block == preheader {
            continue;
        }
        if let Some(term) = func.layout.last_inst_of(block) {
            if let Some(branch) = func.dfg.branch_info(term) {
                if branch.dests().contains(&exit_block) {
                    last_iter_exit_src = Some(block);
                }
            }
        }
    }

    let last_exit_src = match last_iter_exit_src {
        Some(b) => b,
        None => preheader,
    };

    // Rewrite exit block phis.
    let insts: Vec<InstId> = func.layout.iter_inst(exit_block).collect();
    for inst in insts {
        if !func.dfg.is_phi(inst) {
            continue;
        }

        func.dfg.untrack_inst(inst);
        let phi = func.dfg.cast_phi_mut(inst).unwrap();
        let mut new_args: Vec<(ValueId, BlockId)> = Vec::new();

        for &(val, block) in phi.args() {
            if block == header || counted.loop_blocks.contains(&block) {
                let remapped_val = complete_map.get(&val).copied().unwrap_or(val);
                new_args.push((remapped_val, last_exit_src));
            } else {
                new_args.push((val, block));
            }
        }

        let phi = func.dfg.cast_phi_mut(inst).unwrap();
        *phi.args_mut() = new_args;
        func.dfg.attach_user(inst);
    }
}

#[cfg(test)]
mod tests {
    use sonatina_ir::{
        ControlFlowGraph, Type,
        builder::test_util::*,
        inst::{
            arith::Add,
            cmp::Lt,
            control_flow::{Br, Jump, Phi, Return},
        },
        prelude::*,
    };

    use crate::{domtree::DomTree, loop_analysis::LoopTree};

    use super::LoopUnrollSolver;

    /// Build a simple counted loop:
    ///   i = 0; while (i < 4) { i += 1; }
    /// and verify it gets unrolled (loop disappears).
    #[test]
    fn unroll_simple_counted_loop() {
        let mb = test_module_builder();
        let (evm, mut builder) = test_func_builder(&mb, &[], Type::Unit);
        let is = evm.inst_set();

        let preheader = builder.append_block();
        let header = builder.append_block();
        let body = builder.append_block();
        let exit = builder.append_block();

        // preheader: jump header
        builder.switch_to_block(preheader);
        let v_init = builder.make_imm_value(0i32);
        builder.insert_inst_no_result_with(|| Jump::new(is, header));

        // header: v_ind = phi(v_init, preheader)(v_next, body)
        //         v_cmp = lt v_ind, 10
        //         br v_cmp, body, exit
        builder.switch_to_block(header);
        let v_ind = builder.insert_inst_with(
            || Phi::new(is, vec![(v_init, preheader)]),
            Type::I32,
        );
        let v_limit = builder.make_imm_value(20i32);
        let v_cmp = builder.insert_inst_with(|| Lt::new(is, v_ind, v_limit), Type::I1);
        builder.insert_inst_no_result_with(|| Br::new(is, v_cmp, body, exit));

        // body: v_next = add v_ind, 1
        //       jump header
        builder.switch_to_block(body);
        let v_step = builder.make_imm_value(1i32);
        let v_next = builder.insert_inst_with(|| Add::new(is, v_ind, v_step), Type::I32);
        builder.insert_inst_no_result_with(|| Jump::new(is, header));
        builder.append_phi_arg(v_ind, v_next, body);

        // exit: return
        builder.switch_to_block(exit);
        builder.insert_inst_no_result_with(|| Return::new_unit(is));

        builder.seal_all();
        builder.finish();

        let module = mb.build();
        let func_ref = module.funcs()[0];
        module.func_store.modify(func_ref, |func| {
            let mut cfg = ControlFlowGraph::default();
            cfg.compute(func);
            let mut domtree = DomTree::default();
            domtree.compute(&cfg);
            let mut lpt = LoopTree::default();
            lpt.compute(&cfg, &domtree);

            // Before unrolling: there should be 1 loop.
            assert_eq!(lpt.loop_num(), 1);

            let mut solver = LoopUnrollSolver::new();
            solver.run(func, &mut cfg, &mut lpt);

            // After unrolling: the loop should be gone.
            assert_eq!(lpt.loop_num(), 0, "loop should be fully unrolled");
        });
    }

    /// A loop with trip count 0 should just wire preheader to exit.
    #[test]
    fn unroll_zero_trip_loop() {
        let mb = test_module_builder();
        let (evm, mut builder) = test_func_builder(&mb, &[], Type::Unit);
        let is = evm.inst_set();

        let preheader = builder.append_block();
        let header = builder.append_block();
        let body = builder.append_block();
        let exit = builder.append_block();

        builder.switch_to_block(preheader);
        let v_init = builder.make_imm_value(10i32); // init >= limit, so 0 trips
        builder.insert_inst_no_result_with(|| Jump::new(is, header));

        builder.switch_to_block(header);
        let v_ind = builder.insert_inst_with(
            || Phi::new(is, vec![(v_init, preheader)]),
            Type::I32,
        );
        let v_limit = builder.make_imm_value(4i32);
        let v_cmp = builder.insert_inst_with(|| Lt::new(is, v_ind, v_limit), Type::I1);
        builder.insert_inst_no_result_with(|| Br::new(is, v_cmp, body, exit));

        builder.switch_to_block(body);
        let v_step = builder.make_imm_value(1i32);
        let v_next = builder.insert_inst_with(|| Add::new(is, v_ind, v_step), Type::I32);
        builder.insert_inst_no_result_with(|| Jump::new(is, header));
        builder.append_phi_arg(v_ind, v_next, body);

        builder.switch_to_block(exit);
        builder.insert_inst_no_result_with(|| Return::new_unit(is));

        builder.seal_all();
        builder.finish();

        let module = mb.build();
        let func_ref = module.funcs()[0];
        module.func_store.modify(func_ref, |func| {
            let mut cfg = ControlFlowGraph::default();
            let mut lpt = LoopTree::default();

            let mut solver = LoopUnrollSolver::new();
            solver.run(func, &mut cfg, &mut lpt);

            // The loop should be gone.
            assert_eq!(lpt.loop_num(), 0);
        });
    }

    /// A loop that is too large to unroll should be left alone.
    #[test]
    fn skip_large_loop() {
        let mb = test_module_builder();
        let (evm, mut builder) = test_func_builder(&mb, &[], Type::Unit);
        let is = evm.inst_set();

        let preheader = builder.append_block();
        let header = builder.append_block();
        let body = builder.append_block();
        let exit = builder.append_block();

        builder.switch_to_block(preheader);
        let v_init = builder.make_imm_value(0i32);
        builder.insert_inst_no_result_with(|| Jump::new(is, header));

        builder.switch_to_block(header);
        let v_ind = builder.insert_inst_with(
            || Phi::new(is, vec![(v_init, preheader)]),
            Type::I32,
        );
        // 500 iterations * body_insts > MAX_UNROLLED_INSTS
        let v_limit = builder.make_imm_value(500i32);
        let v_cmp = builder.insert_inst_with(|| Lt::new(is, v_ind, v_limit), Type::I1);
        builder.insert_inst_no_result_with(|| Br::new(is, v_cmp, body, exit));

        builder.switch_to_block(body);
        let v_step = builder.make_imm_value(1i32);
        let v_next = builder.insert_inst_with(|| Add::new(is, v_ind, v_step), Type::I32);
        builder.insert_inst_no_result_with(|| Jump::new(is, header));
        builder.append_phi_arg(v_ind, v_next, body);

        builder.switch_to_block(exit);
        builder.insert_inst_no_result_with(|| Return::new_unit(is));

        builder.seal_all();
        builder.finish();

        let module = mb.build();
        let func_ref = module.funcs()[0];
        module.func_store.modify(func_ref, |func| {
            let mut cfg = ControlFlowGraph::default();
            let mut lpt = LoopTree::default();

            let mut solver = LoopUnrollSolver::new();
            solver.run(func, &mut cfg, &mut lpt);

            // The loop should still be there.
            assert_eq!(lpt.loop_num(), 1, "large loop should not be unrolled");
        });
    }
}
