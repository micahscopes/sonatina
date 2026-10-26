//! Keeping reads, calls and writes in program order in the emitted code.
//!
//! WAFFLE places a value "under" its only user, so that the code is a tree
//! and needs no local, when the value is a pure operation, or when it is the
//! last operation with an effect before its user. The user may itself be a pure
//! operation placed under a later user. Then the effectful operation moves
//! with it, past every operation in between. A load that is followed by a pure
//! operation on its value, and then by a store, is emitted after the store if
//! the result of the pure operation is used after the store:
//!
//! ```text
//! v1 = load a      ;; reads the old value
//! v2 = add v1 1
//! store a 9        ;; v1 is read here instead, in the emitted code
//! v3 = mul v2 3
//! ```
//!
//! The fix is a second use of the effectful value, because WAFFLE leaves a value
//! with two uses in a local. The second use is a pure operation whose result
//! nobody reads, which the encoder drops, and it is added only where the
//! pattern above occurs.

use rustc_hash::FxHashMap;
use waffle::{Block, FunctionBody, Operator, SideEffect, Type as WType, Value, ValueDef};

/// A use of a value.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Use {
    /// By the operation at this index of this block.
    Op(Block, usize),
    /// By a terminator or by a pick of one result of a multi-value operation.
    Other,
}

fn writes(effects: &[SideEffect]) -> bool {
    effects.iter().any(|effect| {
        matches!(
            effect,
            SideEffect::WriteMem
                | SideEffect::WriteGlobal
                | SideEffect::WriteTable
                | SideEffect::WriteLocal
                | SideEffect::All
        )
    })
}

/// Whether `later` must stay after `earlier`: an operation that may write
/// stays behind everything with an effect, and one that may only read or trap
/// stays behind everything that may write.
fn must_stay_after(earlier: &[SideEffect], later: &[SideEffect]) -> bool {
    if later.is_empty() || earlier.is_empty() {
        return false;
    }
    writes(earlier) || writes(later)
}

/// A single-result operator of a value, with its effects.
fn single_result_op(body: &FunctionBody, value: Value) -> Option<Operator> {
    match &body.values[value] {
        ValueDef::Operator(op, _, tys) if tys.len() == 1 => Some(*op),
        _ => None,
    }
}

/// Add a second use to every value that WAFFLE would otherwise move past an
/// operation that it must stay before.
pub(super) fn keep_effect_order(body: &mut FunctionBody) {
    // Where each value is used.
    let mut uses: FxHashMap<Value, Vec<Use>> = FxHashMap::default();
    for (block, def) in body.blocks.entries() {
        for (index, &value) in def.insts.iter().enumerate() {
            match &body.values[value] {
                ValueDef::Operator(_, args, _) => {
                    for &arg in &body.arg_pool[*args] {
                        let arg = body.resolve_alias(arg);
                        uses.entry(arg).or_default().push(Use::Op(block, index));
                    }
                }
                ValueDef::PickOutput(from, ..) => {
                    uses.entry(*from).or_default().push(Use::Other);
                }
                _ => {}
            }
        }
        def.terminator.visit_uses(|used| {
            uses.entry(body.resolve_alias(used))
                .or_default()
                .push(Use::Other);
        });
    }

    let mut pins: Vec<(Block, usize, Value, WType)> = Vec::new();
    for (block, def) in body.blocks.entries() {
        let insts = &def.insts;
        let effects_at = |index: usize| -> &'static [SideEffect] {
            match &body.values[insts[index]] {
                ValueDef::Operator(op, _, _) => op.effects(),
                _ => &[],
            }
        };
        for (i, &n) in insts.iter().enumerate() {
            let Some(n_op) = single_result_op(body, n) else {
                continue;
            };
            let n_effects = n_op.effects();
            if n_effects.is_empty() {
                continue;
            }
            // Only a value with exactly one use, by an operation of this block.
            let [Use::Op(user_block, user)] = uses.get(&n).map(Vec::as_slice).unwrap_or(&[]) else {
                continue;
            };
            if *user_block != block || *user <= i {
                continue;
            }
            // WAFFLE nests `n` under its user only if no operation with an
            // effect lies between them, and only a pure user can move.
            if (i + 1..*user).any(|k| !effects_at(k).is_empty()) {
                continue;
            }
            let mut position = *user;
            let mut moves = effects_at(position).is_empty();
            if !moves {
                continue;
            }
            // Follow the chain of pure operations that are each the only use of
            // the one before. The last one is where the code of the chain lands.
            let mut current = insts[position];
            while moves {
                let next = uses.get(&current).map(Vec::as_slice).unwrap_or(&[]);
                match next {
                    [Use::Op(next_block, next_index)]
                        if *next_block == block
                            && *next_index > position
                            && single_result_op(body, current).is_some() =>
                    {
                        position = *next_index;
                        current = insts[position];
                        moves = effects_at(position).is_empty();
                    }
                    _ => break,
                }
            }
            // `position` is the operation at which the chain is emitted. An
            // operation in between that `n` must stay before is a hazard.
            let hazard = (*user + 1..position).any(|k| must_stay_after(n_effects, effects_at(k)));
            if hazard {
                let ty = match &body.values[n] {
                    ValueDef::Operator(_, _, tys) => body.type_pool[*tys][0],
                    _ => continue,
                };
                pins.push((block, i, n, ty));
            }
        }
    }

    // Insert from the end of each block so that earlier indices stay valid.
    pins.sort_by_key(|&(block, index, ..)| (block, std::cmp::Reverse(index)));
    for (block, index, value, ty) in pins {
        let test = match ty {
            WType::I64 => Operator::I64Eqz,
            _ => Operator::I32Eqz,
        };
        let pin = body.add_op(block, test, &[value], &[WType::I32]);
        // `add_op` appends; move the operation to just after `value`.
        let moved = body.blocks[block].insts.pop();
        debug_assert_eq!(moved, Some(pin));
        body.blocks[block].insts.insert(index + 1, pin);
    }
}
