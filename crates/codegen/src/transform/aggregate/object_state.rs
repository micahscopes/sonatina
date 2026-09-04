use cranelift_entity::SecondaryMap;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::ops::Range;
use sonatina_ir::{
    Function, InstId, ValueId,
    inst::{cast, control_flow, data, downcast},
    module::ModuleCtx,
};

use super::{
    object_tracking::{
        ObjectSlice, TrackedObject, enum_tag_object_slice, enum_variant_field_object_slice,
    },
    provenance::MayProvenance,
};

/// A normalized union of half-open live-leaf intervals.
///
/// Whole fixed-size arrays are common in GPU actors. Representing their live
/// leaves one hash entry at a time made an otherwise small control module
/// consume gigabytes during object load/store analysis. Intervals preserve the
/// exact same leaf-set lattice while keeping whole roots and contiguous fields
/// constant-size.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct LeafIntervalSet {
    ranges: Vec<Range<usize>>,
}

impl LeafIntervalSet {
    pub(crate) fn mark_range(&mut self, range: Range<usize>) {
        if range.start >= range.end {
            return;
        }
        let mut start = range.start;
        let mut end = range.end;
        let mut merged = Vec::with_capacity(self.ranges.len() + 1);
        let mut inserted = false;
        for current in std::mem::take(&mut self.ranges) {
            if current.end < start {
                merged.push(current);
            } else if end < current.start {
                if !inserted {
                    merged.push(start..end);
                    inserted = true;
                }
                merged.push(current);
            } else {
                start = start.min(current.start);
                end = end.max(current.end);
            }
        }
        if !inserted {
            merged.push(start..end);
        }
        self.ranges = merged;
    }

    pub(crate) fn clear_range(&mut self, range: Range<usize>) {
        if range.start >= range.end {
            return;
        }
        let mut remaining = Vec::with_capacity(self.ranges.len() + 1);
        for current in std::mem::take(&mut self.ranges) {
            if current.end <= range.start || range.end <= current.start {
                remaining.push(current);
                continue;
            }
            if current.start < range.start {
                remaining.push(current.start..range.start);
            }
            if range.end < current.end {
                remaining.push(range.end..current.end);
            }
        }
        self.ranges = remaining;
    }

    pub(crate) fn union_with(&mut self, other: &Self) {
        for range in &other.ranges {
            self.mark_range(range.clone());
        }
    }

    pub(crate) fn intersect_with(&mut self, other: &Self) {
        let mut intersection = Vec::new();
        let mut left = 0;
        let mut right = 0;
        while left < self.ranges.len() && right < other.ranges.len() {
            let lhs = &self.ranges[left];
            let rhs = &other.ranges[right];
            let start = lhs.start.max(rhs.start);
            let end = lhs.end.min(rhs.end);
            if start < end {
                intersection.push(start..end);
            }
            if lhs.end <= rhs.end {
                left += 1;
            } else {
                right += 1;
            }
        }
        self.ranges = intersection;
    }

    pub(crate) fn intersects(&self, range: Range<usize>) -> bool {
        range.start < range.end
            && self
                .ranges
                .iter()
                .any(|live| live.start < range.end && range.start < live.end)
    }

    pub(crate) fn contains_range(&self, range: Range<usize>) -> bool {
        range.start >= range.end
            || self
                .ranges
                .iter()
                .any(|live| live.start <= range.start && range.end <= live.end)
    }

    pub(crate) fn clear(&mut self) {
        self.ranges.clear();
    }

    #[cfg(test)]
    pub(crate) fn interval_count(&self) -> usize {
        self.ranges.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }
}

pub(crate) type LiveLeafMap = FxHashMap<ValueId, LeafIntervalSet>;
pub(crate) type ObjectSliceList = SmallVec<[ObjectSlice; 4]>;
pub(crate) type ObservedRoots = SmallVec<[ValueId; 4]>;

pub(crate) fn is_pure_object_address_inst(func: &Function, inst: InstId) -> bool {
    downcast::<&data::ObjAlloc>(func.inst_set(), func.dfg.inst(inst)).is_some()
        || downcast::<&data::Gep>(func.inst_set(), func.dfg.inst(inst)).is_some()
        || downcast::<&cast::Bitcast>(func.inst_set(), func.dfg.inst(inst)).is_some()
        || downcast::<&data::ObjProj>(func.inst_set(), func.dfg.inst(inst)).is_some()
        || downcast::<&data::ObjIndex>(func.inst_set(), func.dfg.inst(inst)).is_some()
        || downcast::<&data::EnumProj>(func.inst_set(), func.dfg.inst(inst)).is_some()
        || downcast::<&control_flow::Phi>(func.inst_set(), func.dfg.inst(inst)).is_some()
}

pub(crate) fn observed_roots(
    func: &Function,
    inst: InstId,
    provenance: MayProvenance<'_>,
    skip: &[ValueId],
) -> (ObservedRoots, bool) {
    let mut roots = FxHashSet::default();
    let mut observed_unknown = false;
    for value in func.dfg.inst(inst).collect_values() {
        if skip.contains(&value) {
            continue;
        }
        let root_set = provenance.may_roots(value);
        observed_unknown |= root_set.has_unknown();
        for root in root_set.observed().iter() {
            roots.insert(root.value());
        }
    }
    (roots.into_iter().collect(), observed_unknown)
}

pub(crate) fn observed_roots_ignoring_pure_address_ops(
    func: &Function,
    inst: InstId,
    provenance: MayProvenance<'_>,
    skip: &[ValueId],
) -> (ObservedRoots, bool) {
    if is_pure_object_address_inst(func, inst) {
        return (ObservedRoots::new(), false);
    }
    observed_roots(func, inst, provenance, skip)
}

pub(crate) fn tracked_root_total_leaves(
    tracked: &SecondaryMap<ValueId, Option<TrackedObject>>,
    root: ValueId,
) -> usize {
    tracked[root]
        .as_ref()
        .copied()
        .map(TrackedObject::total_leaves)
        .expect("tracked root should exist")
}

pub(crate) fn union_live_leaf_maps(states: impl Iterator<Item = LiveLeafMap>) -> LiveLeafMap {
    let mut out = LiveLeafMap::default();
    for state in states {
        for (root, leaves) in state {
            out.entry(root).or_default().union_with(&leaves);
        }
    }
    out
}

pub(crate) fn mark_root_live(live: &mut LiveLeafMap, root: ValueId, total_leaves: usize) {
    live.entry(root).or_default().mark_range(0..total_leaves);
}

pub(crate) fn mark_live_slice(live: &mut LiveLeafMap, slice: ObjectSlice) {
    live.entry(slice.root).or_default().mark_range(
        slice.first_leaf..slice.first_leaf + slice.leaf_count,
    );
}

pub(crate) fn mark_live_leaf(live: &mut LiveLeafMap, root: ValueId, leaf: usize) {
    live.entry(root).or_default().mark_range(leaf..leaf + 1);
}

pub(crate) fn mark_live_tracked_object(live: &mut LiveLeafMap, tracked: TrackedObject) {
    match tracked {
        TrackedObject::Exact(slice) => mark_live_slice(live, slice),
        TrackedObject::RootUnknown { root, total_leaves } => {
            mark_root_live(live, root, total_leaves)
        }
    }
}

pub(crate) fn clear_live_slice(live: &mut LiveLeafMap, slice: ObjectSlice) {
    let Some(entry) = live.get_mut(&slice.root) else {
        return;
    };
    entry.clear_range(slice.first_leaf..slice.first_leaf + slice.leaf_count);
    if entry.is_empty() {
        live.remove(&slice.root);
    }
}

pub(crate) fn clear_live_leaf(live: &mut LiveLeafMap, root: ValueId, leaf: usize) {
    let Some(entry) = live.get_mut(&root) else {
        return;
    };
    entry.clear_range(leaf..leaf + 1);
    if entry.is_empty() {
        live.remove(&root);
    }
}

pub(crate) fn slice_has_live_leaf(live: &LiveLeafMap, slice: ObjectSlice) -> bool {
    live.get(&slice.root).is_some_and(|entry| {
        entry.intersects(slice.first_leaf..slice.first_leaf + slice.leaf_count)
    })
}

pub(crate) fn enum_write_variant_slices(
    ctx: &ModuleCtx,
    base_slice: ObjectSlice,
    enum_write_variant: &data::EnumWriteVariant,
) -> ObjectSliceList {
    let mut slices = ObjectSliceList::new();
    if let Some(tag_slice) = enum_tag_object_slice(ctx, base_slice) {
        slices.push(tag_slice);
    }
    for field_idx in 0..enum_write_variant.values().len() {
        let Some(field_idx) = u32::try_from(field_idx).ok() else {
            continue;
        };
        if let Some(field_slice) = enum_variant_field_object_slice(
            ctx,
            base_slice,
            *enum_write_variant.variant(),
            field_idx,
        ) {
            slices.push(field_slice);
        }
    }
    slices
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::aggregate::{
        object_tracking::{collect_root_slices, objref_element_ty},
        provenance::collect_root_provenance,
        shape,
    };
    use sonatina_ir::module::FuncRef;
    use sonatina_parser::parse_module;

    #[test]
    fn live_leaf_intervals_keep_large_roots_compact_and_exact() {
        let mut live = LeafIntervalSet::default();
        live.mark_range(0..100_000_000);
        assert_eq!(live.ranges, vec![0..100_000_000]);

        live.clear_range(40..60);
        assert_eq!(live.ranges, vec![0..40, 60..100_000_000]);
        assert!(live.intersects(39..41));
        assert!(!live.intersects(45..55));

        live.mark_range(45..55);
        assert_eq!(live.ranges, vec![0..40, 45..55, 60..100_000_000]);
        live.mark_range(35..65);
        assert_eq!(live.ranges, vec![0..100_000_000]);

        live.clear_range(0..100_000_000);
        assert!(live.is_empty());
    }

    #[test]
    fn leaf_interval_intersection_is_exact() {
        let mut left = LeafIntervalSet::default();
        left.mark_range(0..10);
        left.mark_range(20..40);
        let mut right = LeafIntervalSet::default();
        right.mark_range(5..25);
        right.mark_range(30..35);

        left.intersect_with(&right);
        assert_eq!(left.ranges, vec![5..10, 20..25, 30..35]);
        assert!(left.contains_range(20..25));
        assert!(!left.contains_range(9..21));
    }

    fn parse_test_module(src: &str) -> sonatina_ir::Module {
        parse_module(src).expect("parse should succeed").module
    }

    fn lookup_func(module: &sonatina_ir::Module, name: &str) -> FuncRef {
        module
            .funcs()
            .into_iter()
            .find(|&func_ref| module.ctx.func_sig(func_ref, |sig| sig.name() == name))
            .expect("function should exist")
    }

    #[test]
    fn pure_address_ops_do_not_observe_roots() {
        let module = parse_test_module(
            r#"
target = "evm-ethereum-osaka"

type @Pair = { i256, i256 };

func private %f() {
block0:
    v0.objref<@Pair> = obj.alloc @Pair;
    v1.objref<i256> = obj.proj v0 0.i8;
    return;
}
"#,
        );

        let func_ref = lookup_func(&module, "f");
        module.func_store.view(func_ref, |func| {
            let mut layout_cache = shape::AggregateLayoutCache::default();
            let root_slices = collect_root_slices(func, None, &mut layout_cache);
            let provenance =
                collect_root_provenance(func, func.ctx(), &root_slices, &mut layout_cache, None);
            let obj_proj = func
                .layout
                .iter_block()
                .flat_map(|block| func.layout.iter_inst(block))
                .find(|&inst| {
                    downcast::<&data::ObjProj>(func.inst_set(), func.dfg.inst(inst)).is_some()
                })
                .expect("obj.proj should exist");

            let (roots, observed_unknown) =
                observed_roots_ignoring_pure_address_ops(func, obj_proj, provenance.may(), &[]);
            assert!(
                roots.is_empty(),
                "pure address ops should not observe roots"
            );
            assert!(
                !observed_unknown,
                "pure address ops should not report unknown roots"
            );
        });
    }

    #[test]
    fn observed_roots_report_unknown_contributors() {
        let module = parse_test_module(
            r#"
target = "evm-ethereum-osaka"

declare external %mystery() -> objref<[i256; 8]>;

type @Take = { objref<[i256; 8]> };

func private %f() {
block0:
    v0.objref<@Take> = obj.alloc @Take;
    v1.objref<objref<[i256; 8]>> = obj.proj v0 0.i8;
    v2.objref<[i256; 8]> = call %mystery;
    obj.store v1 v2;
    return;
}
"#,
        );

        let func_ref = lookup_func(&module, "f");
        let object_effects = super::super::compute_object_effect_summaries(&module);
        module.func_store.view(func_ref, |func| {
            let mut layout_cache = shape::AggregateLayoutCache::default();
            let root_slices = collect_root_slices(func, None, &mut layout_cache);
            let provenance = collect_root_provenance(
                func,
                func.ctx(),
                &root_slices,
                &mut layout_cache,
                Some(&object_effects),
            );
            let store = func
                .layout
                .iter_block()
                .flat_map(|block| func.layout.iter_inst(block))
                .find(|&inst| {
                    downcast::<&data::ObjStore>(func.inst_set(), func.dfg.inst(inst)).is_some()
                })
                .expect("obj.store should exist");
            let projection = func
                .layout
                .iter_block()
                .flat_map(|block| func.layout.iter_inst(block))
                .find_map(|inst| {
                    downcast::<&data::ObjProj>(func.inst_set(), func.dfg.inst(inst))
                        .and_then(|_| func.dfg.inst_result(inst))
                })
                .expect("obj.proj result should exist");

            let (roots, observed_unknown) = observed_roots_ignoring_pure_address_ops(
                func,
                store,
                provenance.may(),
                &[projection],
            );
            assert!(roots.is_empty(), "only skipped known roots should remain");
            assert!(observed_unknown, "unknown contributors must remain visible");
        });
    }

    #[test]
    fn enum_write_variant_slices_cover_tag_and_payload() {
        let module = parse_test_module(
            r#"
target = "evm-ethereum-osaka"

type @OptionI256 = enum {
    #None,
    #Some(i256),
};

func private %f() {
block0:
    v0.objref<@OptionI256> = obj.alloc @OptionI256;
    enum.write_variant v0 #Some (7.i256);
    return;
}
"#,
        );

        let func_ref = lookup_func(&module, "f");
        module.func_store.view(func_ref, |func| {
            let mut layout_cache = shape::AggregateLayoutCache::default();
            let root_slices = collect_root_slices(func, None, &mut layout_cache);
            let enum_root = func
                .layout
                .iter_block()
                .flat_map(|block| func.layout.iter_inst(block))
                .find_map(|inst| {
                    downcast::<&data::ObjAlloc>(func.inst_set(), func.dfg.inst(inst))
                        .and_then(|_| func.dfg.inst_result(inst))
                })
                .expect("enum alloc should exist");
            let enum_write_variant = func
                .layout
                .iter_block()
                .flat_map(|block| func.layout.iter_inst(block))
                .find_map(|inst| {
                    downcast::<&data::EnumWriteVariant>(func.inst_set(), func.dfg.inst(inst))
                })
                .expect("enum.write_variant should exist");
            let pointee_ty = objref_element_ty(func.ctx(), func.dfg.value_ty(enum_root))
                .expect("alloc result should be an objref");
            let base_shape = root_slices
                .get(&enum_root)
                .copied()
                .expect("alloc root slice should exist");
            let base_slice = ObjectSlice {
                root: enum_root,
                ty: pointee_ty,
                first_leaf: 0,
                leaf_count: base_shape.leaf_count,
                total_leaves: base_shape.leaf_count,
            };

            let slices = enum_write_variant_slices(func.ctx(), base_slice, enum_write_variant);
            assert_eq!(
                slices.len(),
                2,
                "enum.write_variant should touch tag and payload"
            );
            assert_eq!(
                slices[0],
                enum_tag_object_slice(func.ctx(), base_slice).expect("tag slice should exist")
            );
            assert_eq!(
                slices[1],
                enum_variant_field_object_slice(
                    func.ctx(),
                    base_slice,
                    *enum_write_variant.variant(),
                    0
                )
                .expect("payload slice should exist")
            );
        });
    }
}
