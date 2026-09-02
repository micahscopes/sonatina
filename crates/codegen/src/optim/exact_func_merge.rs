//! Exact merging of identity-independent private function definitions.
//!
//! This pass is deliberately conservative. It preserves every public,
//! externally declared, address-observed, object-rooted, explicitly protected,
//! or recursive function. Candidate bodies must match field-for-field after
//! values, blocks, and already-merged direct callees are normalized.

use rustc_hash::{FxHashMap, FxHashSet};
use sonatina_ir::{
    Function, InstDowncast, Module, Type, Value,
    inst::data::{GetFunctionPtr, SymAddr, SymSize, SymbolRef},
    module::FuncRef,
    object::Directive,
    visitor::VisitorMut,
};

use crate::module_analysis::{CallGraph, SccBuilder};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExactFuncMergeStats {
    pub candidate_functions: usize,
    pub merged_functions: usize,
    pub rewritten_references: usize,
    pub refinement_rounds: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CoarseFunctionKey {
    args: Vec<Type>,
    returns: Vec<Type>,
    inline_hints: u16,
    block_instruction_counts: Vec<usize>,
    instruction_result_counts: Vec<usize>,
}

/// Merge exact private function definitions and rewrite their direct references.
///
/// `protected_roots` names driver-owned entry functions that may be private but
/// must retain their identity. Object entry/include roots are protected
/// automatically. Recursive SCCs remain untouched until recursive equivalence
/// can be established as a whole component rather than one function at a time.
pub fn run_exact_private_func_merge(
    module: &Module,
    protected_roots: &[FuncRef],
) -> ExactFuncMergeStats {
    let mut protected = collect_observed_function_identities(module);
    protected.extend(protected_roots.iter().copied());
    protected.extend(collect_object_roots(module));

    let call_graph = CallGraph::build_graph(module);
    let sccs = SccBuilder::new().compute_scc(&call_graph);
    let candidates = module
        .funcs()
        .into_iter()
        .filter(|&function| {
            module.ctx.func_linkage(function).is_private()
                && module.ctx.func_linkage(function).has_definition()
                && !protected.contains(&function)
                && !sccs.scc_of(function).is_cycle
        })
        .collect::<Vec<_>>();

    let mut stats = ExactFuncMergeStats {
        candidate_functions: candidates.len(),
        ..ExactFuncMergeStats::default()
    };
    let mut aliases = FxHashMap::default();

    loop {
        stats.refinement_rounds += 1;
        let aliases_before = aliases.len();
        let mut buckets: FxHashMap<CoarseFunctionKey, Vec<FuncRef>> = FxHashMap::default();

        for &function in &candidates {
            if aliases.contains_key(&function) {
                continue;
            }

            let key = coarse_function_key(module, function);
            let representatives = buckets.entry(key).or_default();
            let equivalent = representatives.iter().copied().find(|&representative| {
                exact_function_eq(module, representative, function, &aliases)
            });

            if let Some(representative) = equivalent {
                aliases.insert(function, canonical_func(representative, &aliases));
            } else {
                representatives.push(function);
            }
        }

        if aliases.len() == aliases_before {
            break;
        }
    }

    if aliases.is_empty() {
        return stats;
    }

    stats.rewritten_references = rewrite_function_references(module, &aliases);
    let mut removed = aliases.keys().copied().collect::<Vec<_>>();
    removed.sort_unstable();
    for function in removed {
        if module.remove_func(function).is_some() {
            stats.merged_functions += 1;
        }
    }

    stats
}

fn coarse_function_key(module: &Module, function: FuncRef) -> CoarseFunctionKey {
    let (args, returns) = module.ctx.func_sig(function, |signature| {
        (signature.args().to_vec(), signature.ret_tys().to_vec())
    });
    let (block_instruction_counts, instruction_result_counts) =
        module.func_store.view(function, |body| {
            let mut per_block = Vec::new();
            let mut per_instruction = Vec::new();
            for block in body.layout.iter_block() {
                let instructions = body.layout.iter_inst(block).collect::<Vec<_>>();
                per_block.push(instructions.len());
                per_instruction.extend(
                    instructions
                        .into_iter()
                        .map(|instruction| body.dfg.inst_results(instruction).len()),
                );
            }
            (per_block, per_instruction)
        });

    CoarseFunctionKey {
        args,
        returns,
        inline_hints: module.ctx.func_hints(function).bits(),
        block_instruction_counts,
        instruction_result_counts,
    }
}

fn exact_function_eq(
    module: &Module,
    left_ref: FuncRef,
    right_ref: FuncRef,
    aliases: &FxHashMap<FuncRef, FuncRef>,
) -> bool {
    if module.ctx.func_effects(left_ref) != module.ctx.func_effects(right_ref) {
        return false;
    }

    module.func_store.view(left_ref, |left| {
        module.func_store.view(right_ref, |right| {
            exact_body_eq(left_ref, left, right_ref, right, aliases)
        })
    })
}

fn exact_body_eq(
    left_ref: FuncRef,
    left: &Function,
    right_ref: FuncRef,
    right: &Function,
    aliases: &FxHashMap<FuncRef, FuncRef>,
) -> bool {
    if left.arg_values.len() != right.arg_values.len() {
        return false;
    }

    let left_blocks = left.layout.iter_block().collect::<Vec<_>>();
    let right_blocks = right.layout.iter_block().collect::<Vec<_>>();
    if left_blocks.len() != right_blocks.len() {
        return false;
    }

    let block_map = right_blocks
        .iter()
        .copied()
        .zip(left_blocks.iter().copied())
        .collect::<FxHashMap<_, _>>();
    let mut value_map = FxHashMap::default();

    for (&right_value, &left_value) in right.arg_values.iter().zip(&left.arg_values) {
        if !bind_equivalent_value(left, left_value, right, right_value, &mut value_map) {
            return false;
        }
    }

    for (&left_block, &right_block) in left_blocks.iter().zip(&right_blocks) {
        let left_instructions = left.layout.iter_inst(left_block).collect::<Vec<_>>();
        let right_instructions = right.layout.iter_inst(right_block).collect::<Vec<_>>();
        if left_instructions.len() != right_instructions.len() {
            return false;
        }

        for (&left_inst, &right_inst) in left_instructions.iter().zip(&right_instructions) {
            let left_results = left.dfg.inst_results(left_inst);
            let right_results = right.dfg.inst_results(right_inst);
            if left_results.len() != right_results.len() {
                return false;
            }
            for (&left_value, &right_value) in left_results.iter().zip(right_results) {
                if left.dfg.value_ty(left_value) != right.dfg.value_ty(right_value) {
                    return false;
                }
                if value_map.insert(right_value, left_value).is_some() {
                    return false;
                }
            }
        }
    }

    for (&left_block, &right_block) in left_blocks.iter().zip(&right_blocks) {
        let left_instructions = left.layout.iter_inst(left_block).collect::<Vec<_>>();
        let right_instructions = right.layout.iter_inst(right_block).collect::<Vec<_>>();

        for (&left_inst, &right_inst) in left_instructions.iter().zip(&right_instructions) {
            let left_data = left.dfg.inst(left_inst);
            let right_data = right.dfg.inst(right_inst);
            let left_values = left_data.collect_values();
            let right_values = right_data.collect_values();
            if left_values.len() != right_values.len() {
                return false;
            }
            for (&left_value, &right_value) in left_values.iter().zip(&right_values) {
                if !bind_equivalent_value(left, left_value, right, right_value, &mut value_map) {
                    return false;
                }
            }

            let mut normalized_left = left.dfg.clone_inst(left_inst);
            let mut normalized_right = right.dfg.clone_inst(right_inst);
            let mut left_normalizer = IdentityNormalizer {
                values: None,
                blocks: None,
                aliases,
                current_from: left_ref,
                current_to: left_ref,
                valid: true,
            };
            normalized_left.accept_mut(&mut left_normalizer);
            let mut right_normalizer = IdentityNormalizer {
                values: Some(&value_map),
                blocks: Some(&block_map),
                aliases,
                current_from: right_ref,
                current_to: left_ref,
                valid: true,
            };
            normalized_right.accept_mut(&mut right_normalizer);

            if !left_normalizer.valid
                || !right_normalizer.valid
                || !normalized_left.structurally_eq(normalized_right.as_ref())
            {
                return false;
            }
        }
    }

    true
}

fn bind_equivalent_value(
    left: &Function,
    left_value: sonatina_ir::ValueId,
    right: &Function,
    right_value: sonatina_ir::ValueId,
    value_map: &mut FxHashMap<sonatina_ir::ValueId, sonatina_ir::ValueId>,
) -> bool {
    if let Some(&mapped) = value_map.get(&right_value) {
        return mapped == left_value;
    }

    let equivalent = match (left.dfg.value(left_value), right.dfg.value(right_value)) {
        (
            Value::Immediate {
                imm: left_imm,
                ty: left_ty,
            },
            Value::Immediate {
                imm: right_imm,
                ty: right_ty,
            },
        ) => left_imm == right_imm && left_ty == right_ty,
        (
            Value::Global {
                gv: left_global,
                ty: left_ty,
            },
            Value::Global {
                gv: right_global,
                ty: right_ty,
            },
        ) => left_global == right_global && left_ty == right_ty,
        (
            Value::Arg {
                ty: left_ty,
                idx: left_index,
            },
            Value::Arg {
                ty: right_ty,
                idx: right_index,
            },
        ) => left_ty == right_ty && left_index == right_index,
        (Value::Undef { ty: left_ty }, Value::Undef { ty: right_ty }) => left_ty == right_ty,
        _ => false,
    };

    if equivalent {
        value_map.insert(right_value, left_value);
    }
    equivalent
}

struct IdentityNormalizer<'a> {
    values: Option<&'a FxHashMap<sonatina_ir::ValueId, sonatina_ir::ValueId>>,
    blocks: Option<&'a FxHashMap<sonatina_ir::BlockId, sonatina_ir::BlockId>>,
    aliases: &'a FxHashMap<FuncRef, FuncRef>,
    current_from: FuncRef,
    current_to: FuncRef,
    valid: bool,
}

impl VisitorMut for IdentityNormalizer<'_> {
    fn visit_value_id(&mut self, value: &mut sonatina_ir::ValueId) {
        let Some(values) = self.values else {
            return;
        };
        if let Some(&normalized) = values.get(value) {
            *value = normalized;
        } else {
            self.valid = false;
        }
    }

    fn visit_block_id(&mut self, block: &mut sonatina_ir::BlockId) {
        let Some(blocks) = self.blocks else {
            return;
        };
        if let Some(&normalized) = blocks.get(block) {
            *block = normalized;
        } else {
            self.valid = false;
        }
    }

    fn visit_func_ref(&mut self, function: &mut FuncRef) {
        if *function == self.current_from {
            *function = self.current_to;
        }
        *function = canonical_func(*function, self.aliases);
    }
}

fn collect_observed_function_identities(module: &Module) -> FxHashSet<FuncRef> {
    let mut observed = FxHashSet::default();
    for function in module.funcs() {
        if !module.ctx.func_linkage(function).has_definition() {
            continue;
        }
        module.func_store.view(function, |body| {
            let is = body.inst_set();
            for block in body.layout.iter_block() {
                for instruction in body.layout.iter_inst(block) {
                    let data = body.dfg.inst(instruction);
                    if let Some(pointer) = <&GetFunctionPtr as InstDowncast>::downcast(is, data) {
                        observed.insert(*pointer.func());
                    }
                    if let Some(address) = <&SymAddr as InstDowncast>::downcast(is, data)
                        && let SymbolRef::Func(target) = address.sym()
                    {
                        observed.insert(*target);
                    }
                    if let Some(size) = <&SymSize as InstDowncast>::downcast(is, data)
                        && let SymbolRef::Func(target) = size.sym()
                    {
                        observed.insert(*target);
                    }
                }
            }
        });
    }
    observed
}

fn collect_object_roots(module: &Module) -> FxHashSet<FuncRef> {
    module
        .objects
        .values()
        .flat_map(|object| object.sections.iter())
        .flat_map(|section| section.directives.iter())
        .filter_map(|directive| match directive {
            Directive::Entry(function) | Directive::Include(function) => Some(*function),
            _ => None,
        })
        .collect()
}

fn canonical_func(function: FuncRef, aliases: &FxHashMap<FuncRef, FuncRef>) -> FuncRef {
    let mut current = function;
    while let Some(&next) = aliases.get(&current) {
        if next == current {
            break;
        }
        current = next;
    }
    current
}

fn rewrite_function_references(module: &Module, aliases: &FxHashMap<FuncRef, FuncRef>) -> usize {
    let mut rewrites = 0;
    for function in module.funcs() {
        module.func_store.modify(function, |body| {
            for block in body.layout.iter_block() {
                for instruction in body.layout.iter_inst(block) {
                    let mut rewriter = AliasRewriter {
                        aliases,
                        rewrites: 0,
                    };
                    body.dfg.inst_mut(instruction).accept_mut(&mut rewriter);
                    rewrites += rewriter.rewrites;
                }
            }
        });
    }
    rewrites
}

struct AliasRewriter<'a> {
    aliases: &'a FxHashMap<FuncRef, FuncRef>,
    rewrites: usize,
}

impl VisitorMut for AliasRewriter<'_> {
    fn visit_func_ref(&mut self, function: &mut FuncRef) {
        let canonical = canonical_func(*function, self.aliases);
        if canonical != *function {
            *function = canonical;
            self.rewrites += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use sonatina_ir::{InstDowncast, Module, inst::control_flow::Call, module::FuncRef};

    use super::run_exact_private_func_merge;

    fn parse(source: &str) -> sonatina_parser::ParsedModule {
        sonatina_parser::parse_module(source)
            .unwrap_or_else(|errors| panic!("module should parse: {errors:?}"))
    }

    fn function(module: &Module, name: &str) -> FuncRef {
        module
            .funcs()
            .into_iter()
            .find(|&function| {
                module
                    .ctx
                    .func_sig(function, |signature| signature.name() == name)
            })
            .unwrap_or_else(|| panic!("function {name} should exist"))
    }

    fn function_names(module: &Module) -> Vec<String> {
        let mut names = module
            .funcs()
            .into_iter()
            .map(|function| {
                module
                    .ctx
                    .func_sig(function, |signature| signature.name().to_string())
            })
            .collect::<Vec<_>>();
        names.sort();
        names
    }

    fn first_direct_callee(module: &Module, function: FuncRef) -> FuncRef {
        module.func_store.view(function, |body| {
            body.layout
                .iter_block()
                .flat_map(|block| body.layout.iter_inst(block))
                .find_map(|instruction| {
                    <&Call as InstDowncast>::downcast(body.inst_set(), body.dfg.inst(instruction))
                        .map(|call| *call.callee())
                })
                .expect("function should contain a direct call")
        })
    }

    #[test]
    fn merges_exact_leaf_and_wrapper_bodies_to_a_fixed_point() {
        let source = r#"
target = "evm-ethereum-london"

func private %wrapper_a(v0.i32) -> i32 {
block0:
    v1.i32 = call %leaf_a v0;
    return v1;
}

func private %wrapper_b(v0.i32) -> i32 {
block0:
    v1.i32 = call %leaf_b v0;
    return v1;
}

func private %leaf_a(v0.i32) -> i32 {
block0:
    v1.i32 = add v0 1.i32;
    return v1;
}

func private %leaf_b(v0.i32) -> i32 {
block0:
    v1.i32 = add v0 1.i32;
    return v1;
}

func public %entry(v0.i32) -> i32 {
block0:
    v1.i32 = call %wrapper_b v0;
    return v1;
}
"#;
        let parsed = parse(source);
        let entry = function(&parsed.module, "entry");
        let wrapper_a = function(&parsed.module, "wrapper_a");
        let stats = run_exact_private_func_merge(&parsed.module, &[entry]);

        assert_eq!(stats.merged_functions, 2);
        assert!(stats.refinement_rounds >= 2);
        assert_eq!(
            function_names(&parsed.module),
            vec!["entry", "leaf_a", "wrapper_a"]
        );
        assert_eq!(first_direct_callee(&parsed.module, entry), wrapper_a);
    }

    #[test]
    fn preserves_constant_and_type_differences() {
        let source = r#"
target = "evm-ethereum-london"

func private %one(v0.i32) -> i32 {
block0:
    v1.i32 = add v0 1.i32;
    return v1;
}

func private %two(v0.i32) -> i32 {
block0:
    v1.i32 = add v0 2.i32;
    return v1;
}

func private %wide(v0.i64) -> i64 {
block0:
    v1.i64 = add v0 1.i64;
    return v1;
}
"#;
        let parsed = parse(source);
        let stats = run_exact_private_func_merge(&parsed.module, &[]);

        assert_eq!(stats.merged_functions, 0);
        assert_eq!(function_names(&parsed.module), vec!["one", "two", "wide"]);
    }

    #[test]
    fn preserves_protected_address_observed_object_and_recursive_identities() {
        let source = r#"
target = "evm-ethereum-london"

func private %protected() {
block0:
    return;
}

func private %addressed() {
block0:
    return;
}

func private %object_root() {
block0:
    return;
}

func private %recursive() {
block0:
    call %recursive;
    return;
}

func public %observer() -> i256 {
block0:
    v0.i256 = sym_addr %addressed;
    return v0;
}

object @O {
    section runtime {
        entry %object_root;
    }
}
"#;
        let parsed = parse(source);
        let protected = function(&parsed.module, "protected");
        let stats = run_exact_private_func_merge(&parsed.module, &[protected]);

        assert_eq!(stats.merged_functions, 0);
        assert_eq!(
            function_names(&parsed.module),
            vec![
                "addressed",
                "object_root",
                "observer",
                "protected",
                "recursive"
            ]
        );
    }
}
