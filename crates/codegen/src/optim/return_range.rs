//! Conservative result bounds across direct calls, without inlining bodies.
use super::range_branch_simplify::RangeBranchSimplify;
use crate::{
    domtree::DomTree,
    loop_analysis::LoopTree,
    range_analysis::{RangeAnalysis, RangeEnv, RangeFact, fact_for_value, join_facts},
};
use rustc_hash::FxHashMap;
use sonatina_ir::{
    ControlFlowGraph, Function, Type,
    inst::{control_flow::Return, downcast},
    module::{FuncRef, Module},
};

type Summaries = FxHashMap<FuncRef, Vec<Option<RangeFact>>>;

#[derive(Default, Debug)]
pub struct ReturnRangeStats {
    pub rounds: usize,
    pub bounded_results: usize,
    pub simplified_functions: usize,
}

/// Each round starts from previously proved bounds (initially none). Stopping
/// early only loses precision. Unknown calls, non-integer lanes, and functions
/// without a reachable return contribute no fact. Calls and their effects are
/// never removed by this analysis. Summaries are valid for arbitrary arguments.
pub fn run_return_range_branch_simplify(module: &Module, max_rounds: usize) -> ReturnRangeStats {
    let mut summaries = Summaries::default();
    let mut stats = ReturnRangeStats::default();
    for _ in 0..max_rounds {
        let mut next = Summaries::default();
        for reference in module.funcs() {
            let signature = module.ctx.get_sig(reference).expect("module signature");
            if !module.ctx.func_linkage(reference).has_definition()
                || signature.ret_tys().is_empty()
            {
                continue;
            }
            let result = module.func_store.view(reference, |function| {
                summarize(function, signature.ret_tys(), &summaries)
            });
            if let Some(result) = result {
                next.insert(reference, result);
            }
        }
        stats.rounds += 1;
        if next == summaries {
            break;
        }
        summaries = next;
    }
    stats.bounded_results = summaries.values().flatten().filter(|r| r.is_some()).count();
    for reference in module.funcs() {
        if !module.ctx.func_linkage(reference).has_definition() {
            continue;
        }
        let changed = module.func_store.modify(reference, |function| {
            let results = call_results(function, &summaries);
            if results.is_empty() {
                return false;
            }
            let (cfg, lpt) = flow(function);
            RangeBranchSimplify::new().run_with_call_results(function, &cfg, &lpt, &results)
        });
        stats.simplified_functions += usize::from(changed);
    }
    stats
}

fn flow(function: &Function) -> (ControlFlowGraph, LoopTree) {
    let mut cfg = ControlFlowGraph::default();
    cfg.compute(function);
    let mut dom = DomTree::default();
    dom.compute(&cfg);
    let mut lpt = LoopTree::default();
    lpt.compute(&cfg, &dom);
    (cfg, lpt)
}

fn call_results(function: &Function, summaries: &Summaries) -> RangeEnv {
    let mut results = RangeEnv::default();
    for block in function.layout.iter_block() {
        for inst in function.layout.iter_inst(block) {
            let Some(call) = function.dfg.call_info(inst) else {
                continue;
            };
            let Some(bounds) = summaries.get(&call.callee()) else {
                continue;
            };
            for (&value, &bound) in function.dfg.inst_results(inst).iter().zip(bounds) {
                if let Some(bound) = bound {
                    results.insert(value, bound);
                }
            }
        }
    }
    results
}

fn summarize(
    function: &Function,
    types: &[Type],
    summaries: &Summaries,
) -> Option<Vec<Option<RangeFact>>> {
    let (cfg, lpt) = flow(function);
    let mut analysis = RangeAnalysis::default();
    analysis.compute_with_call_results(function, &cfg, &lpt, &call_results(function, summaries));
    let mut result = vec![None; types.len()];
    let mut saw_return = false;
    for block in function.layout.iter_block() {
        if !analysis.is_reachable(block) {
            continue;
        }
        let Some(inst) = function.layout.last_inst_of(block) else {
            continue;
        };
        let Some(ret) = downcast::<&Return>(function.inst_set(), function.dfg.inst(inst)) else {
            continue;
        };
        if ret.args().len() != types.len() {
            return None;
        }
        for (index, (&value, &ty)) in ret.args().iter().zip(types).enumerate() {
            if !ty.is_integral() {
                continue;
            }
            let fact = fact_for_value(function, analysis.exit_env(block), value);
            result[index] = Some(match result[index] {
                Some(previous) => join_facts(previous, fact, ty),
                None => fact,
            });
        }
        saw_return = true;
    }
    if !saw_return {
        return None;
    }
    for (bound, &ty) in result.iter_mut().zip(types) {
        if bound.is_some_and(|fact| fact.is_full_for(ty)) {
            *bound = None;
        }
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonatina_ir::ir_writer::FuncWriter;
    use sonatina_verifier::{VerificationLevel, VerifierConfig, verify_module};

    fn fixture(second_tag: &str) -> sonatina_parser::ParsedModule {
        sonatina_parser::parse_module(&format!(
            r#"
target = "evm-ethereum-osaka"
func private %choice(v0.i32) -> (i32, i32) {{
    block0:
        v1.i1 = eq v0 0.i32;
        br v1 block1 block2;
    block1:
        return (0.i32, v0);
    block2:
        return ({second_tag}, v0);
}}
func private %wrapper(v0.i32) -> (i32, i32) {{
    block0:
        (v1.i32, v2.i32) = call %choice v0;
        return (v1, v2);
}}
func public %entry(v0.i32) -> i32 {{
    block0:
        (v1.i32, v2.i32) = call %wrapper v0;
        v3.i1 = eq v1 0.i32;
        br v3 block3 block1;
    block1:
        v4.i1 = eq v1 1.i32;
        br v4 block3 block2;
    block2:
        unreachable;
    block3:
        return v2;
}}
"#
        ))
        .expect("valid range fixture")
    }

    fn entry(module: &Module) -> String {
        let f = module
            .funcs()
            .into_iter()
            .find(|&f| module.ctx.func_sig(f, |s| s.name() == "entry"))
            .unwrap();
        module
            .func_store
            .view(f, |body| FuncWriter::new(f, body).dump_string())
    }

    #[test]
    fn closes_tag_domain_across_wrappers_without_inlining_or_losing_payload() {
        let parsed = fixture("1.i32");
        let stats = run_return_range_branch_simplify(&parsed.module, 4);
        assert_eq!(stats.simplified_functions, 1);
        let text = entry(&parsed.module);
        assert!(!text.contains("unreachable;"), "{text}");
        assert!(text.contains("call %wrapper"), "{text}");
        assert!(text.contains("return v2;"), "{text}");
        assert!(
            !verify_module(
                &parsed.module,
                &VerifierConfig::for_level(VerificationLevel::Fast)
            )
            .has_errors()
        );
    }

    #[test]
    fn real_out_of_domain_or_unknown_results_keep_the_trap() {
        for tag in ["2.i32", "v0"] {
            let parsed = fixture(tag);
            run_return_range_branch_simplify(&parsed.module, 4);
            assert!(entry(&parsed.module).contains("unreachable;"), "{tag}");
        }
    }

    #[test]
    fn zero_budget_leaves_unknown_results_unchanged() {
        let parsed = fixture("1.i32");
        let before = entry(&parsed.module);
        let stats = run_return_range_branch_simplify(&parsed.module, 0);
        assert_eq!(stats.simplified_functions, 0);
        assert_eq!(entry(&parsed.module), before);
    }

    #[test]
    fn external_call_is_not_given_a_closed_domain() {
        let parsed = sonatina_parser::parse_module(r#"
target = "evm-ethereum-osaka"
declare external %host() -> i32;
func public %entry() -> i32 {
    block0:
        v0.i32 = call %host;
        v1.i1 = eq v0 0.i32;
        br v1 block1 block2;
    block1:
        return v0;
    block2:
        unreachable;
}
"#).expect("external fixture");
        run_return_range_branch_simplify(&parsed.module, 4);
        assert!(entry(&parsed.module).contains("unreachable;"));
        assert!(entry(&parsed.module).contains("call %host"));
    }
}
