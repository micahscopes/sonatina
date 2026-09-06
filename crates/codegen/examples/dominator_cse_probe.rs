//! Measure exact scalar CSE on a named function in an existing IR capture.
//! This measures IR only, not emitted shader bytes or GPU execution time.
use std::{env, fs, time::Instant};

use sonatina_codegen::{domtree::DomTree, optim::dominator_cse::eliminate_dominated_expressions};
use sonatina_ir::{ControlFlowGraph, Function};
use sonatina_verifier::{VerificationLevel, VerifierConfig, verify_function};

fn instruction_count(func: &Function) -> usize {
    func.layout
        .iter_block()
        .map(|block| func.layout.iter_inst(block).count())
        .sum()
}

fn main() {
    let args: Vec<_> = env::args().collect();
    assert!(
        args.len() == 3 || (args.len() == 4 && args[3] == "--conditional-bridges"),
        "usage: dominator_cse_probe INPUT.sona FUNCTION [--conditional-bridges]"
    );
    let source = fs::read_to_string(&args[1]).expect("read IR capture");
    let module = sonatina_parser::parse_module(&source)
        .expect("parse IR capture")
        .module;
    let matches: Vec<_> = module
        .funcs()
        .into_iter()
        .filter(|id| module.ctx.func_sig(*id, |sig| sig.name() == args[2]))
        .collect();
    assert_eq!(matches.len(), 1, "expected exactly one matching function");
    let id = matches[0];
    let config = VerifierConfig::for_level(VerificationLevel::Full);
    module.func_store.modify(id, |func| {
        // The text parser reserves every block ID up to the largest label,
        // including holes in optimized captures. Remove only empty, unlisted
        // placeholders; full verification below still rejects dangling uses.
        let placeholders: Vec<_> = func.dfg.block_ids()
            .filter(|block| !func.layout.is_block_inserted(*block))
            .collect();
        let placeholder_count = placeholders.len();
        for block in placeholders {
            func.erase_block(block);
        }
        let before = verify_function(&module.ctx, id, func, &config);
        assert!(!before.has_errors(), "invalid input: {before:?}");
        let instructions_before = instruction_count(func);
        let started = Instant::now();
        let mut cfg = ControlFlowGraph::default();
        cfg.compute(func);
        let mut dom = DomTree::default();
        dom.compute(&cfg);
        let analysis_us = started.elapsed().as_micros();
        let started = Instant::now();
        let stats = eliminate_dominated_expressions(func, &dom);
        let cse_us = started.elapsed().as_micros();
        let instructions_after = instruction_count(func);
        let after = verify_function(&module.ctx, id, func, &config);
        assert!(!after.has_errors(), "invalid output: {after:?}");
        assert_eq!(
            eliminate_dominated_expressions(func, &dom).removed_instructions,
            0,
            "CSE was not idempotent"
        );
        println!("function={} before={} after={} removed={} reused_values={} peak_expressions={} analysis_us={} cse_us={} parser_empty_blocks={} verified=true",
            args[2], instructions_before, instructions_after,
            stats.removed_instructions, stats.reused_values,
            stats.peak_available_expressions, analysis_us, cse_us, placeholder_count);
        if args.len() == 4 {
            use sonatina_codegen::{cfg_edit::{CfgEditor, CleanupMode}, structurize::structurize_function};
            let structure_before = structurize_function(func).map(|s| s.stats());
            let instructions_before = instruction_count(func);
            let mut folded = 0;
            let mut editor = CfgEditor::new(func, CleanupMode::Strict);
            loop {
                let blocks: Vec<_> = editor.func().layout.iter_block().collect();
                let mut changed = false;
                for block in blocks {
                    if editor.fold_conditional_bridge(block) {
                        folded += 1;
                        changed = true;
                    }
                }
                if !changed { break; }
            }
            let func = editor.func();
            let after = verify_function(&module.ctx, id, func, &config);
            assert!(!after.has_errors(), "invalid branch-fold output: {after:?}");
            println!("conditional_bridges={} instructions_before={} instructions_after={} verified=true", folded, instructions_before, instruction_count(func));
            println!("structure_before={structure_before:?}");
            println!("structure_after={:?}", structurize_function(func).map(|s| s.stats()));
        }
    });
}
