use std::collections::HashMap;

use smallvec::smallvec;
use sonatina_ir::{
    BlockId, DataFlowGraph, Immediate, Type, Value, ValueId,
    inst::{control_flow::CallIndirect, data::GetFunctionPtr},
    interpret::{Action, EvalResults, EvalValue, Interpret, State},
    isa::wasm32::Wasm32,
    module::{FuncRef, ModuleCtx},
};
use sonatina_triple::{Architecture, OperatingSystem, TargetTriple, Vendor};

struct TestState {
    dfg: DataFlowGraph,
    values: HashMap<ValueId, EvalValue>,
    called: Option<(FuncRef, Vec<EvalValue>)>,
}

impl State for TestState {
    fn lookup_val(&mut self, value: ValueId) -> EvalValue {
        self.values.get(&value).cloned().unwrap_or_default()
    }
    fn call_func(&mut self, func: FuncRef, args: Vec<EvalValue>) -> EvalResults {
        self.called = Some((func, args));
        smallvec![EvalValue::Imm(Immediate::I32(9))]
    }
    fn set_action(&mut self, action: Action) {
        assert_eq!(action, Action::Continue);
    }
    fn prev_block(&mut self) -> BlockId {
        unreachable!()
    }
    fn load(&mut self, _: EvalValue, _: Type) -> EvalValue {
        unreachable!()
    }
    fn store(&mut self, _: EvalValue, _: EvalValue, _: Type) -> EvalValue {
        unreachable!()
    }
    fn alloca(&mut self, _: Type) -> EvalValue {
        unreachable!()
    }
    fn dfg(&self) -> &DataFlowGraph {
        &self.dfg
    }
}

#[test]
fn indirect_call_invokes_function_value_and_returns_results() {
    let isa = Wasm32::new(TargetTriple::new(
        Architecture::Wasm32,
        Vendor::Unknown,
        OperatingSystem::Native,
    ));
    let mut dfg = DataFlowGraph::new(ModuleCtx::new(&isa));
    let callee = dfg.make_value(Value::Arg {
        ty: Type::I32,
        idx: 0,
    });
    let arg = dfg.make_value(Value::Arg {
        ty: Type::I32,
        idx: 1,
    });
    let target = FuncRef::from_u32(7);
    let mut state = TestState {
        dfg,
        values: HashMap::from([
            (callee, EvalValue::Func(target)),
            (arg, EvalValue::Imm(Immediate::I32(4))),
        ]),
        called: None,
    };
    let results =
        CallIndirect::new(state.dfg.inst_set(), callee, Type::I32, smallvec![arg])
            .interpret(&mut state);
    assert_eq!(
        results.as_slice(),
        &[EvalValue::Imm(Immediate::I32(9))]
    );
    assert_eq!(
        state.called,
        Some((target, vec![EvalValue::Imm(Immediate::I32(4))]))
    );
}

#[test]
fn get_function_ptr_interprets_as_a_function_value() {
    let isa = Wasm32::new(TargetTriple::new(
        Architecture::Wasm32,
        Vendor::Unknown,
        OperatingSystem::Native,
    ));
    let dfg = DataFlowGraph::new(ModuleCtx::new(&isa));
    let target = FuncRef::from_u32(11);
    let mut state = TestState {
        dfg,
        values: HashMap::new(),
        called: None,
    };
    let result = GetFunctionPtr::new(state.dfg.inst_set(), target).interpret(&mut state);
    assert_eq!(result.as_slice(), &[EvalValue::Func(target)]);
}
