# WebAssembly backend (integer programs)

The optional `sonatina-codegen/wasm` feature emits a WebAssembly module from
a module for the `wasm32-unknown-native` target. It is not enabled by default,
and EVM and native users do not build it. Translation uses the
[WAFFLE](https://crates.io/crates/waffle) crate, which recovers structured
control flow and encodes the module.

This slice covers integer programs. Floating point, aggregates, objects,
enums, globals, function pointers and host-memory allocation are not
translated; see [Unsupported input](#unsupported-input).

```no_run
use sonatina_codegen::{Compile, compile::OptLevel, isa::wasm::WasmBackend};
use sonatina_ir::Module;

fn compile_wasm(module: Module) -> Vec<u8> {
    Compile::new(module, WasmBackend::new())
        .with_opt_level(OptLevel::O2)
        .compile()
        .expect("wasm compilation failed")
        .into_bytes()
}
```

Compiling the same module twice produces identical bytes.

## Target and layout

The `wasm32` ISA shares the instruction set and little-endian layout of the
native ISA, with 4-byte pointers represented as `i32`. There is one address
space, the module's linear memory.

| Sonatina type | WebAssembly type |
| --- | --- |
| `i1`, `i8`, `i16`, `i32` | `i32` |
| `i64` | `i64` |
| `*T` | `i32`, an offset into linear memory |

`i128`, `i256`, enum tags, `unit`, references and aggregates cannot be
parameters, results, phi values or operands. A module that uses one is
rejected with an error naming the type.

A narrow integer is held zero-extended in its `i32`. Signed operations
sign-extend their operands and re-canonicalize results. Parameters of
exported functions and results of imports are canonicalized on entry, so a
host may pass or return any bits above the width. Results of exported functions
are canonical: an `i8` holding -1 is returned as 255.

## Module structure

- One linear memory is always emitted. It starts at one page, has no maximum
  and is exported as `memory`. The backend allocates nothing in it: addresses
  used by `mload` and `mstore` are chosen by the program or the host.
- A function with `public` linkage is exported under its symbol. `private`
  functions are only callable from the module. Export names must be unique.
- A function with `external` linkage and no body is imported as a function
  whose field name is its symbol. The module name is `env` unless
  `WasmBackend::with_import_module` selects another.
- Functions may return several values (WebAssembly multi-value).

## Instruction semantics

Integer arithmetic follows the IR semantics used by the interpreter and the
native backend:

- `add`, `sub`, `mul`, `neg` and shifts wrap to the width of the type.
- A shift by at least the bit width gives zero (`shl`, `shr`) or the sign fill
  (`sar`); WebAssembly itself would mask the count.
- `uaddo`, `usubo`, `umulo`, `saddo`, `ssubo`, `smulo` and `snego` return the
  wrapped result and an `i1` overflow flag. The saturating forms
  (`uaddsat`, `saddsat`, ...) clamp to the bound in the direction of overflow.
- `sdiv` of the minimum value by -1 wraps to the minimum value. Division or
  remainder by zero is undefined in the IR and traps in WebAssembly.
- `mload` and `mstore` access `i1`, `i8`, `i16`, `i32`, `i64` and pointers
  at an `i32` or pointer address. An access outside the memory traps. Natural
  alignment is only a hint, so unaligned addresses are valid. An `i1` load
  keeps the low bit of the byte.
- `br_table` compares the scrutinee with each case in order and traps if no
  case matches and there is no default.
- `unreachable` traps.
- An `undef` operand is materialized as zero.

## Unsupported input

Every instruction of the native instruction set is either translated or
rejected with an error that names it. These are rejected: `memzero`, `gep`,
`get_function_ptr`, `sym_addr`, `sym_size`, `alloca`, `mem.alloc_dynamic`, all
object, constant-reference and enum instructions, and `insert_value` and
`extract_value`. Global variables are rejected as a module-level error.
Errors are returned from `compile_module`; the backend does not panic on
unsupported input.
