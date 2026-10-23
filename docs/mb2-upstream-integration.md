# MB2 upstream integration checkpoint

Experimental branch only. Do not use as a production dependency pin yet.

History replay completed at e68fa172 on upstream 6dcaaa557. Original MB2 and naming branches are preserved. Current follow-up work ports capability-token construction, BackendOptions, native float lowering and range-analysis APIs.

# Required after history replay, before any dependency update

The upstream native backend replaces the historical monolithic translator and
its JIT artifact API with separate object/JIT backends and an options contract.
Do not resurrect the old translator or name-substring intrinsic interception.

- Port all MB2 f32 carrier/immediate/ABI and numeric instruction lowering into
  the modern translator; run existing native numeric intrinsic regressions.
- Preserve exact signed-zero/NaN contracts (including later relaxed variants).
- Preserve or replace the old u256 intrinsic capability with typed lowering;
  run modular-arithmetic and native tests, not only shader/Wasm tests.
- Adapt MB2 instruction construction to upstream capability tokens or the
  explicit runtime-checked new_unchecked API. Never weaken upstream checks.
- Adapt custom backends/Fe callers to BackendOptions; keep the full latest
  upstream optimization pipeline and native/SP1 target support.
- Regenerate lockfile only after manifest history replay completes; validate
  all enabled feature combinations and unchanged app numerical oracles.

Library checks now pass with shader/Wasm and with shader/Wasm/native JIT features. This is not yet a replacement pin.
Native translator history conflicts preserve the upstream architecture; float
ports above are mandatory follow-up work, not silently dropped features.

- a456f2cc allocation port: constant-sized MemAllocDynamic, reject loop-carried/nonconstant allocation, zero-initialize fresh storage, preserve distinct allocations and correct native pointer widths. Retained five native regressions must pass against modern backend; do not indiscriminately widen integer arithmetic.

- d73cea9b / b8cf282a: odd-sized zero-fill alignment and pointer-round-up mask regression must pass; negative i32 alignment masks must preserve high native pointer bits. Retain typed integer semantics rather than a blanket mixed-width coercion.

- 71a8aa2d: ordered Feq/Flt/Fle (NaN behavior), and >2 scalar result private ABI; adapt to modern upstream ABI rather than resurrecting obsolete hard-coded 16-byte-slot convention. Retained native tests are release gates.

- 92a0cd0b: native mixed-width pointer/integer comparison regression; signed comparisons sign-extend, unsigned zero-extend when logical address carrier is narrower than actual pointer. Preserve verifier type contracts.

- 09416b91 aggregate reconciliation: upstream already has exact sparse interval LiveLeaves, sparse layout counts and scalarization extent checks; retain modern ObjectAccessFacts API. Ported remaining whole_object_slice count-only query to shape_leaf_count. Run large-array/interval regression gates before pin.

- 6e50de9a: explicit typed external modular runtime declarations may resolve only exact ABI/name, authored functions must never be shadowed. Preserve modern import/linkage architecture; retained shadowing/wrong-signature native tests remain gates.

## Validation checkpoint

- Locked library check with `spirv-backend,wasm`: PASS (`check-7.log`).
- Locked library check with `spirv-backend,wasm,cranelift-jit`: PASS (`check-native-3.log`).
- Readable shader field names: focused regression is running.
- Native numerical/ABI/allocation regressions, SP1 build, Fe frontend adaptation and application GPU oracle checks: pending.

Native object atomics and dynamic arena checkpoints currently fail closed. Native
allocation follows upstream's host-heap model; the old stack-allocation tests
need reconciliation against its initialization and lifetime contract before use.
Typed modular arithmetic now declares a fixed runtime ABI and rejects authored
symbol conflicts; runtime execution and object-linking tests remain pending.

Raw build receipts live locally under
`/lab/argot/scratch/mb2-sonatina-upstream-rebase-20261022/`.
