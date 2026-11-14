# Design: one SYCL offload descriptor per link in the new offload model

Issue: https://github.com/intel/llvm/issues/23760
Code baseline: intel/llvm `sycl` at `ca54ec9715c3`. Unless noted otherwise, line numbers refer to
`clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp` at that commit.
Upstream reference: llvm-project `4c2f760c13a1526022d9c6d0ac66592c88fcc40f` (CLW = the same file
upstream, OW = `llvm/lib/Frontend/Offloading/OffloadWrapper.cpp`).

## 1. Intent

The new offload model is about to become the default. Before that, it has to cover what the old model
supports. Issue #23760 is one gap: `-fsycl-link` with more than one `-fsycl-targets` fails with
`clang-linker-wrapper: error: Expect single output from the device linker.`, while the old model
produces one device-only wrapper object.

Root cause: `clang-linker-wrapper` wraps SYCL device images **per target group**, giving one SYCL
offload descriptor and one host object per target. `--sycl-device-link` then needs exactly one output.

Decision (option B): `clang-linker-wrapper` wraps **all SYCL images of one invocation into one
descriptor**, in every mode. That fixes `-fsycl-link`. It also makes SYCL match what upstream does for
every offload kind (one wrap per kind covering all targets) and keeps the later move to the community
bundle/wrap path (NOM7) simple.

## 2. Goals, non-goals, anti-goals

Goals
- G1. `-fsycl-link` with N SYCL targets in the new model produces one wrapper object, which holds the
  device images of all targets in one descriptor.
- G2. Every new-model path in `clang-linker-wrapper` that wraps SYCL images emits one SYCL descriptor
  per invocation: normal link, `--sycl-device-link`, and the `--emit-fatbin-only` / `-fno-sycl-rdc`
  wrapper-module path. This includes the `-fsycl-embed-ir` images (`llvm_<arch>`), which today get a
  wrapper of their own (`:2254-2265`).
- G3. NativeCPU + `-fsycl-link` fails with a clear, specific "not supported" diagnostic instead of the
  generic single-output error.

Non-goals
- NG1. Making NativeCPU `-fsycl-link` work. It isn't supported by the old model (§3.3) and is tracked
  separately.
- NG2. Claiming support for `-fsycl-embed-ir` together with `-fsycl-link`. No test exercises it in
  either model (§3.4), so it isn't treated as supported. This design adds no code or tests specific to it.
- NG3. Moving SYCL onto `bundleLinkedOutput`/`wrapDeviceImages` (NOM7). This design must not make that
  harder.
- NG4. Lifting the driver restriction on `-fno-sycl-rdc` with multiple SYCL targets
  (`clang/lib/Driver/Driver.cpp:1558-1568`, `err_drv_sycl_no_rdc_multiple_targets`, #22833). That's a
  separate gap in the driver's job graph. G2 makes the wrapper side ready for it.
- NG5. Changes to the old offload model.
- NG6. SYCLBIN output. It already collects all targets and packages them once (`:3054-3061`, `:3143`).
  `-fsycl-embed-ir` with SYCLBIN is already rejected (`:3623`).
- NG7. Restoring `-fsycl-targets` order inside the descriptor (§5.3 explains why it doesn't matter).

Anti-goals (would be a failure even if the issue's repro passes)
- AG1. A fix that only works on Linux. No reliance on a relocatable host link (`ld -r`), which COFF
  doesn't support.
- AG2. Any change to the descriptor format or the SYCL runtime ABI (§4).
- AG3. Non-deterministic image order inside the descriptor.
- AG4. Silently dropping images. (Upstream does this today: it wraps only `front()` for SYCL.)

## 3. Current state (verified)

### 3.1 intel/llvm new model, `linkAndWrapDeviceFiles` (`:2942`)
1. `main` calls `linkAndWrapDeviceFiles` (`:3666`). The input groups come from `getDeviceInput`
   (`:3319`, grouping in a `MapVector` at `:3440`), in a deterministic order.
2. `parallelForEachError` runs over the groups (`:2962`). The SYCL branch (`:2993`):
   - takes per-group `LinkerArgs` from `getLinkerArgs(..., OFK_SYCL)`;
   - runs `runSYCLOffloadingPipeline` (`:2310`): llvm-link, sycl-post-link split, then
     `compileDeviceAndBundle` for each split module (`:2284`);
   - returns `vector<SplitModule>`.
3. Then one of three branches:
   - `OutputSYCLBIN` (`:3054`): stash the modules and package them after the loop.
   - `!NeedsWrapping` (`:3062`): `wrapSYCLBinariesFromFile(..., IsEmbeddedIR=false)` gives one `.bc`
     per group.
   - default (`:3072`): `runWrapperAndCompile` (`:1646`), which is `wrapSYCLBinariesFromFile`
     (`:1223`) + `runCompile` (`:1320`), gives one host `.o` **per group**.
4. Extra outputs go through `WrappedOutputCallback` during the pipeline:
   - `-fsycl-embed-ir` on NVPTX/AMDGCN: a second wrapper object (`:2254-2265`), produced before
     `compileDeviceAndBundle` rewrites `ModuleFilePath`.
   - NativeCPU: the compiled native kernel object(s) (`:2290`). These are real host objects that must
     reach the host linker.
5. After the loop, SYCL is skipped by the community bundle/wrap (`:3150`).
6. `main` requires exactly one output for `--sycl-device-link` (`:3674`) and for SYCL with
   `--emit-fatbin-only` (`:3686`).

`wrapSYCLBinariesFromFile` (`:1223`) gives each image its own target string:
- `llvm_<arch>` for embedded IR;
- `spirv64` is mapped to `spir64` (`:1250-1263`).

It then copies to the dump dir if requested, builds `SYCLImage`s, runs `compressImages` (`:1186`),
wraps everything into one `Module` and writes `.bc`.

The only per-group input it reads from `Args` is `OPT_triple_EQ`. Everything else it uses, and
everything `runCompile` uses, is global:
- `OPT_host_triple_EQ`, `OPT_compress`, `OPT_compression_level_eq`, `OPT_print_wrapped_module`,
  `OPT_shared`;
- the dump dir.

The driver adds `--compress` once per invocation (`clang/lib/Driver/ToolChains/CommonArgs.cpp:3346`).

### 3.2 Old model (from `-###`)
- `-fsycl-link`: one `clang-offload-wrapper` call with a `-kind=sycl -target=<t> -batch <table>`
  group per target, then one host `-emit-obj`. That's one descriptor across all targets.
- Normal link: one `clang-offload-wrapper` call per target, so N descriptors.
- `-fsycl-embed-ir`: the `llvm_<arch>` image gets a separate wrapper call. With `-fsycl-link`, the final
  call embeds that compiled wrapper `.o` as an `nvptx64` image, the same object-as-image pattern that
  fails for NativeCPU (§3.3).

### 3.3 NativeCPU `-fsycl-link` (run on a NativeCPU-enabled build of this tree)
A single-step NativeCPU compile and run passes in both models. Harness: `/tmp/sl23760/check/run.sh`.

| Model | Targets | `-fsycl-link` result |
|---|---|---|
| old | `native_cpu` | `clang-offload-wrapper: error: '': No such file or directory` |
| old | `native_cpu,spir64` | `clang-offload-wrapper` assertion in `SimpleTable::Row::getCell` ("column name not found") |
| new | `native_cpu` | `Expect single output from the device linker.` |
| new | `native_cpu,spir64` | `Expect single output from the device linker.` |

The old model fails the same way for direct source input and for the `-c` then `-fsycl-link a.o`
flow. `sycl/test-e2e/DeviceLib/separate_compile_test.cpp` is `XFAIL: target-native_cpu` (#20142).

### 3.4 `-fsycl-embed-ir` + `-fsycl-link` test coverage
Nothing tests this combination.
- The only E2E test using `-fsycl-embed-ir` is `sycl/test-e2e/AmdNvidiaJIT/kernel_and_bundle.cpp`, a
  single-step build.
- `%{embed-ir}` in `sycl/test-e2e/NewOffloadDriver/lit.local.cfg` is never combined with `-fsycl-link`.
- Driver LIT tests (`sycl-embed-ir.cpp`, `sycl-no-rdc-new-driver.cpp`,
  `sycl-specific-args-diagnostics.cpp`, `sycl-offload-new-driver.cpp`) don't combine it with
  `-fsycl-link` either.

So it's treated as unsupported (NG2).

## 4. ABI and runtime compatibility (verified, no break)
- **The format is unchanged.** The wrapper keeps emitting `BinDescStructVersion = 1` and
  `DeviceImageStructVersion = 3` (`llvm/lib/Frontend/Offloading/SYCLOffloadWrapper.cpp:543`, `:605`).
  Descriptors already carry several images today (device code split), each with its own target.
- **The runtime is per image.**
  - `__sycl_register_lib` → `ProgramManager::addImages` loops over `addImage(&DeviceBinaries[I])`
    (`sycl/source/detail/program_manager/program_manager.cpp:1891-1894`, `:3651`).
  - `addImage` (`:1706`) receives only the raw image.
  - `removeImages` (`:1908`) is per image, using each image's entries.
  - Descriptor-level `HostEntriesBegin/End` are null (`SYCLOffloadWrapper.cpp:603-604`).
- **Precedent:**
  - Old-model `-fsycl-link` already emits descriptors with several targets.
  - `sycl/unittests/program_manager/CompileTarget.cpp:68-98` registers one descriptor with images for
    `spir64`, `spir64_x86_64`, `intel_gpu_pvc`, `intel_gpu_skl` and `intel_gpu_bdw`.
- **Image selection doesn't depend on registration order.**
  - Images are stored in an `unordered_multimap` keyed by kernel ID (`program_manager.cpp:1309-1316`)
    and filtered by `doesImageTargetMatchDevice`.
  - The adapter then picks by target type, preferring AOT over SPIR-V (for example
    `unified-runtime/source/adapters/opencl/device.cpp:1821`).
- **Not covered by any existing test:** NativeCPU images in the same descriptor as other targets.
  `native_cpu` image data is the `__sycl_native_cpu_decls` table, handled per image like any other.
  §7 adds a NativeCPU + SPIR-V E2E run for this.

## 5. Approaches considered

| | Approach | Trade-offs | Verdict |
|---|---|---|---|
| U1a / B | Keep intel's `SYCLImage`-based wrapper. Collect per-group results in the parallel loop and wrap all SYCL images **once** after it, in every mode. | Same format and ABI. OS independent. Local change. Matches upstream's one-wrap-per-kind for every kind (OpenMP: one `__tgt_bin_desc`, CUDA/HIP: one fatbinary, CLW#L1200-L1242). Changes the normal-link layout (N descriptors → 1). Still bypasses `bundleLinkedOutput`/`wrapDeviceImages` (NOM7 later). | **Chosen** |
| U1a / A | Same as above, but only under `--sycl-device-link`. | Exact old-model parity, no layout change for normal links. Two SYCL wrapping behaviors to maintain, and further from upstream. | Rejected |
| U1b | Upstream structure: SYCL results go into `Images[OFK_SYCL]` and are wrapped in `wrapDeviceImages`. | Closest to upstream, but `OffloadingImage` doesn't carry split-module properties/symbols. Upstream's SYCL wrapper (`wrapSYCLBinaries(Module&, ArrayRef<char>)`, OW#L776) uses a different registration layout and wraps only `front()`. Much larger change. | Deferred (NOM7) |
| U2 | Keep per-target wrappers and merge them with a host relocatable link for `--sycl-device-link`. | Small, but N descriptors remain, it needs `ld -r` (no COFF), and it only fixes `-fsycl-link`. | Rejected (AG1) |

## 6. Design

### 6.1 Collect instead of wrap, inside the parallel loop
Add a per-invocation collection, protected by a mutex, next to `SYCLBINModules`:

```cpp
struct SYCLWrapInput {
  size_t GroupIndex;          // position in LinkerInputFiles; deterministic order
  llvm::Triple TargetTriple;  // the group's OPT_triple_EQ
  bool IsEmbeddedIR;          // -fsycl-embed-ir images (llvm_<arch>)
  std::vector<module_split::SplitModule> Modules;
};
```

Producers:
- Default branch (`:3072`) and `!NeedsWrapping` branch (`:3062`): push
  `{GroupIndex, Triple, false, std::move(Modules)}` instead of calling `runWrapperAndCompile` /
  `wrapSYCLBinariesFromFile`.
- `-fsycl-embed-ir` in `postLinkProcessModule` (`:2254-2265`): push a copy of the split modules
  (paths, properties, symbols) with `IsEmbeddedIR=true`, at the point where it runs today (before
  `compileDeviceAndBundle` rewrites `ModuleFilePath`), instead of calling `runWrapperAndCompile`.
- NativeCPU native objects (`:2290`): unchanged. They still go directly to the host-link outputs.
  NativeCPU's descriptor images are collected by the default branch like any other target.
- `OutputSYCLBIN`: unchanged (NG6).

### 6.2 Split `wrapSYCLBinariesFromFile`
- `Error appendSYCLImages(SmallVectorImpl<SYCLImage> &Images, ArrayRef<SplitModule> Modules,
  const llvm::Triple &TargetTriple, bool IsEmbeddedIR)` does the target-string derivation
  (`:1250-1263`), the dump-dir copy, the file read and `SYCLImage` construction. It takes the triple
  as a parameter. Keep the existing empty-triple error.
- `Expected<StringRef> wrapSYCLImages(SmallVector<SYCLImage> &Images, const ArgList &Args)` does
  `compressImages`, creates the `.sycl.image.wrapper` `.bc`, calls `wrapSYCLBinaries`, honors
  `print_wrapped_module`, and writes the bitcode. It uses only global `Args`.
- The `--dry-run`/`-v` `offload-wrapper: output: ..., input: ...` line is printed once per combined
  wrap and lists every input module with its target string.

### 6.3 Wrap once after the loop
After `parallelForEachError`, before the `Images` loop (next to `:3143`), if the collection isn't
empty:
1. Sort by `(GroupIndex, IsEmbeddedIR ? 0 : 1)`. This order is deterministic (AG3). It isn't the
   `-fsycl-targets` order, and that's intentional (NG7): the runtime doesn't select by registration
   order (§4), and new-model normal links already don't keep `-fsycl-targets` order.
2. Build one `SmallVector<SYCLImage>` by calling `appendSYCLImages` for each entry.
3. Call `wrapSYCLImages` once, with the top-level `Args`.
4. If `NeedsWrapping`, call `runCompile` once (top-level `Args`) and push the `.o` to
   `WrappedOutput`. Otherwise push the `.bc`.

Result: one SYCL wrapper output per `clang-linker-wrapper` invocation, plus NativeCPU native objects
if there are any. Both `main` checks (`:3674`, `:3686`) then pass without changes for every
non-NativeCPU case.

### 6.4 NativeCPU + `-fsycl-link` diagnostic
- **Driver (primary).** In the `-fsycl-link` validation (`Driver.cpp:1323` area): when the new offload
  model is in use and any SYCL offload toolchain triple `isNativeCPU()`, emit a new error such as
  `err_drv_sycl_link_unsupported_target` ("'-fsycl-link' is not supported for SYCL target
  'native_cpu' with the new offloading model"). This follows the pattern of
  `err_drv_sycl_no_rdc_multiple_targets` (`Driver.cpp:1563-1568`).
- **`clang-linker-wrapper` (backstop for direct invocations).** Under `--sycl-device-link`, a SYCL
  group whose triple is `isNativeCPU()` returns a specific error before device linking.
- Old model: unchanged (NG5).

### 6.5 What doesn't change
- The driver job graph for `-fsycl-link` (`Driver.cpp:7746`, `Clang.cpp:12255`).
- The SYCL runtime, the descriptor format, and `SYCLOffloadWrapper`.
- SYCLBIN, the non-SYCL offload kinds, and the `Images` loop.
- The `main` single-output checks.

## 7. Behavioral impact
- **Normal multi-target links:** one SYCL descriptor and one registration constructor instead of N.
  The same images get registered.
- **Build pipeline:** wrap and host compile run once, after all device links, instead of N times in
  parallel. Device link and compile stay parallel.
- **`-fsycl-embed-ir`:** the `llvm_<arch>` and native images share one descriptor instead of two.
- **`--save-temps`/`--dry-run` output:** one `offload-wrapper:` line and one wrapper `clang -c` per
  invocation.

## 8. Testing
- **LIT, `clang/test/OffloadTools/clang-linker-wrapper/linker-wrapper-sycl.cpp` (`--dry-run`):**
  - Two SYCL groups (for example `spir64_gen` pvc + `spir64_gen` dg2, and `spir64` +
    `spir64_x86_64`), with and without `--sycl-device-link`.
  - Expect one `offload-wrapper:` line listing both targets' images, one wrapper `clang -c`, and no
    single-output error.
  - Also cover `--emit-fatbin-only` with two groups.
- **LIT:** NVPTX `-fsycl-embed-ir` gives one wrapper with an `llvm_nvptx64` image before the
  `nvptx64` image.
- **LIT:** `--sycl-device-link` with a NativeCPU group gives the specific error.
- **Driver LIT, `clang/test/Driver/sycl-offload-new-driver.cpp`:**
  - Multi-target `-fsycl-link` gives one `clang-linker-wrapper` job with `--sycl-device-link`.
  - `-fsycl-targets=native_cpu` and `native_cpu,spir64` with `-fsycl-link` give the new driver error.
- **Existing tests:** update any whose expectations assumed one wrapper per target (for example
  `linker-wrapper-verbose.c`).
- **E2E:**
  - New: a new-model multi-target `-fsycl-link` test modeled on
    `sycl/test-e2e/DeviceLib/separate_compile_test.cpp` (link the host object with `dev.o` and run),
    for a target pair available in CI.
  - New: a new-model normal-link NativeCPU + SPIR-V run (for example `native_cpu,spir64`, run on the
    native_cpu device), covering the one mixed layout no test has exercised so far (§4). It can be
    checked locally with the NativeCPU-enabled build in `build-ncpu/`.
  - Existing new-model multi-target and embed-ir E2E tests must keep passing: `AOT/multiple-devices.cpp`,
    `AOT/fallback.cpp`, `OptionalKernelFeatures/is_compatible/is_compatible_several_targets.cpp`,
    `SpecConstants/2020/image_selection.cpp`, `AmdNvidiaJIT/kernel_and_bundle.cpp`.

## 9. Left to the builder
- How `postLinkProcessModule` hands embed-ir modules to the collection (callback or return value).
- How the group index reaches the parallel lambda (`&Input - LinkerInputFiles.data()`, or iterate
  over indices).
- Helper names and the exact format of the verbose/dry-run line.
- Wording of the two new diagnostics.
- Whether `runWrapperAndCompile` stays as a thin wrapper or is removed.
- How the work is split into PRs. Suggested order: (1) collect and wrap once, plus LIT; (2) NativeCPU
  diagnostics; (3) E2E.

## 10. Related gaps (tracked separately)
- NativeCPU `-fsycl-link` support.
- `-fsycl-embed-ir` + `-fsycl-link` (untested in both models).
- `-fno-sycl-rdc` with multiple SYCL targets in the new model (`Driver.cpp:1558-1568`).
- NOM7 / U1b: move SYCL onto `bundleLinkedOutput`/`wrapDeviceImages`. Upstream needs multi-buffer
  SYCL wrapping first.
