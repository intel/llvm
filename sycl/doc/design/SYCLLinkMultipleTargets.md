# Design: -fsycl-link with multiple -fsycl-targets in the new offload model

Issue: https://github.com/intel/llvm/issues/23760

Code references point to intel/llvm `sycl` at [`c74ecddc2fd0`](https://github.com/intel/llvm/commit/c74ecddc2fd0542450e31e45bcc46cda98214058) and llvm-project `main` at
[`b45550157944`](https://github.com/llvm/llvm-project/commit/b45550157944b2932823713c6d47ac48bd01f64a).

## Non-goals, anti-goals

Non-goals
- NG1. Making NativeCPU `-fsycl-link` work. It isn't supported by the old model.
- NG2. Claiming support for `-fsycl-embed-ir` together with `-fsycl-link`. No test exercises it in
  either model, so it isn't treated as supported. This design adds no code or tests specific to it.
- NG3. Moving SYCL onto `bundleLinkedOutput`/`wrapDeviceImages` (NOM7). This design must not make that
  harder.
- NG4. Lifting the driver restriction on `-fno-sycl-rdc` with multiple SYCL targets
  ([`Driver.cpp` L1572-L1583](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/lib/Driver/Driver.cpp#L1572-L1583), `err_drv_sycl_no_rdc_multiple_targets`, [#22833](https://github.com/intel/llvm/issues/22833), tracked in [#23526](https://github.com/intel/llvm/issues/23526)). That's a
  separate gap in the driver's job graph. On the `clang-linker-wrapper` side, `--emit-fatbin-only`
  already packages all groups into one container ([#23552](https://github.com/intel/llvm/issues/23552)).
- NG5. Changes to the old offload model.

Anti-goals (would be a failure even if the issue's repro passes)
- Any change to the descriptor format or the SYCL runtime ABI.

## Current state

This section first describes upstream llvm/llvm-project, as the reference, and then intel/llvm, which
this design changes.

### llvm/llvm-project (reference)

Upstream `clang-linker-wrapper` handles every offload kind with the same two-step structure in
`linkAndWrapDeviceFiles` ([L1104](https://github.com/llvm/llvm-project/blob/b45550157944b2932823713c6d47ac48bd01f64a/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L1104)):
1. **Link each target group, in parallel, without wrapping** ([L1117](https://github.com/llvm/llvm-project/blob/b45550157944b2932823713c6d47ac48bd01f64a/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L1117)). The device linker runs
   per group (`linkDevice`, [L621](https://github.com/llvm/llvm-project/blob/b45550157944b2932823713c6d47ac48bd01f64a/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L621)). For SYCL it runs clang with `--sycl-link`, which calls
   `clang-sycl-linker` ([L586-L590](https://github.com/llvm/llvm-project/blob/b45550157944b2932823713c6d47ac48bd01f64a/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L586-L590)) and produces one blob per group. A group can't mix SYCL
   with other offload kinds ([L1135-L1141](https://github.com/llvm/llvm-project/blob/b45550157944b2932823713c6d47ac48bd01f64a/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L1135-L1141)). Each result is stored as an
   `OffloadingImage{triple, arch}` in `Images[Kind]` ([L1185-L1187](https://github.com/llvm/llvm-project/blob/b45550157944b2932823713c6d47ac48bd01f64a/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L1185-L1187)).
2. **After the loop, bundle and wrap once per offload kind** ([L1198-L1240](https://github.com/llvm/llvm-project/blob/b45550157944b2932823713c6d47ac48bd01f64a/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L1198-L1240)): sort by
   triple/arch, `bundleLinkedOutput`, then `wrapDeviceImages`, which gives one wrapper object per
   kind.

How the kinds use step 2:
- **OpenMP.** `bundleOmpAcc` ([L907-L914](https://github.com/llvm/llvm-project/blob/b45550157944b2932823713c6d47ac48bd01f64a/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L907-L914)) writes each
  target's image as its own `OffloadBinary` entry with triple, arch and kind. `wrapOpenMPBinaries`
  ([L804-L807](https://github.com/llvm/llvm-project/blob/b45550157944b2932823713c6d47ac48bd01f64a/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L804-L807)) → `createBinDesc` ([`OffloadWrapper.cpp` L129-L197](https://github.com/llvm/llvm-project/blob/b45550157944b2932823713c6d47ac48bd01f64a/llvm/lib/Frontend/Offloading/OffloadWrapper.cpp#L129-L197)) puts all of them into **one**
  `__tgt_bin_desc`, an array of device images, and libomptarget picks the compatible image for each
  device. That is the shape of the SYCL descriptor too: an array of device images, each with its own
  target.
- **CUDA / HIP.** The architectures are merged by a vendor tool:
  `bundleCuda` ([L931-L954](https://github.com/llvm/llvm-project/blob/b45550157944b2932823713c6d47ac48bd01f64a/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L931-L954)) runs NVIDIA's `fatbinary`, and `bundleHIP` ([L957-L981](https://github.com/llvm/llvm-project/blob/b45550157944b2932823713c6d47ac48bd01f64a/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L957-L981))
  makes an AMD fat binary. The wrapper embeds that one opaque buffer (`wrapCudaBinary` /
  `wrapHIPBinary`, [L816-L825](https://github.com/llvm/llvm-project/blob/b45550157944b2932823713c6d47ac48bd01f64a/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L816-L825)), and the vendor runtime picks the architecture.

Ahead-of-time device linking:
- Upstream has no `-fsycl-link`. It uses a relocatable link: `clang++ -fsycl -r`, or
  `--offload-link -r` without an offload flag ([`OffloadingDesign.md` L482-L510](https://github.com/llvm/llvm-project/blob/b45550157944b2932823713c6d47ac48bd01f64a/clang/docs/OffloadingDesign.md#L482-L510)). `runLinker` ([L357](https://github.com/llvm/llvm-project/blob/b45550157944b2932823713c6d47ac48bd01f64a/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L357)) does
  the host relocatable link, then `relocateOffloadSection` ([L306](https://github.com/llvm/llvm-project/blob/b45550157944b2932823713c6d47ac48bd01f64a/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L306), [L383-L384](https://github.com/llvm/llvm-project/blob/b45550157944b2932823713c6d47ac48bd01f64a/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L383-L384))
  renames the offloading sections so a later link doesn't process them again.
- SYCL support for `-r` landed in [llvm/llvm-project#242098](https://github.com/llvm/llvm-project/pull/242098)
  (2026-11-14): the linked image goes into `.sycl_fatbin` ([`OffloadWrapper.cpp` L644](https://github.com/llvm/llvm-project/blob/b45550157944b2932823713c6d47ac48bd01f64a/llvm/lib/Frontend/Offloading/OffloadWrapper.cpp#L644)).
- Differences from `-fsycl-link`: the `-r` output also contains the host code, and `-r` isn't
  supported for COFF ([L311](https://github.com/llvm/llvm-project/blob/b45550157944b2932823713c6d47ac48bd01f64a/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L311)).

What this design takes from upstream: the structure (link per group, then wrap all images of a kind
once) and the OpenMP model of one descriptor holding one entry per target. It doesn't take the
upstream SYCL wrap, which wraps only one blob.

### intel/llvm new model, `linkAndWrapDeviceFiles` ([L3034](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L3034))
1. `main` calls `linkAndWrapDeviceFiles` ([L3925](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L3925)). The input groups come from `getDeviceInput`
   ([L3577](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L3577), grouping in a `MapVector` at [L3694](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L3694)), in a deterministic order.
2. `parallelForEachError` runs over the groups ([L3058](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L3058)). The SYCL branch ([L3090](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L3090)):
   - takes per-group `LinkerArgs` from `getLinkerArgs(..., OFK_SYCL)`;
   - runs `runSYCLOffloadingPipeline` ([L2373](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L2373)): llvm-link, sycl-post-link split, then
     `compileDeviceAndBundle` for each split module ([L2327-L2328](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L2327-L2328));
   - returns `vector<SplitModule>`.
3. Then one of three branches:
   - `OutputSYCLBIN` ([L3160](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L3160)): stash the modules and package them after the loop.
   - `EmitFatbinOnly` ([L3168-L3194](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L3168-L3194), [#23552](https://github.com/intel/llvm/issues/23552)): `createSYCLImages` turns the modules into
     `SYCLImage`s, which go into `Images[OFK_SYCL]` with their SYCL metadata as string data
     (`addSYCLImageStrings`). After the loop, all groups' images are written as **one** offload
     container ([L3304-L3324](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L3304-L3324)), and the host compilation wraps it. NativeCPU and NVPTX/AMDGCN
     `--sycl-embed-ir` are rejected in this mode ([L3109-L3118](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L3109-L3118)).
   - default ([L3196-L3200](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L3196-L3200)): `runWrapperAndCompile` ([L1679](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L1679)), which is
     `wrapSYCLBinariesFromFile` ([L1290](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L1290)) + `runCompile` ([L1353](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L1353)), gives one host `.o`
     **per group**.
4. Extra outputs go through `WrappedOutputCallback` during the pipeline:
   - `-fsycl-embed-ir` on NVPTX/AMDGCN: a second wrapper object ([L2296-L2309](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L2296-L2309)), produced before
     `compileDeviceAndBundle` rewrites `ModuleFilePath`.
   - NativeCPU: the compiled native kernel object(s) ([L2353-L2358](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L2353-L2358)). These are real host objects that
     must reach the host linker.
5. After the loop, SYCL is skipped by the community bundle/wrap unless `EmitFatbinOnly` is set
   ([L3289-L3290](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L3289-L3290)).
6. `main` requires exactly one output for `--sycl-device-link`, and emits an empty wrapper object when
   there is none ([#23545](https://github.com/intel/llvm/issues/23545)) ([L3935-L3946](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L3935-L3946)).

`wrapSYCLBinariesFromFile` ([L1290](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L1290)) takes the group's triple as a parameter and calls
`createSYCLImages` ([L1238](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L1238)), which gives each image its own target string:
- `llvm_<arch>` for embedded IR;
- `spirv64` is mapped to `spir64` ([L1242-L1249](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L1242-L1249)).

`createSYCLImages` also copies to the dump dir if requested, reads the files (empty buffers under
`--dry-run`, [L1266-L1268](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L1266-L1268)) and runs `compressImages` ([L1195](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L1195)). `wrapSYCLBinariesFromFile`
then calls `defineRegisterLibFuncs` ([L1090](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L1090), [#23388](https://github.com/intel/llvm/issues/23388)), wraps everything into one
`Module` and writes `.bc`.

The per-group inputs of the wrap step are:
- the triple (passed to `createSYCLImages` as `Target`);
- the triple and arch read from `Args` by `defineRegisterLibFuncs`. It defines the
  `__sycl_registerlib_*` functions that host objects reference (collected in `SYCLRegisterLibFuncMap`
  by `(triple, arch)`, [L3521-L3522](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L3521-L3522)), for every key compatible with the group's target.

Everything else the wrap step and `runCompile` use is global:
- `OPT_host_triple_EQ`, `OPT_compress`, `OPT_compression_level_eq`, `OPT_print_wrapped_module`,
  `OPT_shared`;
- the dump dir.

The driver adds `--compress` once per invocation ([`CommonArgs.cpp` L3378](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/lib/Driver/ToolChains/CommonArgs.cpp#L3378)).

### 3.2 Old model (from `-###`, observed on a build of [`ca54ec9715c3`](https://github.com/intel/llvm/commit/ca54ec9715c39ec605f6e44f493a0ad6bd87894f))
- `-fsycl-link`: one `clang-offload-wrapper` call with a `-kind=sycl -target=<t> -batch <table>`
  group per target, then one host `-emit-obj`. That's one descriptor across all targets.
- Normal link: one `clang-offload-wrapper` call per target, so N descriptors.
- `-fsycl-embed-ir`: the `llvm_<arch>` image gets a separate wrapper call. With `-fsycl-link`, the final
  call embeds that compiled wrapper `.o` as an `nvptx64` image, the same object-as-image pattern that
  fails for NativeCPU (§3.3).

### 3.3 NativeCPU `-fsycl-link` (run on a NativeCPU-enabled build of [`ca54ec9715c3`](https://github.com/intel/llvm/commit/ca54ec9715c39ec605f6e44f493a0ad6bd87894f))
A single-step NativeCPU compile and run passes in both models.

| Model | Targets | `-fsycl-link` result |
|---|---|---|
| old | `native_cpu` | `clang-offload-wrapper: error: '': No such file or directory` |
| old | `native_cpu,spir64` | `clang-offload-wrapper` assertion in `SimpleTable::Row::getCell` ("column name not found") |
| new | `native_cpu` | `Expect single output from the device linker.` |
| new | `native_cpu,spir64` | `Expect single output from the device linker.` |

The old model fails the same way for direct source input and for the `-c` then `-fsycl-link a.o`
flow. [`sycl/test-e2e/DeviceLib/separate_compile_test.cpp`](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/sycl/test-e2e/DeviceLib/separate_compile_test.cpp) is `XFAIL: target-native_cpu` ([#20142](https://github.com/intel/llvm/issues/20142)).

### 3.4 `-fsycl-embed-ir` + `-fsycl-link` test coverage
Nothing tests this combination.
- The only E2E test using `-fsycl-embed-ir` is [`sycl/test-e2e/AmdNvidiaJIT/kernel_and_bundle.cpp`](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/sycl/test-e2e/AmdNvidiaJIT/kernel_and_bundle.cpp),
  a single-step build.
- `%{embed-ir}` in [`sycl/test-e2e/NewOffloadDriver/lit.local.cfg`](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/sycl/test-e2e/NewOffloadDriver/lit.local.cfg) is never combined with
  `-fsycl-link`.
- Driver LIT tests ([`sycl-embed-ir.cpp`](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/test/Driver/sycl-embed-ir.cpp),
  [`sycl-no-rdc-new-driver.cpp`](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/test/Driver/sycl-no-rdc-new-driver.cpp),
  [`sycl-specific-args-diagnostics.cpp`](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/test/Driver/sycl-specific-args-diagnostics.cpp),
  [`sycl-offload-new-driver.cpp`](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/test/Driver/sycl-offload-new-driver.cpp)) don't combine it
  with `-fsycl-link` either.

So it's treated as unsupported (NG2).

## 4. ABI and runtime compatibility (verified, no break)
- **The format is unchanged.** The wrapper keeps emitting `BinDescStructVersion = 1` and
  `DeviceImageStructVersion = 3` ([`SYCLOffloadWrapper.cpp` L565](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/llvm/lib/Frontend/Offloading/SYCLOffloadWrapper.cpp#L565), [`SYCLOffloadWrapper.cpp` L627](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/llvm/lib/Frontend/Offloading/SYCLOffloadWrapper.cpp#L627)). Descriptors already carry several images
  today (device code split, and all architectures of a `-fno-sycl-rdc` compile since [#23492](https://github.com/intel/llvm/issues/23492)), each
  with its own target.
- **The runtime is per image.**
  - `__sycl_register_lib` → `ProgramManager::addImages` loops over `addImage(&DeviceBinaries[I])`
    ([`program_manager.cpp` L1917-L1920](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/sycl/source/detail/program_manager/program_manager.cpp#L1917-L1920), [`program_manager.cpp` L3686-L3688](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/sycl/source/detail/program_manager/program_manager.cpp#L3686-L3688)).
  - `addImage` ([`program_manager.cpp` L1732](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/sycl/source/detail/program_manager/program_manager.cpp#L1732)) receives only the raw image.
  - `removeImages` ([`program_manager.cpp` L1934](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/sycl/source/detail/program_manager/program_manager.cpp#L1934)) is per image, using each image's entries.
  - Descriptor-level `HostEntriesBegin/End` are null ([`SYCLOffloadWrapper.cpp` L625-L626](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/llvm/lib/Frontend/Offloading/SYCLOffloadWrapper.cpp#L625-L626)).
- **Precedent:**
  - Old-model `-fsycl-link` already emits descriptors with several targets.
  - [`CompileTarget.cpp` L68-L98](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/sycl/unittests/program_manager/CompileTarget.cpp#L68-L98) registers one descriptor with images for `spir64`, `spir64_x86_64`, `intel_gpu_pvc`,
    `intel_gpu_skl` and `intel_gpu_bdw`.
- **Image selection doesn't depend on registration order.**
  - Images are stored in an `unordered_multimap` keyed by kernel ID ([`program_manager.cpp` L1325-L1346](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/sycl/source/detail/program_manager/program_manager.cpp#L1325-L1346)) and filtered by
    `doesImageTargetMatchDevice`.
  - The adapter then picks by target type, preferring AOT over SPIR-V (for example [`device.cpp` L1865](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/unified-runtime/source/adapters/opencl/device.cpp#L1865)).
- **Not covered by any existing test:** NativeCPU images in the same descriptor as other targets.
  `native_cpu` image data is the `__sycl_native_cpu_decls` table, handled per image like any other.
  §8 adds a NativeCPU + SPIR-V E2E run for this.

## 5. Approaches considered

| | Approach | Trade-offs | Verdict |
|---|---|---|---|
| U1a / B | Keep intel's `SYCLImage`-based wrapper. Collect per-group results in the parallel loop and wrap all SYCL images **once** after it, in every mode. | Same format and ABI. OS independent. Local change. Matches upstream's one-wrap-per-kind for every kind (OpenMP: one `__tgt_bin_desc`, CUDA/HIP: one fatbinary, [upstream `ClangLinkerWrapper.cpp` L1198-L1240](https://github.com/llvm/llvm-project/blob/b45550157944b2932823713c6d47ac48bd01f64a/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L1198-L1240)), and what `--emit-fatbin-only` already does for SYCL ([#23552](https://github.com/intel/llvm/issues/23552)). Changes the normal-link layout (N descriptors → 1). Still bypasses `bundleLinkedOutput`/`wrapDeviceImages` (NOM7 later). | **Chosen** |
| U1a / A | Same as above, but only under `--sycl-device-link`. | Exact old-model parity, no layout change for normal links. Two SYCL wrapping behaviors to maintain, and further from upstream. | Rejected |
| U1b | Upstream structure: SYCL results go into `Images[OFK_SYCL]` and are wrapped in `wrapDeviceImages`. | Closest to upstream. `Images[OFK_SYCL]` can now carry the SYCL metadata ([#23552](https://github.com/intel/llvm/issues/23552)), but `case OFK_SYCL` in `wrapDeviceImages` uses the upstream blob wrapper (`wrapSYCLBinaries(Module&, ArrayRef<char>)`, [upstream `OffloadWrapper.cpp` L776](https://github.com/llvm/llvm-project/blob/b45550157944b2932823713c6d47ac48bd01f64a/llvm/lib/Frontend/Offloading/OffloadWrapper.cpp#L776)), a different registration layout that wraps only `front()` ([upstream `ClangLinkerWrapper.cpp` L826-L834](https://github.com/llvm/llvm-project/blob/b45550157944b2932823713c6d47ac48bd01f64a/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L826-L834)). Much larger change. | Deferred (NOM7) |
| U2 | Keep per-target wrappers and merge them with a host relocatable link for `--sycl-device-link`. | Small, but N descriptors remain, it needs `ld -r` (no COFF), and it only fixes `-fsycl-link`. | Rejected |

## 6. Design

### 6.1 Collect images instead of wrapping, inside the parallel loop
Add a per-invocation collection, protected by a mutex, next to `SYCLBINModules`:

```cpp
struct SYCLWrapGroup {
  size_t GroupIndex;                          // position in LinkerInputFiles; deterministic order
  OffloadFile::TargetID Target;               // the group's triple and arch
  SmallVector<offloading::SYCLImage> Images;  // from createSYCLImages
};
```

Producers:
- Default branch ([L3196-L3200](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L3196-L3200)): push
  `{GroupIndex, Target, createSYCLImages(Modules, LinkerArgs, /*IsEmbeddedIR=*/false, TargetTriple)}`
  instead of calling `runWrapperAndCompile`.
- `-fsycl-embed-ir` in `postLinkProcessModule` ([L2296-L2309](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L2296-L2309)): push the result of
  `createSYCLImages(SplitModules, LinkerArgs, /*IsEmbeddedIR=*/true, Triple)` instead of calling
  `runWrapperAndCompile`. `createSYCLImages` reads the files at that point, before
  `compileDeviceAndBundle` rewrites `ModuleFilePath`, so nothing needs to be copied.
- NativeCPU native objects ([L2353-L2358](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L2353-L2358)): unchanged. They still go directly to the host-link
  outputs. NativeCPU's descriptor images are collected by the default branch like any other target.
- `EmitFatbinOnly` ([L3168-L3194](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L3168-L3194)) and `OutputSYCLBIN`: unchanged.

### 6.2 Wrap once after the loop
After `parallelForEachError`, next to the SYCLBIN packaging ([L3281-L3286](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L3281-L3286)), if the collection
isn't empty:
1. Stable-sort by `GroupIndex`. Within a group, the embedded-IR images come first, because they are
   pushed inside the pipeline before the default branch. This order is deterministic. It isn't the
   `-fsycl-targets` order, and that's intentional: the runtime doesn't select by registration order
   (§4), and new-model normal links already don't keep `-fsycl-targets` order.
2. Build one wrapper module (the part of `wrapSYCLBinariesFromFile` after `createSYCLImages`,
   [L1290](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L1290)), with the top-level `Args`:
   - call `defineRegisterLibFuncs` for **every** collected `Target`. Today it reads the triple and arch
     from the per-group `Args` ([L1090](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L1090)), so it has to take the target as a
     parameter instead;
   - call `wrapSYCLBinaries` once with all images;
   - honor `print_wrapped_module`, and write the bitcode.
3. Call `runCompile` once ([L1353](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L1353), top-level `Args`) and push the `.o` to `WrappedOutput`.
4. Print the `--dry-run`/`-v` `offload-wrapper: output: ..., input: ...` line once, listing every
   input with its target string. Under `--dry-run`, `createSYCLImages` already substitutes empty
   buffers ([L1266-L1268](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L1266-L1268)), so collecting still works.

Result: one SYCL wrapper output per `clang-linker-wrapper` invocation, plus NativeCPU native objects
if there are any. The `--sycl-device-link` check in `main` ([L3935-L3946](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L3935-L3946)) then passes without
changes for every non-NativeCPU case.

### 6.3 NativeCPU + `-fsycl-link` diagnostic
- **Driver (primary).** In the `-fsycl-link` validation ([`Driver.cpp` L1337-L1344](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/lib/Driver/Driver.cpp#L1337-L1344)): when the new offload
  model is in use and any SYCL offload toolchain triple `isNativeCPU()`, emit an error such as
  "'-fsycl-link' is not supported with '-fsycl-targets=native_cpu'". This follows the pattern of
  `err_drv_sycl_no_rdc_unsupported` ([`Driver.cpp` L1584-L1595](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/lib/Driver/Driver.cpp#L1584-L1595)), which rejects NativeCPU and
  `-fsycl-embed-ir` with `-fno-sycl-rdc`.
- **`clang-linker-wrapper` (backstop for direct invocations).** Under `--sycl-device-link`, a SYCL
  group whose triple is `isNativeCPU()` returns a specific error before device linking, like the
  existing `--emit-fatbin-only` NativeCPU error ([L3109-L3118](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L3109-L3118)).
- Old model: unchanged (NG5).

### 6.4 What doesn't change
- The driver side of `-fsycl-link` (`--sycl-device-link` is still passed as today, [`Clang.cpp` L12319-L12320](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/lib/Driver/ToolChains/Clang.cpp#L12319-L12320)).
- The SYCL runtime, the descriptor format, and `SYCLOffloadWrapper`.
- The `--emit-fatbin-only` and SYCLBIN paths, the non-SYCL offload kinds, and the `Images` loop.
- The `main` single-output check.

## 7. Behavioral impact
- **Normal multi-target links:** one SYCL descriptor and one registration constructor instead of N.
  The same images get registered.
- **Build pipeline:** wrap and host compile run once, after all device links, instead of N times in
  parallel. Device link and compile stay parallel.
- **`-fsycl-embed-ir`:** the `llvm_<arch>` and native images share one descriptor instead of two.
- **`--save-temps`/`--dry-run` output:** one `offload-wrapper:` line and one wrapper `clang -c` per
  invocation.

## 8. Testing
- **LIT, [`linker-wrapper-sycl.cpp`](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/test/OffloadTools/clang-linker-wrapper/linker-wrapper-sycl.cpp) (`--dry-run`):**
  - Two SYCL groups (for example `spir64_gen` pvc + `spir64_gen` dg2, and `spir64` +
    `spir64_x86_64`), with and without `--sycl-device-link`.
  - Expect one `offload-wrapper:` line listing both targets' images, one wrapper `clang -c`, and no
    single-output error.
- **LIT:** NVPTX `-fsycl-embed-ir` gives one wrapper with an `llvm_nvptx64` image before the
  `nvptx64` image.
- **LIT:** with two groups whose host objects reference `__sycl_registerlib_*` functions ([#23388](https://github.com/intel/llvm/issues/23388)), the
  combined wrapper defines the functions for both targets.
- **LIT:** `--sycl-device-link` with a NativeCPU group gives the specific error.
- **Driver LIT, [`sycl-offload-new-driver.cpp`](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/test/Driver/sycl-offload-new-driver.cpp):**
  - Multi-target `-fsycl-link` gives one `clang-linker-wrapper` job with `--sycl-device-link`.
  - `-fsycl-targets=native_cpu` and `native_cpu,spir64` with `-fsycl-link` give the new driver error.
- **Existing tests:** update any whose expectations assumed one wrapper per target (for example
  [`linker-wrapper-verbose.c`](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/test/OffloadTools/clang-linker-wrapper/linker-wrapper-verbose.c)).
- **E2E:**
  - New: a new-model multi-target `-fsycl-link` test modeled on
    [`separate_compile_test.cpp`](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/sycl/test-e2e/DeviceLib/separate_compile_test.cpp) (link the host
    object with `dev.o` and run), for a target pair available in CI.
  - New: a new-model normal-link NativeCPU + SPIR-V run (for example `native_cpu,spir64`, run on the
    native_cpu device), covering the one mixed layout no test has exercised so far (§4).
  - Existing new-model multi-target and embed-ir E2E tests must keep passing:
    [`AOT/multiple-devices.cpp`](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/sycl/test-e2e/AOT/multiple-devices.cpp),
    [`AOT/fallback.cpp`](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/sycl/test-e2e/AOT/fallback.cpp),
    [`OptionalKernelFeatures/is_compatible/is_compatible_several_targets.cpp`](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/sycl/test-e2e/OptionalKernelFeatures/is_compatible/is_compatible_several_targets.cpp),
    [`SpecConstants/2020/image_selection.cpp`](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/sycl/test-e2e/SpecConstants/2020/image_selection.cpp),
    [`AmdNvidiaJIT/kernel_and_bundle.cpp`](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/sycl/test-e2e/AmdNvidiaJIT/kernel_and_bundle.cpp),
    [`SeparateCompile/empty_device_link.cpp`](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/sycl/test-e2e/SeparateCompile/empty_device_link.cpp).

## 9. Left to the builder
- How `postLinkProcessModule` hands embed-ir images to the collection (callback or return value).
- How the group index reaches the parallel lambda (`&Input - LinkerInputFiles.data()`, or iterate
  over indices).
- Helper names and the exact format of the verbose/dry-run line.
- Wording of the two new diagnostics.
- Whether `runWrapperAndCompile` and `wrapSYCLBinariesFromFile` stay as thin wrappers or are removed.
- How the work is split into PRs. Suggested order: (1) collect and wrap once, plus LIT; (2) NativeCPU
  diagnostics; (3) E2E.

## 10. Related gaps (tracked separately)
- NativeCPU `-fsycl-link` support.
- `-fsycl-embed-ir` + `-fsycl-link` (untested in both models).
- `-fno-sycl-rdc` with multiple SYCL targets in the new model ([`Driver.cpp` L1572-L1583](https://github.com/intel/llvm/blob/c74ecddc2fd0542450e31e45bcc46cda98214058/clang/lib/Driver/Driver.cpp#L1572-L1583), [#23526](https://github.com/intel/llvm/issues/23526)).
- NOM7 / U1b: move SYCL onto `bundleLinkedOutput`/`wrapDeviceImages`. Upstream needs multi-buffer
  SYCL wrapping first.
