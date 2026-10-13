# Problem: `-fno-sycl-rdc` Support for New Offload Model

## Context

The new SYCL offload model (`--offload-new-driver`) currently treats all device code as RDC (Relocatable Device Code): all TUs are merged in `clang-linker-wrapper` before post-link. `-fno-sycl-rdc` is a high-priority gap (P0) that enables per-TU self-contained device images, important for reducing link time, shipping precompiled device libraries, and scenarios where `SYCL_EXTERNAL` is not used.

The old model implements this via `shouldDoPerObjectFileLinking()` + per-TU device link steps during the compile phase in the driver. This plan implements the **linker-wrapper-side approach** as it is simpler, consistent with the new model's architecture, and can be done incrementally.

## Approach: Linker-Wrapper Non-RDC Mode

In non-RDC mode, instead of merging all device inputs into one module before post-link, each input file is processed independently through the full pipeline (device lib linking → post-link → SPIR-V/AOT), and all resulting images are wrapped together at the end.

### Current RDC flow in `linkAndWrapDeviceFiles()`:
```
[TU1.bc, TU2.bc, ...] → link all input device code → ONE_MERGED.bc
                       → link device libs  → ONE_MERGED_WITH_LIBS.bc
                       → sycl-post-link → [split0, split1, ...]
                       → per-split codegen (SPIR-V, AOT)
                       → runWrapperAndCompile() → wrapper.o
```

### Non-RDC flow:
```
For each TU.bc independently:
  TU.bc → link device libs → TU_WITH_LIBS.bc
         → sycl-post-link → [split0, ...]
         → per-split codegen (SPIR-V, AOT)
         → append to AllSplitModules
runWrapperAndCompile(AllSplitModules) → wrapper.o  (one object containing many images)
```

**Implementation note:**
* Refactor the existing codegen loop (per-split SPIR-V/AOT) into a helper lambda/function so it can be reused for both RDC and non-RDC paths without duplication.
* clang-linker-wrapper already contains a suitable command line flag `--relocatable` from upstream which should be reused. It means that Clang Driver should start pass `--relocatable` for the existing RDC mode. Later, an absence of `--relocatable` flag will imply non-rdc sycl mode.

**Implementation note on device lib extraction:** The per-TU device lib linking in non-RDC mode needs to extract compatible device lib files from the device library list. This logic can be factored out of `sycl::linkDevice()` into a helper (e.g., `sycl::collectDeviceLibFiles()`) shared by both paths.

## Post-Link Split Mode in Non-RDC

In non-RDC mode, each module is already a single TU. The split mode should respect the user's `-fsycl-device-code-split` option, but default to `SPLIT_NONE` (equivalent to per-source, since each TU is already separate).

## Error Handling

Following the old model, add validation: when `-fno-sycl-rdc` is used with `--offload-new-driver`, emit an error if AOT target is missing (consistent with `err_drv_no_rdc_sycl_target_missing` in the old model). This check belongs in `Clang.cpp` or `Driver.cpp` near the `--sycl-no-rdc` propagation.

## Key Constraints / Non-Goals

- `SYCL_EXTERNAL` across TUs is disallowed with `-fno-sycl-rdc` (already enforced by frontend, unchanged)
- SYCLBIN packaging in non-RDC mode: follows same wrapping path; no separate changes needed
- Non-RDC in `--fsycl-device-only` mode: skip linker wrapper; driver-side packaging unchanged
