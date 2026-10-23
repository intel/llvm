This is the Intel staging area for llvm.org contributions and the home for
Intel LLVM-based projects:

- [oneAPI DPC++ compiler](#oneapi-dpc-compiler)
- [Late-outline OpenMP and OpenMP Offload](#late-outline-openmp-and-openmp-offload)

For general contribution process see [CONTRIBUTING.md](./CONTRIBUTING.md)

## oneAPI DPC++ compiler

[![oneAPI logo](https://spec.oneapi.io/oneapi-logo-white-scaled.jpg)](https://www.oneapi.io/)

[![SYCL Post Commit](https://github.com/intel/llvm/actions/workflows/sycl-post-commit.yml/badge.svg?branch=sycl)](https://github.com/intel/llvm/actions/workflows/sycl-post-commit.yml)
[![Generate Doxygen documentation](https://github.com/intel/llvm/actions/workflows/sycl-docs.yml/badge.svg?branch=sycl)](https://github.com/intel/llvm/actions/workflows/sycl-docs.yml)
[![Coverity](https://scan.coverity.com/projects/31090/badge.svg)](https://scan.coverity.com/projects/intel-llvm)

The DPC++ is a LLVM-based compiler project that implements compiler and runtime
support for the SYCL\* language. The project is hosted in the
[sycl](https://github.com/intel/llvm/tree/sycl) branch and is synced with the
tip of the LLVM upstream main branch on a regular basis (revisions delay is
usually not more than 1-2 weeks). DPC++ compiler takes everything from LLVM
upstream as is, however some modules of LLVM might be not included in the
default project build configuration. Additional modules can be enabled by
modifying build framework settings.

The DPC++ goal is to support the latest SYCL\* standard and work on that is in
progress. DPC++ also implements a number of extensions to the SYCL\* standard,
which can be found in the [sycl/doc/extensions](./sycl/doc/extensions)
directory.

The main purpose of this project is open source collaboration on the DPC++
compiler implementation in LLVM across a variety of architectures, prototyping
compiler and runtime library solutions, designing future extensions, and
conducting experiments. As the implementation becomes more mature, we try to
upstream as much DPC++ support to LLVM main branch as possible. See
[SYCL upstreaming working group notes](https://github.com/intel/llvm/wiki/SYCL-upstreaming-working-group-meeting-notes)
for more details.

Note that this project can be used as a technical foundation for some
proprietary compiler products, which may leverage implementations from this open
source project. One of the examples is
[Intel(R) oneAPI DPC++ Compiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html)
Features parity between this project and downstream projects is not guaranteed.

Project documentation is available at:
[DPC++ Documentation](https://intel.github.io/llvm/).

### Releases

For release plans, cadence, and version-specific notes, see the
[Releases documentation](./sycl/doc/Releases.md).

### How to use DPC++

#### Dockerfiles

See available Dockerfiles to create containers with pre-built/pre-installed DPC++ compiler at:
[Containers](./sycl/doc/developer/DockerBKMs.md#sycl-dockerfiles-overview)

#### Build from sources

See [Get Started Guide](./sycl/doc/GetStartedGuide.md).

### MemorySanitizer over the liboffload backend

Device MemorySanitizer (MSan) works for SYCL programs that run through the UR
offload adapter (`ONEAPI_DEVICE_SELECTOR=offload:gpu`) on Level Zero GPUs. It
is separate from the UR MSan layer, which intercepts UR calls: here MSan works
below liboffload instead. A shim library with SONAME `libze_loader.so` is
`dlopen`ed before liboffload's level_zero plugin loads Level Zero, so the
plugin's `dlopen("libze_loader.so")` binds to the shim. The shim sends the
calls MSan needs to hooks and forwards every other call to the real loader:
- `zeMemAlloc*` (shadow setup and origins)
- `zeCommandListAppendMemoryCopy/Fill` (shadow propagation)
- `zeModuleCreate`, `zeKernelCreate`, `zeKernelSetGroupSize`
- `zeCommandListAppendLaunchKernel` (launch info and reports)

There are no changes under `unified-runtime/`.

#### Usage

Build liboffload with `-DLIBOMPTARGET_LEVEL_ZERO_MSAN=ON`. This needs Linux,
`level_zero` in `LIBOMPTARGET_DLOPEN_PLUGINS` and the Level Zero headers
(`LIBOMPTARGET_DEP_LEVEL_ZERO_INCLUDE_DIR`). `liboffload_msan_ze.so` is
installed next to `libLLVMOffload.so`. The offload adapter finds liboffload
through its RPATH (`UR_OFFLOAD_INSTALL_DIR`), and `LD_LIBRARY_PATH` does not
override it.

Compile as for any device MSan build. `-g` adds source lines to the reports.
`-Xarch_device -fsanitize-memory-track-origins=1` adds origins.

```bash
clang++ -fsycl -Xarch_device -fsanitize=memory -g app.cpp -o app
LIBOFFLOAD_MSAN_ENABLE=1 ONEAPI_DEVICE_SELECTOR=offload:gpu ./app
```

It prints `==== DeviceSanitizer: MSAN` followed by any
`use-of-uninitialized-value` reports, and aborts on the first report.
Options come from `OFFLOAD_MSAN_OPTIONS`, else `UR_LAYER_MSAN_OPTIONS`.

When `LIBOFFLOAD_MSAN_ENABLE` is set, SYCL (`sycl/source/detail/ur.cpp`) does
not enable `UR_LAYER_MSAN` for MSan images. Instead it sets:
- `LIBOMPTARGET_LEVEL_ZERO_MSAN=1`, which makes the plugin load the shim
- `LIBOMPTARGET_LEVEL_ZERO_MEMORY_POOL=0`: pooled memory has no shadow mapped
  and causes GPU page faults
- `NEOReadDebugKeys=1`, `AllocateHostAllocationsInHeapExtendedHost=0` and
  `UseHighAlignmentForHeapExtended=0`

Without `LIBOFFLOAD_MSAN_ENABLE`, behavior is unchanged from upstream.

**Prerequisite (separate fix):** the offload adapter's `urProgramBuildExp`
fails with "Liboffload doesn't support link options" for any build option, so
every SYCL program compiled with `-g` fails on `offload:gpu`, with or without
MSan. That has to be fixed in the adapter separately.

#### Changes

- **liboffload, backend-agnostic** (`offload/plugins-nextgen/sanitizers/msan/`):
  the MSan host runtime ported from UR. This covers the device-library ABI
  (`msan_libdevice.hpp`, `sanitizer_libdevice.hpp`), shadow layout, origins,
  the stack depot, stack traces and symbolization, options and report
  printing. It builds as the OBJECT library `offload_sanitizer_msan`.
- **liboffload, Level Zero** (`offload/plugins-nextgen/level_zero/msan/`):
  - the hooks (`msan_ze_intercept.cpp`)
  - the shim loader (`msan_ze.cpp`)
  - the hook table ABI (`msan_ze_hooks.hpp`)
  - the shim itself (`ze_shim/msan_ze_shim.cpp`)

  `level_zero/CMakeLists.txt` adds the `LIBOMPTARGET_LEVEL_ZERO_MSAN` option.
  `dynamic_l0/L0DynWrapper.cpp` calls `loadZeShim()` before loading Level
  Zero.
- **SYCL**:
  - `sycl/source/detail/ur.cpp`: the `LIBOFFLOAD_MSAN_ENABLE` switch.
  - `sycl/test-e2e/MemorySanitizer/lit.local.cfg`: liboffload reports no
    architecture, so offload on PVC is treated as `level_zero` +
    `arch-intel_gpu_pvc`.
  - `sycl/test-e2e/format.py`: keeps a `ZE_AFFINITY_MASK` the config already
    set.
- **Tests** (`sycl/test-e2e/MemorySanitizer/offload/`), each built at `-O0`
  and `-O2`:
  - `check_memcpy_partial.cpp`: shadow propagation through partial copies.
  - `check_fill_partial.cpp`: a partial `fill` leaves the tail uninitialized.
  - `check_nd_range_args.cpp`: local and global ids in reports for an
    `nd_range` kernel.
  - `check_alloc_reuse.cpp`: no false reports when allocations are reused; a
    new allocation is reported.

#### Testing

```bash
LIBOFFLOAD_MSAN_ENABLE=1 llvm-lit --param sycl_devices="offload:gpu" \
  --param extra_system_environment=LIBOFFLOAD_MSAN_ENABLE \
  sycl/test-e2e/MemorySanitizer
```

On PVC, with the adapter `-g` fix applied, 20 tests pass, 4 fail and 3 are
unsupported. The 4 failures are offload adapter limitations that hit before
MSan runs:
- `check_usm2d`: 2D USM
- `gpu_aot`: AOT images
- `check_device_global`: device globals
- `local/local_accessor`: local accessors

#### Limitations

- Linux and Level Zero GPUs only.
- With `LIBOFFLOAD_MSAN_ENABLE` set, the UR layer is off for the whole
  process, so devices on other backends in the same process are not checked.
- If `LIBOFFLOAD_MSAN_ENABLE` is set but liboffload was built without the shim,
  nothing is checked and there is no warning.
- `zeCommandListAppendLaunchKernelWithArguments` is not supported by the shim.

### Report a problem

Submit an [issue](https://github.com/intel/llvm/issues) or initiate a 
[discussion](https://github.com/intel/llvm/discussions).

### How to contribute to DPC++

This project welcomes contributions from the community. Please refer to [CONTRIBUTING](/CONTRIBUTING.md) 
for general guidelines around contributing to this project. You can then see 
[ContributeToDPCPP](./sycl/doc/developer/ContributeToDPCPP.md) for DPC++ specific 
guidelines.

# License

See [LICENSE](./sycl/LICENSE.TXT) for details.

<sub>\*Other names and brands may be claimed as the property of others.</sub>
