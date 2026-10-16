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
offload adapter (`ONEAPI_DEVICE_SELECTOR=offload:gpu`) on Level Zero GPUs. The
UR MSan layer can't instrument offload contexts directly, so MSan works below
liboffload instead: a shim library with SONAME `libze_loader.so` is `dlopen`ed
before liboffload's level_zero plugin loads Level Zero. The plugin's
`dlopen("libze_loader.so")` then binds to the shim. The shim sends the calls
MSan needs to hooks and forwards every other call to the real loader:
- `zeMemAlloc*` (shadow setup and origins)
- `zeCommandListAppendMemoryCopy/Fill` (shadow propagation)
- `zeModuleCreate`, `zeKernelCreate`, `zeKernelSetGroupSize`
- `zeCommandListAppendLaunchKernel` (launch info and reports)

There are two copies of this interception. The `LIBOFFLOAD_MSAN_ENABLE`
environment variable selects which one is used at run time:

| | UR shim (default) | liboffload shim (`LIBOFFLOAD_MSAN_ENABLE` set) |
|---|---|---|
| Loaded by | UR MSan layer (`ur_sanddi.cpp`) | level_zero plugin (`L0DynWrapper.cpp`) |
| Shim library | `libur_msan_ze.so` | `liboffload_msan_ze.so` |
| Build switch | always on with `UR_BUILD_ADAPTER_OFFLOAD` | `-DLIBOMPTARGET_LEVEL_ZERO_MSAN=ON` |
| Options | `UR_LAYER_MSAN_OPTIONS` | `OFFLOAD_MSAN_OPTIONS`, else `UR_LAYER_MSAN_OPTIONS` |

#### Usage

Compile as for any device MSan build. `-g` adds source lines to the reports.
`-Xarch_device -fsanitize-memory-track-origins=1` adds origins.

```bash
clang++ -fsycl -Xarch_device -fsanitize=memory -g app.cpp -o app

# UR shim: nothing else needed.
ONEAPI_DEVICE_SELECTOR=offload:gpu ./app

# liboffload shim: needs a liboffload built with LIBOMPTARGET_LEVEL_ZERO_MSAN=ON.
LIBOFFLOAD_MSAN_ENABLE=1 ONEAPI_DEVICE_SELECTOR=offload:gpu ./app
```

Both modes print `==== DeviceSanitizer: MSAN` followed by any
`use-of-uninitialized-value` reports, and abort on the first report.

The liboffload shim needs:
- Linux
- `level_zero` in `LIBOMPTARGET_DLOPEN_PLUGINS`
- Level Zero headers (`LIBOMPTARGET_DEP_LEVEL_ZERO_INCLUDE_DIR`)

`liboffload_msan_ze.so` is installed next to `libLLVMOffload.so`. The offload
adapter finds liboffload through its RPATH (`UR_OFFLOAD_INSTALL_DIR`), and
`LD_LIBRARY_PATH` does not override it.

For MSan images, SYCL (`sycl/source/detail/ur.cpp`) sets
`LIBOMPTARGET_LEVEL_ZERO_MEMORY_POOL=0` in both modes: pooled memory has no
shadow mapped and causes GPU page faults. When `LIBOFFLOAD_MSAN_ENABLE` is set,
SYCL also does not enable `UR_LAYER_MSAN`. Instead it sets the environment the
layer would otherwise set:
- `LIBOMPTARGET_LEVEL_ZERO_MSAN=1`, which makes the plugin load the shim
- `NEOReadDebugKeys=1`, `AllocateHostAllocationsInHeapExtendedHost=0` and
  `UseHighAlignmentForHeapExtended=0`

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
- **Unified Runtime** (`unified-runtime/source/`):
  - `loader/layers/sanitizer/msan/msan_ze*`, `msan/ze_shim/` and the
    `CMakeLists.txt` block: the UR copy of the shim and hooks.
  - `msan/msan_ddi.cpp`, `msan/msan_interceptor.*`: offload contexts are left
    unregistered, so every UR-level MSan hook that would look up their state
    passes them through.
  - `msan/msan_report.*`: the report takes the kernel name as a string, so the
    Level Zero hooks can use it.
  - `ur_sanddi.cpp`: loads the shim when the MSan layer starts.
  - `adapters/offload/program.cpp`: `urProgramBuildExp` ignores build options
    instead of failing, because liboffload builds programs on creation.
    Without this, any SYCL program compiled with `-g` fails on the offload
    adapter, with or without MSan.
- **SYCL**:
  - `sycl/source/detail/ur.cpp`: sets `LIBOMPTARGET_LEVEL_ZERO_MEMORY_POOL=0`
    for MSan in both modes, and adds the `LIBOFFLOAD_MSAN_ENABLE` switch.
  - `sycl/cmake/modules/BuildUnifiedRuntime.cmake`: installs `ur_msan_ze`.
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
llvm-lit --param sycl_devices="offload:gpu" sycl/test-e2e/MemorySanitizer
# liboffload shim:
LIBOFFLOAD_MSAN_ENABLE=1 llvm-lit --param sycl_devices="offload:gpu" \
  --param extra_system_environment=LIBOFFLOAD_MSAN_ENABLE \
  sycl/test-e2e/MemorySanitizer
```

On PVC both modes pass 20 tests, fail 4 and mark 3 unsupported. The 4
failures are offload adapter limitations that hit before MSan runs:
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
