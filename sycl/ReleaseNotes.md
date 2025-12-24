# Release notes for 6.3.0 release

This document covers commit range
[`v6.2.0`](https://github.com/intel/llvm/tree/v6.2.0)...
[`v6.3.0`](https://github.com/intel/llvm/tree/v6.3.0)

## New Features

### KHR extensions

- Prototyped
  [`sycl_khr_work_item_queries`](https://github.com/KhronosGroup/SYCL-Docs/pull/682)
  extension. intel/llvm#18519
- Prototyped
  [`sycl_khr_group_interface`](https://github.com/KhronosGroup/SYCL-Docs/pull/638)
  extension. intel/llvm#17595
- Prototyped
  [`sycl_khr_queue_empty_query`](https://github.com/KhronosGroup/SYCL-Docs/pull/700)
  extension. intel/llvm#18303 intel/llvm#18799 intel/llvm#19151 intel/llvm#18601
- Prototyped
  [`sycl_khr_{static, dynamic}_addrspace_cast`](https://github.com/KhronosGroup/SYCL-Docs/pull/650)
  extensions. intel/llvm#18521

When a said extension is _prototyped_, it means that the extension specification
had not been finalized when the implementation was added. As such, its
implementation is not available by default and requires a special macro to be
set. That is because the implementation is only intended for gathering early
feedback and it is not yet suitable for a production use. See 
[this document](https://github.com/intel/llvm/blob/4a905ca36143ffe0f4ef3a6b30cb366a76528ee1/sycl/doc/developer/KHRExtensions.md)
for more details.

### Other extensions

- Introduced and provided initial implementation to
  [`sycl_ext_oneapi_memory_export`](https://github.com/intel/llvm/blob/v6.3.0-rc1/sycl/doc/extensions/experimental/sycl_ext_oneapi_memory_export.asciidoc)
  extension which allows to share memory allocated by SYCL RT with 3rd-party
  APIs (like Vulkan) without having to re-allocate it and copy data around.
  Please refer to the extension specification to learn about known issues and
  limitations. intel/llvm#19018
- Introduced and provided initial implementation to
  [`sycl_ext_oneapi_async_memory_alloc`](https://github.com/intel/llvm/blob/4a905ca36143ffe0f4ef3a6b30cb366a76528ee1/sycl/doc/extensions/proposed/sycl_ext_oneapi_async_memory_alloc.asciidoc)
  extension.
  Initial implementation only supports CUDA and device-allocated memory pools.
  intel/llvm#14800 intel/llvm#16900 intel/llvm#17863 intel/llvm#17955
  intel/llvm#17689 intel/llvm#19402
- Expanded
  [`sycl_ext_oneapi_kernel_compiler`](https://github.com/intel/llvm/blob/4a905ca36143ffe0f4ef3a6b30cb366a76528ee1/sycl/doc/extensions/experimental/sycl_ext_oneapi_kernel_compiler.asciidoc)
  extension with support for compiling source-based kernel bundles so that they
  can be linked with other kernel bundles. intel/llvm#17442
- 8e347de21ab1 [SYCL] Add support for getting device LUID on windows (#19349)

### Compiler

- Introduced CUDA compatibility mode that can be enabled by `-fsycl-cuda-compat`
  command line option which allows SYCL code to interact with CUDA code more
  closely, such as calling CUDA device functions from SYCL device functions. The
  new mode is intended to help with transition from CUDA to SYCL.
  intel/llvm#12757

## Improvements and bugfixes

### SYCLcompat library

Please note that even though some changes were made to the library, it was made
deprecated in this release and will be removed in the future. See also
Deprecations section below.

- Prevented unnecessary instantiation of internal utility kernels under the hood
  of `fill` and `memcpy` APIs if they are unused. intel/llvm#17743
- Fixed a bug preventing the Hello, World example from the documentation being
  built. intel/llvm#18401

### Sanitizers

- Extended address sanitizer and memory sanitizer to work with features that
  require L0 cooperative dispatch (such as root-group). intel/llvm#18198
- Fixed an undefined behavior in address and memory sanitizers implementations.
  intel/llvm#19063
- Fixed a bug in address and memory sanitizers which caused sporadic false
  positive reports. intel/llvm#18253 intel/llvm#18525
- Fixed a bug where sanitizers could segfault after reporting an error they had
  found. intel/llvm#19009
- Fixed a bug preventing sanitizers to be used together with shared libraries.
  intel/llvm#18819
- Fixed some memory leaks in sanitizers implementation. intel/llvm#19066
- Fixed possible hangs when an application is stopped by a sanitizer because
  errors were discovered. intel/llvm#19085

- 97f508bd257d [DevSAN] Set maximum supported local/private shadow memory size (#19465)
  -  What if the limit we set is too small? What is a user-visible side effect of this?

- 69c3b9f9e01b [SYCL] Cherry-pick sanitizer patches (#19867)
  -  Looks like 2/3 were fixes for regressions, as such the plan is to omit this
- fae8cc2630b5 [DeviceSanitizer] Print log when build/link fail (#17521)
- 55416cae415f [DeviceSanitizers] Set names to nameless globals (#17599)

#### Address sanitizer

- Improved error messages which are logged if enqueue kernel failed with
  address sanitizer enabled. intel/llvm#20237
- Fixed a memory leak and potential undefined behavior that may happen in
  address sanitizer when application exits. intel/llvm#18612

#### Thread sanitizer

- Fixed a bug in thread sanitizer which may result in false positives reported
  on subsequent kernel launches. intel/llvm#19355
- Improved thread sanitizer performance by avoiding unnecessary device -> host
  synchronization if there are no reports to transfer. intel/llvm#19133
- Added support for detecting data races in local memory. intel/llvm#18718
- Refined thread mapping algorithm for better performance of the thread
  sanitizer. intel/llvm#19010
- Added support for detecting data races in SYCL buffers. intel/llvm#17625
- Fixed false positive reports coming from joint matrix usage. intel/llvm#18773
- Added support for thread sanitizer in AOT mode. intel/llvm#18130
- Fixed false positive reports coming from barriers usage. intel/llvm#17900
- Fixed memory leaks in thread sanitizer. intel/llvm#18880
- Improved thread sanitizer performance. intel/llvm#19127

- 1d7e9ef8e71a [DevTSAN] Fix missing intercepted API and release local shadow (#18942)
  -  Looks like some bugfix
- e2968564552a [DevTSAN] Don't insert cleanup instruction for dynamic allocas (#18090)
  -  Unclear what this is
- 597709bd52c0 [DevTSAN] Treat each work item as a thread for both CPU & GPU device (#18580)
- 1a83e6ec3272 [DevTSAN] Treat each work item as a thread for GPU device (#18347)
  -  ???

#### Memory sanitizer

- Enabled origin tracking which provides information about where a memory
  allocation associated with an issue report comes from. intel/llvm#18693
- Fixed a bug with `async_work_group_copy` causing false negative report.
  intel/llvm#18216

- b07a0cf68226 [DevMSAN] Always make sure poison shuffle mask point to clean shadow (#18191)
  -  Some bugfix?
- 923eca9baf7c [DevMSAN] Don't report error when write `__MsanLaunchInfo` failed (#18129)
  -  Some bugfix?
- 23969c664d94 [DevMSAN] Fix wrong arguments when calling Memset (#19188)
  -  Not sure what does it fix

### Native CPU

- Fixed compilation issues with `--offload-new-driver`. intel/llvm#19422
- Improved performance of generating kernels by generating correct aliasing
  metadata for implicit kernel arguments emitted by the compiler.
  intel/llvm#19288
- Fixed some failures with free function kernels on `native_cpu` backend.
  intel/llvm#19052
- Reduced usage of thread-local variables in generated kernels which should
  improve their performance. intel/llvm#17822
- Added initial implementation of queue info queries for the `native_cpu`
  backend. intel/llvm#18646
- Fixed some of spec constants tests that were failing on the `native_cpu`
  backend. intel/llvm#17558
- Fixed a bug where using `fill` operation with pattern size equal to the size
  of a memory region that needs to be filled caused runtime error about invalid
  size on the `native_cpu` backend. intel/llvm#18366
- Fixed incorrect values returned from device info queries for atomic fence
  and atomic memory scope capabilities. intel/llvm#18537
- Fixed a bug causing incorrect results from math built-ins operating on
  3-element vectors on AArch64 on the `native_cpu` backend. intel/llvm#19087

- 0386b1d169da [SYCL][NativeCPU] Add libclc at compile time. (#19346)
  -  ???
- 8ceaf1d03efc [SYCL][NativeCPU] Update OCK. (#17862)
- f12844427bad [SYCL][NativeCPU] Update OCK. (#18533)
- e22b8e49bb2f [SYCL][NativeCPU] Update OCK. (#18803)
- 52c775b35b46 [SYCL][NativeCPU] Update OCK. (#18930)
- c666567b7b90 [SYCL][NativeCPU] Update OCK. (#19241)
  -  Are there any benefits? should it be mentioned?
- 65d926c3f9a3 [SYCL][NativeCPU] Set target attrs for subkernels. (#19169)
  -  Some kind of bugfix?
- 72829cdeb123 [SYCL][NativeCPU] Fix alignment of global and local memory. (#19076)
  -  Also some kind of bugfix?
- a877430a6b39 [NATIVECPU][UR] added mutex to backend queue (#18923)

- d653b20d863d [SYCL][NativeCPU] Copy over more host/aux target data. (#17999)
  -  Some bugfix?
- 85030796ab40 [SYCL] Revert prepare-builtins change for non-NativeCPU. (#18490)
  -  this is a follow-up fix for #17999

- 07d8bab1b3f0 [SYCL][NativeCPU] Unify definitions of `__spirv_*` functions. (#18010)
  -  Bugfix? or just a cleanup?

- 9433a80ef9f9 [SYCL][NativeCPU] Process nativecpu_utils with prepare_builtins (#17850)
  -  Fixed some compilation issues?

- e811c53a2ed3 [SYCL][NativeCPU] Build libclc target-independently. (#17408)
  -  Looks like build system improvement

Plan to ignore:
-   5a99c82dd888 [SYCL][NativeCPU][NFC] Prevent generating unnamed functions (#17707)
-   735410dd5ed8 [SYCL][NATIVECPU] remove Comdat from kernel whose name was taken/renamed (#18726)
    -  Bugfix for #17707
-   dc2fc8d2ff66 [SYCL][NativeCPU] Support native_cpu in llvm::Triple. (#18783)
-   b902d64bba9e [SYCL][NativeCPU] Create NativeCPUABIInfo (#19344)
-   ebe975834995 [SYCL][NativeCPU] Move __spirv_AtomicF{Add,Min,Max}EXT into libclc. (#19365)

### ESIMD

- Made experimental `get_hw_thread_id` and `get_subdevice_id` functions always
  return 0, because underlying functionality had been deprecated and removed
  from GPU drivers. intel/llvm#19391
- Fixed a couple of memory leaks in in one of ESIMD-specific compiler passes.
  intel/llvm#17706 intel/llvm#17632

### Bindless images

- Clarified backend support status in the
  [`sycl_ext_oneapi_bindless_images`](https://github.com/intel/llvm/blob/4a905ca36143ffe0f4ef3a6b30cb366a76528ee1/sycl/doc/extensions/experimental/sycl_ext_oneapi_bindless_images.asciidoc)
  extension specification. intel/llvm#18555
- Added support for `max_image_linear_width` and `max_image_linear_height`
  device info queries on Level Zero backend. intel/llvm#19529
- Expanded the
  [`sycl_ext_oneapi_bindless_images`](https://github.com/intel/llvm/blob/4a905ca36143ffe0f4ef3a6b30cb366a76528ee1/sycl/doc/extensions/experimental/sycl_ext_oneapi_bindless_images.asciidoc)
  extension specification and implementation to support DX11 memory
  interoperability. intel/llvm#19217
- Expanded the
  [`sycl_ext_oneapi_bindless_images`](https://github.com/intel/llvm/blob/4a905ca36143ffe0f4ef3a6b30cb366a76528ee1/sycl/doc/extensions/experimental/sycl_ext_oneapi_bindless_images.asciidoc)
  extension specification and implementation to support interoperability using
  `dma_buf` external memory handles. intel/llvm#18988
- Fixed a bug where operations with external semaphore would ignore any SYCL
  command group dependencies and won't participate in the dependency graph on
  their own, causing race conditions. intel/llvm#20196

- b0351a0f2c6c [UR][CUDA][HIP] Fix error handling in bindless images code (#18246)
  - Any user-visible impact?

### Performance improvements

A series of patches was submitted with aim to simplify our header files with
intent to improve compilation times of `<sycl/sycl.hpp>` header. Even though
results may not be visible in average application, certain specific scenarios
have been improved.

- Removed declarations of some internal built-ins from headers. intel/llvm#17672
  intel/llvm#17732 intel/llvm#17471
- Reduced amount of template instantiations needed to submit a kernel.
  intel/llvm#18065 intel/llvm#17640 intel/llvm#17670 intel/llvm#18019
  intel/llvm#18534
- Made other smaller code cleanups/simplifications. intel/llvm#19183
  intel/llvm#18114

Even bigger series of patches was submitted with aim to improve performance of
SYCL RT and reduce overheads over underlying layers such as Unified Runtime.

- Refactored code to reduce amount of unnecessary copies made in SYCL RT, such
  as smart pointer copies made when SYCL objects are passed between different
  functions within the SYCL RT. intel/llvm#17859 intel/llvm#17963
  intel/llvm#17835 intel/llvm#17880 intel/llvm#17705 intel/llvm#17941
  intel/llvm#17340 intel/llvm#19478 intel/llvm#18099 intel/llvm#18163
  intel/llvm#19462 intel/llvm#19251 intel/llvm#19156 intel/llvm#19102
  intel/llvm#19153 intel/llvm#19147 intel/llvm#19148 intel/llvm#19120
  intel/llvm#19126 intel/llvm#19123 intel/llvm#19006 intel/llvm#19004
  intel/llvm#18983 intel/llvm#19007 intel/llvm#18979 intel/llvm#18981
  intel/llvm#18928 intel/llvm#18980 intel/llvm#18966 intel/llvm#18877
  intel/llvm#18714 intel/llvm#18712 intel/llvm#18306 intel/llvm#18709
  intel/llvm#18178 intel/llvm#19484 intel/llvm#19459 intel/llvm#19456
  intel/llvm#17584 intel/llvm#17569 intel/llvm#17491 intel/llvm#17750
  intel/llvm#18143 intel/llvm#18720 intel/llvm#19202 intel/llvm#18320
  intel/llvm#18125 intel/llvm#18899 intel/llvm#17921 intel/llvm#17802
  intel/llvm#17952 intel/llvm#19383 intel/llvm#19315 intel/llvm#19376
  intel/llvm#19314 intel/llvm#19313 intel/llvm#19312 intel/llvm#19299
  intel/llvm#19186 intel/llvm#19187 intel/llvm#19030 intel/llvm#18291
  intel/llvm#18477 intel/llvm#18834 intel/llvm#18715 intel/llvm#19334
  intel/llvm#19366 intel/llvm#19371 intel/llvm#19372 intel/llvm#19350
  intel/llvm#19438 intel/llvm#19487 intel/llvm#19351 intel/llvm#18748
- Fixed places where short-lived (i.e. temporary) objects were unnecessary
  allocated. intel/llvm#17895 intel/llvm#17912 intel/llvm#18539
  intel/llvm#17868
- Refactored code to remove unnecessary heap allocations replacing them with
  stack-allocated objects. intel/llvm#18314 intel/llvm#17319
- Refactored how locks are handled to reduce amount of unnecessary locking.
  intel/llvm#17575 intel/llvm#17883 intel/llvm#17613 intel/llvm#18678
  intel/llvm#18041 intel/llvm#18427
- Introduced caching mechanism for various device information descriptors to
  avoid repeated calls to low-level runtimes. intel/llvm#18546 intel/llvm#18467
  intel/llvm#18597 intel/llvm#18603
- Optimized `.wait()` operation for in-order queues. intel/lvm#18656
- Made various other optimizations for kernels submission. intel/llvm#17849
  intel/llvm#17791 intel/llvm#17669 intel/llvm#19264 intel/llvm#18081
  intel/llvm#19555 intel/llvm#18386 intel/llvm#18157 intel/llvm#18318
  intel/llvm#18826 intel/llvm#18538 intel/llvm#18654 intel/llvm#18804
  intel/llvm#18387 intel/llvm#18661 intel/llvm#18833 intel/llvm#18851
  intel/llvm#18582 intel/llvm#18565
- Optimized how SYCL RT works with its own internal global objects.
  intel/llvm#17740
- Reduced XPTI overhead when it is disabled. intel/llvm#18334 intel/llvm#18005
- Optimized XPTI when it is enabled. intel/llvm#18452 intel/llvm#19160
- Optimized check for nested kernel submissions, i.e. `queue::submit()` called
  from within another `queue.submit()`. intel/llvm#18787
- Ensured that a program is not re-built for sub-sub-devices making kernel
  submission to them quicker. intel/llvm#20126
- Optimized events management within SYCL RT avoiding the overhead of keeping
  and tracking events entirely in some cases. intel/llvm#18277 intel/llvm#20235
  intel/llvm#20353

### Free function kernels

- Added a diagnostic to enforce that the first declaration of a free function
  kernel should be marked with the corresponding properties. intel/llvm#18405
- Added support for defining a free function kernel within a namespace.
  intel/llvm#17585
- Fixed a bug preventing usage of work group scratch memory within free function
  kernels
- Fixed a bug preventing usage of free function kernels with 3rd-party host
  compiler. intel/llvm#19901
- Added support for kernel info queris for free function kernels.
  intel/llvm#19868 intel/llvm#18866
- Added support for free function kernels being defined in a separate
  translation unit. intel/llvm#18955
- Added a diagnostic to highlight the limitation that free function kernels
  cannot be defined as static class members. intel/llvm#18761
- Added support for so-called special types (such as `accessor`) to be
  arguments of free function kernels. intel/llvm#17789
- Added diagnostics to enforce free function kernels restrictions, such as:
  `void` return type, no arguments with default values. intel/llvm#18329
- Added support for free function kernels to be defined as function templates.
  intel/llvm#17936 intel/llvm#19916 intel/llvm#18929 intel/llvm#20236
- Added support for free function kernels on CUDA & HIP backends.
  intel/llvm#17899

### Graph

- Updated the
  [`sycl_ext_oneapi_graph`](https://github.com/intel/llvm/blob/7ce4d2f374ad2c2ebc8569dc46fe74c354238796/sycl/doc/extensions/experimental/sycl_ext_oneapi_graph.asciidoc)
  extension specification:
  - cleaned up outdated known issues. intel/llvm#18482
  - updated the formatting to match recent formatting changes in the core
    SYCL 2020 spec. intel/llvm#18471
  - clarified relationship between new aspects introduced by the extension.
    intel/llvm#17996
  - clarified an executable graph state after its destruction. intel/llvm#18619

- Introduced and implemented graph-owned memory allocations, i.e.
  interoperability with the
    [`sycl_ext_oneapi_async_memory_alloc`](https://github.com/intel/llvm/blob/7ce4d2f374ad2c2ebc8569dc46fe74c354238796/sycl/doc/extensions/proposed/sycl_ext_oneapi_async_memory_alloc.asciidoc)
    extension. intel/llvm#18001 intel/llvm#18002
- Made several changes to optimize the extension implementation:
  - reduced number of temporary objects which are re-created. intel/llvm#18223
  - made `enqueue` more efficient for in-order queues. intel/llvm#18792
- Implemented a mechanism which allows for allocations to be re-used within a
  graph. intel/llvm#18340
- Improved error message when trying to update a node that doesn't support
  update. intel/llvm#17637
- Optimized `sycl::ext::oneapi::experimental::execute_graph` to avoid creating
  events internally. intel/llvm#18344
- Added support for using CUDA-Graph async alloc/free nodes in a
  [`sycl_ext_codeplay_enqueue_native_command`](https://github.com/intel/llvm/blob/7ce4d2f374ad2c2ebc8569dc46fe74c354238796/sycl/doc/extensions/experimental/sycl_ext_codeplay_enqueue_native_command.asciidoc)
  native-command object in a graph. intel/llvm#19091

- 7e7dffcaab28 [SYCL][UR][Graph] Require OpenCL simultaneous use (#17658)
  -   Looks like internal changes to make graph spec work on OpenCL backend
- 99cd1adb038a [UR][OpenCL][Graph] Set simultaneous-use property (#18154)
  -   Follow-up to the one above

- 4ed8534f20ea [SYCL][Graph][OpenCL] Map copy/fill to SVM (#18177)
  -   Looks like internal changes to make graph spec work on OpenCL backend

- 6635e44d8eb9 [UR][Graph] Serialize submissions of the same command-buffer (#18295)
- 320516be6443 [UR][Graph] Strengthen in-order command-buffer property (#18444)

- 2abe0d616777 [SYCL][Graph] Implement dynamic_work_group_memory for SYCL-Graphs  (#17314)
- c20712f4b8dd [SYCL][Graph] Implement dynamic local accessors (#18437)
  -   Corresponding specification was newer merged, plan to omit this

- 877beeec6f34 [SYCL][Graph] Remove explicit L0 wait from SYCL-RT (#18064)
  -   Looks like internal refactoring, plan to omit this

- bc10261335d0 [L0] Revert copy engine refactor and disable copy-engine for SYCL-Graph on DG2 (#18720)
  - The reverted commit was not mentioned in release notes for the previous
  - release, however this commit does fix some bug.

### Documentation

- Added
  [documentation](https://github.com/intel/llvm/blob/4a905ca36143ffe0f4ef3a6b30cb366a76528ee1/sycl/doc/Releases.md)
  about releases made from intel/llvm. intel/llvm#17879
- Updated the
  [ABI Policy Guide](https://github.com/intel/llvm/blob/4a905ca36143ffe0f4ef3a6b30cb366a76528ee1/sycl/doc/developer/ABIPolicyGuide.md)
  with the description of `__INTEL_PREVIEW_BREAKING_CHANGES` macro.
  intel/llvm#18422
- Updated pre-requisites in the
  [Get Started Guide](https://github.com/intel/llvm/blob/4a905ca36143ffe0f4ef3a6b30cb366a76528ee1/sycl/doc/GetStartedGuide.md)
  to address intel/llvm#17478. intel/llvm#17565
- Refreshed CUDA & HIP sections of the
  [Get Started Guide](https://github.com/intel/llvm/blob/4a905ca36143ffe0f4ef3a6b30cb366a76528ee1/sycl/doc/GetStartedGuide.md)
  intel/llvm#17928
- Fixed rendering of the
  [`sycl_ext_oneapi_bindless_images`](https://github.com/intel/llvm/blob/4a905ca36143ffe0f4ef3a6b30cb366a76528ee1/sycl/doc/extensions/experimental/sycl_ext_oneapi_bindless_images.asciidoc)
  extension specification. intel/llvm#19545
- Fixed description of the default value of the
  [`SYCL_ENABLE_DEFAULT_CONTEXTS`](https://github.com/intel/llvm/blob/4a905ca36143ffe0f4ef3a6b30cb366a76528ee1/sycl/doc/EnvironmentVariables.md)
  environment variable to match the implementation. intel/llvm#19277
- Improved documentation about `-offload-compress` command line option.
  intel/llvm#17990
- Reflected removal of the experimenal kernel fusion feature in our
  documentation. intel/llvm#17929

- 637476efb32e [SYCL][Doc] Add SPV_INTEL_function_variants extension (#18365)
- b12d2803a61e [SYCL][Docs] Move sycl_ext_intel_event_mode to experimental (#18417)
- c8986cd400b6 [SYCL] Update `khr_free_function_commands` extension interfaces to accept `const queue&`  (#18564)

### Compiler

- Added support for `-fsycl-device-obj=asm` compiler flag to allow simple
  inspection of device assembly code that is being generated. intel/llvm#17390
- Addressed issue intel/llvm#15120 by adding a compiler diagnostic indicating
  that `-fsycl-unnamed-lambda` cannot be used together with
  `-fsycl-host-compiler`. intel/llvm#17840
- Fixed a bug where `-nocudalib` compiler flag would have no effect.
  intel/llvm#19216
- Made `-gline-tables-only` apply only for device compilation, thus fixing
  potential errors reported by device compiler during JIT/AOT compilation.
  intel/llvm#18522
- Made SYCL headers to be treated as system headers when `-fsycl-host-compiler`
  is used to silence unrelated warnings coming out of them. intel/llvm#18449
- Added support for passing multiple NVIDIA and AMD targets to `-fsycl-targets`
  command line option. intel/llvm#18145
- Fixed a bug where resulting object files were not removed if compilation
  fails. intel/llvm#18190
- Addressed issue intel/llvm#7738. intel/llvm#18097
- Made compiler aware of optional features supported by `intel_cpu_spr` target,
  preventing possible unexpected failures during AOT compilation for that
  target. intel/llvm#19265
- Refreshed compiled knowledge about optional kernel features supported by HIP
  targets. intel/llvm#18835
- Refreshed compiled knowledge about optional kernel features supported by NVPTX
  targets. intel/llvm#18782
- Fixed a bug where using `group_local_memory_for_overwrite` from a
  `SYCL_EXTERNAL` function would lead to linkage error from device compiler
  saying that the said `SYCL_EXTERNAL` function is undefined. intel/llvm#18660
- Updated behavior of `-fsycl-dump-device-code` option to support printing
  non-SPIRV device images as well. intel/llvm#17546
- Optimized implementation of various ID queries (like work-item global ID) for
  NVPTX targets in `-fsycl-id-queries-fit-in-int` mode. intel/llvm#18999
- Added support for `std::rint` in device code as part of the
  [`C-CXX-StandardLibrary`](https://github.com/intel/llvm/blob/4a905ca36143ffe0f4ef3a6b30cb366a76528ee1/sycl/doc/extensions/supported/C-CXX-StandardLibrary.rst)
  extension. intel/llvm#18857
- Added a diagnostic when `-offload-compress` option is used when the compiler
  is built without device code compression capabilities. intel/llvm#17990
- Addressed issue intel/llvm#17591 where the compiler may crash when
  `annotated_ptr` is constructed using a function argument. intel/llvm#18590
- Added support for `rand` built-in for CUDA & HIP. intel/llvm#19001
- Fixed a bug where using `--dependent-lib=msvcrtd` and
  `-fms-runtime-lib=dll_dbg` would lead to link issues. intel/llvm#19380
- Fixed compilation issues caused by `__glibcxx_assert_fail` function used in
  STL implementation from GCC 15. intel/llvm#18856
- Optimized `memcpy` on CUDA backend. intel/llvm#18598
- Fixed a bug preventing usage of `std::complex` on CUDA & HIP backends.
  intel/llvm#18667
- Introduced a new optimization pass to reduce amount of work-group barriers in
  device code. For example, in a SYCL application multiple work-group algorithms
  (like scans) may be invoked one right after the other and each of them
  requires synchronization. This can lead to situations where there are multiple
  unnecessary barriers emitted into device code one right after another.
  intel/llvm#19353

### Misc

- Made fixes and adjustments to the hardening flags which are applied to the
  project. intel/llvm#19268 intel/llvm#19357 intel/llvm#18398 intel/llvm#17690
  intel/llvm#19235
- Made set of changes intended to improve the way of how we deal with the
  project's dependencies, thus simplifying the build and packaging process of
  the repository:
  - Fixed build with XPTI disabled. intel/llvm#18269
  - Added ability to use system installed vc-intrinsics dependency.
    intel/llvm#18206
  - Skipped installation of UMF if we already using a pre-installed version of
    it. intel/llvm#18968
  - Dropped dependency on boost. intel/llvm#15850
  - Fixed build in environments where GCC is installed in a non-standard
    location. intel/llvm#18493
- Setup `RUNPATH` for the `libsycl.so` so that environment setup for running
  SYCL applications is easier. intel/llvm#15850
- Enabled `-Wl,--gc-section` flag for `libsycl.so` linking which potentially
  reduces the library size. intel/llVM#18293
- Cleaned up device libraries, dropping unnecessary files and portions of them,
  thus reducing the package size. intel/llvm#19190 intel/llvm#19873
  intel/llvm#18483
- Made improvements to the `sycl-ls` tool:
  - it now displays errors if any Unified Runtime adapters are present, but
    failed to load. intel/llvm#17490 intel/llvm#17651 intel/llvm#18025
  - it now displays a warning if a user of the tool is not included into groups
    as `render` to access GPUs on a system. intel/llvm#19520 intel/llvm#19538
  - made UUID to be displayed in a formatted manner. intel/llvm#18561
- Added a diagnostic if `<sycl/sycl.hpp>` is included, but no `-fsycl`
  compilation flag is specified. intel/llvm#19279
- Expanded
  [`sycl_ext_oneapi_device_architecture`](https://github.com/intel/llvm/blob/4a905ca36143ffe0f4ef3a6b30cb366a76528ee1/sycl/doc/extensions/experimental/sycl_ext_oneapi_device_architecture.asciidoc)
  extension and implementation to recognize BMG G31 and WCL hardware.
  intel/llvm#18890
- Made `swizzle::operator vec` available for 1-element swizzles in accordance
  with KhronosGroup/SYCL-Docs#800. intel/llvm#17870
- Dropped support of the secondary queue fallback (which is now allowed since
  KhronosGroup/SYCL-Docs#811). intel/llvm#18201 intel/llvm#18188
- Improved error reporting by including backend name in the exception message.
  intel/llvm#18889
- Significantly ramped up the quality of the new Unified Runtime adapter for the
  Level Zero backend called `v2` (which was announced in previous release). Now
  the new adapter is used by default for Intel Battlemage GPUs and newer.
  intel/llvm#19333
- Fixed pretty printing of `sycl::device` class in debugger. intel/llvm#19059
- Updated the `SYCL_EXT_INTEL_DEVICE_INFO` macro value to correctly represent
  the
  [extension](https://github.com/intel/llvm/blob/4a905ca36143ffe0f4ef3a6b30cb366a76528ee1/sycl/doc/extensions/supported/sycl_ext_intel_device_info.md)
  version that is implemented. intel/llvm#19012
- Updated the `SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND` macro value to correctly
  represent the
  [extension](https://github.com/intel/llvm/blob/4a905ca36143ffe0f4ef3a6b30cb366a76528ee1/sycl/doc/extensions/experimental/sycl_ext_codeplay_enqueue_native_command.asciidoc)
  version that is implemented. intel/llvm#18321
- Expanded SYCL-RTC to support CUDA & HIP targets. intell/llvm#19342
- Fixed a bug in the
  [`sycl_ext_oneapi_kernel_compiler`](https://github.com/intel/llvm/blob/4a905ca36143ffe0f4ef3a6b30cb366a76528ee1/sycl/doc/extensions/experimental/sycl_ext_oneapi_kernel_compiler.asciidoc)
  extension implementation where `build_options` property would be ignored when
  source language is OpenCL C. intel/llvm#18853
- Updated the
  [`sycl_ext_oneapi_kernel_properties`](https://github.com/intel/llvm/blob/4a905ca36143ffe0f4ef3a6b30cb366a76528ee1/sycl/doc/extensions/experimental/sycl_ext_oneapi_kernel_properties.asciidoc)
  extension specification and implementation to clarify that the method
  `get(properties_tag)` to attach properties to functors must be defined as
  `const`. intel/llvm#17947
- Fixed a bug with event profiling info not being available if a command is
  is submitted through a queue shortcut like `queue::memcpy`. intel/llvm#18455
- Fixed a potential infinite loop when JIT compilation fails because of lack of
  memory. intel/llvm#18888
- Fixed a race condition in XPTI. intel/llvm#20354
- Fixed a bug in error reporting from kernel submissions where an error code
  received from a low-level runtime would not be included into an exception
  thrown by SYCL RT. intel/llvm#18517
- Fixed a bug where querying profiling info of an event produced from a queue
  created _without_ profiling enabled would _not_ result in an exception being
  thrown as required by the SYCL 2020 specification. intel/llvm#18982
- Addressed issue intel/llvm#19777 where submitting a barrier that has a
  non-empty wait list which includes event from a barrier submitted to another
  queue would not result in a correct synchronization. intel/llvm#20125
- Improved accuracy of command submission timestamps for Level Zero backend.
  intel/llvm#18735
- Fixed a bug where setting local via `std::locale::global` would interfere
  with SYCL RT causing various issues (`SYCL_DEVICE_ALLOWLIST` parsing being an
  example of such case). intel/llvm#18550 intel/llvm#18578
- Fixed a bug where a global range that does not fit into `signed int` would
  still be accepted under `-fsycl-id-queries-fit-in-int` mode.
  intel/llvm#18439
- Fixed a bug where creating `usm_allocator` with properties would always result
  in exception being thrown, even if properties are correct and legal.
  intel/llvm#20022
- Dropped warning suppression macro from SYCL headers which were mistakenly
  applied globally and as such affected other headers and code in a user's
  application. intel/llvm#17918
- Made a change to reduce size of generated object files that contain device
  code. Too huge object files lead to compilation errors where clang complains
  that input file is too large. intel/llvm#17727 intel/llvm#17723
- Added mock implementation of `_invoke_watson` to resolve compilation issues
  with `std::array` being used in device code on Windows when application is
  compiled in debug mode with certain MSVC versions. intel/llvm#19977
- Fixed a bug where `kernel_queue_specific::max_num_work_group` info query
  would return incorrect results on multi-GPU systems. intel/llvm#18958
- Fixed an error with SYCL RT loading on Linux when the runtime is packaged
  without using symlinks (i.e. some shared libraries are being duplicated).
  intel/llvm#20254
- Fixed a bug where using shared libraries where device code is one library is
  compressed and in another is not would cause runtime errors about missing
  symbols. intel/llvm#18906
- Fixed a couple of race conditions during decompression of device images.
  intel/llvm#20124 intel/llvm#19942
- Fixed a bug where on systems with multiple Intel GPUs the same GPU may have
  been reported twice. intel/llvm#20432
- Made improvements to the teardown/shutdown processes. In particular, those
  improvements should have resolved a known issue about Unified Runtime's
  Level Zero leak checker not working correctly on Windows with default
  contexts. intel/llvm#17869 intel/llvm#19195 intel/llvm#18251
- Added support for different accumulator and output types in joint matrix APIs.
  intel/llvm#17502
- Fixed a bug where kernel properties defined via `auto get(properties_tag)`
  method of a functor may be ignored if a kernel defined this way is launched
  via simple `parallel_for(range)` mechanism. intel/llvm#18900
- Refactored the way how functions from the `cmath` header are provided for SYCL
  kernels on CUDA & HIP backends. The new mechanism moves device implementation
  of the standard C/C++ library (but only `cmath` header for now) earlier in
  the compilation flow, allowing us to reduce the package size and in the future
  save some time during the link stage. intel/llvm#18706

- 155a0aa3c12f [SYCL][Doc] Add new device descriptors to sycl_ext_intel_device_info extension (#17386)
  -  Doc-only change, do we have an implementation for this?
- 418be5a98bae [SYCL][CUDA] Add implementation of new device descriptors (#17590)
  -  Does it implement support for some Intel extension on CUDA?

- 4a905ca36143 [SYCL] Cherry-pick fixes for UR L0 v1 adapter memory leaks (#20253)
- 85a3861afc8b [SYCL][UR][OpenCL] allow passing events from multiple contexts to urEventWait (#18711)
- c60a1fc51295 [SYCL] Make host task timestamps share the same base as device tasks (#18710)
- 51e8d95d8bcb [SYCL][host_task] Align host_task event submission clock with start/end clock (#18377)
- 8cd13a7a2e45 [SYCL][CL] Fix ownership of native handle in sycl::make_queue (#18309)
  - Follow-up fix for #17572
- 35ed0b04b011 [SYCL] Fix handling of discarded events in preview lib (#19005)
  -  Do we need to mention this at all?
- 2799ba32e240 [SYCL] Always store last event (for ioq) if scheduler was not bypassed (#18867)
- f7583cabf6c4 [SYCL] Do not count devices for banned platforms (#18878)
- 1e9c41dc6b69 [SYCL] Fix potential misaligned allocation size (#18375)
- 1dee646dfc8b [SYCL] Hide inline definitions of stdio functions (#18174)
- 80fd6651b00d [SYCL] Keep multiple copies for bf16 device library image (#17461)
- ad788883ddbc [SYCL] Fix race condition in submission time read/write (#18716)
- 3b42472928ba [SYCL] Make adapter releasing more robust wrt adapter specific errors (#19050)

- d0279af830d8 [SYCL] Support UR_DEVICE_INFO_USM_CONTEXT_MEMCPY_SUPPORT_EXP on Cuda (#19267)
- e2bd09d76ec0 [SYCL][LowerWGScope] Fix getSizeTTy to use pointer size in global address space (#19011)

## API/ABI breakages

### Changes that are effective immediately

- Renamed `-Wno-libspirv-hip-cuda` compiler flag to
  `-Wno-unsafe-libspirv-not-linked`. intel/llvm#19053

### Deprecations

Those APIs are still present and tested, but they will be removed in future
releases:

- Deprecated SYCLcompat library. intel/llvm#19976
- Deprecated various `parallel_for`, `parallel_for_work_group` and
  `single_task` overloads which were accepting both `sycl::kernel` object and
  a lambda. intel/llvm#18044
- Deprecated
  [`sycl_ext_oneapi_discard_queue_events`](https://github.com/intel/llvm/blob/4a905ca36143ffe0f4ef3a6b30cb366a76528ee1/sycl/doc/extensions/deprecated/sycl_ext_oneapi_discard_queue_events.asciidoc)
  extension in favor of
  [`sycl_ext_oneapi_enqueue_functions`](https://github.com/intel/llvm/blob/4a905ca36143ffe0f4ef3a6b30cb366a76528ee1/sycl/doc/extensions/experimental/sycl_ext_oneapi_enqueue_functions.asciidoc).
  Properties defined by the deprecated extension do not have any effect anymore.
  intel/llvm#18059
- Deprecated `dynamic_parameter` constructors from the
  [`sycl_ext_oneapi_graph`](https://github.com/intel/llvm/blob/4a905ca36143ffe0f4ef3a6b30cb366a76528ee1/sycl/doc/extensions/experimental/sycl_ext_oneapi_graph.asciidoc)
  extension that take a `graph` object. intel/llvm#18199
- Deprecated `-f[no-]sycl-device-lib=`, `-f[no-]sycl-device-lib-jit-link`
  command line options. intel/llvm#18581
- Deprecated `-fsycl-dump-device-code` command line option in favor of
  `-save-offload-code`. intel/llvm#18908

### Upcoming API/ABI breakages

This changes are available for preview under `-fpreview-breaking-changes` flag.
They will be enabled by default (with no option to switch to the old behavior)
in the next ABI-breaking release:

- Fixed 1-element `vec` ambiguities in accordance with
  KhronosGroup/SYCL-Docs#670. intel/llvm#17722
- Updated `vec` constructors in accordance with KhronosGroup/SYCL-Docs#668.
  intel/llvm#17712
- Removed underspecified `vec::vector_t` in accordance with
  KhronosGroup/SYCL-Docs#676. intel/llvm#17867
- Implemented `vec` explicit conversions in accordance with
  KhronosGroup/SYCL-Docs#669. intel/llvm#17713
- Reimplemented `__swizzled_vec__` following recent SYCL 2020 specification
  changes (see items about `vec` above). intel/llvm#17817
- Changed return types of `logical_or` and `logical_and` in accordance with
  KhronosGroups/SYCL-Docs#648. intel/llvm#17239
- Performed general cleanup of legacy ABI entry points. intel/llvm#19276
  intel/llvm#19377

## Known Issues

- SYCL headers use unreserved identifiers which sometimes cause clashes with
  user-provided macro definitions (intel/llvm#3677). Known identifiers include:
  - `G`. intel/llvm#11335
  - `VL`. intel/llvm#2981
- Intel Graphic Compiler's Vector Compute backend does not support
  O0 code and often gets miscompiled, produces wrong answers
  and crashes. This issue directly affects ESIMD code at O0. As a
  temporary workaround, we optimize ESIMD code even in O0 mode.
  [00749b1e8](https://github.com/intel/llvm/commit/00749b1e8e3085acfdc63108f073a255842533e2)
- When using `sycl_ext_oneapi_matrix` extension it is important for some
  devices to use the sm version (Compute Capability) corresponding to the
  device that will run the program, i.e. use `-fsycl-targets=nvidia_gpu_sm_xx`
  during compilation. This particularly affects matrix operations using
  `half` data type. For more information on this issue consult with
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma-restrictions
- C/C++ math built-ins (like `exp` or `tanh`) can return incorrect results
  on Windows for some edge-case input. The problems have been fixed in the
  SYCL implementation, and the remaining issues are thought to be in MSVC.
- There are known issues and limitations in virtual functions
    functionality, such as:
    - Optional kernel features handling implementation is not complete yet.
    - AOT support is not complete yet.
    - A virtual function definition and definitions of all kernels using it
    must be in the same translation unit. Please refer to
    [`sycl/test-e2e/VirtualFunctions`](https://github.com/intel/llvm/tree/b23d69e2c3fda1d69351137991897c96bf6a586d/sycl/test-e2e/VirtualFunctions)
    to see the list of working and non-working examples.
