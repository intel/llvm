# May'20 release notes

Release notes for the commit range ba404be..67d3d9e

## New features
  - Implemented [reduction extension](doc/extensions/Reduction/Reduction.md)
    for `sycl::handler::parallel_for` accepting a `sycl::nd_range` object
    [bb73d926] [04a360a] [05625f1]
  - XPTI instrumentation has been added to the runtime to capture semantic and
    execution trace information for constructing the task graphs for offline
    graph and performance analysis [9bf81eb] [cece82e]

## Improvements
### SYCL Frontend and driver changes
  - Added a diagnostic for implicit declaration of kernel function type
    [20e8cde]
  - Added a diagnostic on attempt to set C compilation together with `-fsycl`
    [4cf1610]
  - Improved diagnostics on attempt to use not supported on device types (such
    as `__int128`, zero length arrays) to catch cases when they are used as
    `typedef` or `auto` [808c5c8]
  - Reduced possibility to load incorrect version of the OpenCL headers by
    reording default include paths.
  - Treat `.lo` file extension as a static archive [82b93be]
  - Added a diagnostic on attempt to use `long double` in the device code
    [62f841d]
  - Improved handling of AOCX based archives on Windows [f117aa4]
  - Removed object format designator used for Windows device binaries [0052d08]
  - Improved `-Xsycl-target` option parsing [f62de21]
  - `-g` and `-O0` options are now imply `-g` and `-cl-opt-disable` for the
    device compilation [779a601]

### SYCL headers and runtime
  - Improved handling of host accessors with read-only type of access. Now they
    do not trigger redundant memory copy operation [de85c35e]
  - Default selector doesn't select devices of accelerator type anymore
    [b217bc5]
  - Added more user friendly diagnostics on errors during kernel submission
    [be42ff7]
  - Implemented `get_native` for CUDA backend which allows querying native
    handles of SYCL objects: `sycl::queue`, `sycl::event`, `sycl::context`,
    `sycl::device` [2d71c8e]
  - Added support for 0-dim `sycl::accessor` in `sycl::handler::copy`
    [5666107] [aedd449]
  - Added support for various rounding modes for non-host devices in
    `sycl::vec::convert` method [7e3cca4]
  - Added support for more image channel types for `half4` data type on the host
    device [9b1d8b8]
  - Changed logic of `SYCL_BE` environment variable which now forces the SYCL RT
    to consider only devices of the specified backend during the device
    selection [937fec1]
  - `libsycl.so` library is now versioned [4370630]
  - [fd14167]
  - Added a diagnostic on attempt to construct a `sycl::queue` passing a
    `sycl::context` which is not bound to a passed `sycl::device`[9e79d31]

### Misc
  - `opencl-aot` tool is now included in the `sycl-toolchain` target [ccc0c27]
  - [cece82e]
  - Added `sycl-ls` utility for listing devices discovered/selected by SYCL RT
    [cc0c33b]

### Documentation
  - Improved [contribution guidelines](../CONTRIBUTING.md) [2f5cd28]
  - Updated prerequisites in GetStartedGuide(doc/GetStartedGuide.md) [5d0d034]
  - Published a [proposal](doc/extensions/KernelRHSAttributes/SYCL_INTEL_attribute_style.asciidoc)
    for function-type attributes (right-sided) for kernel attributes [5d5351b]
  - The [compiler and runtime design doc](doc/CompilerAndRuntimeDesign.md) has
    been updated to describe the CUDA target and reflect changed action graphs
    [91b597b] [212a26c]
  - [ExtendedAtomics documentation](doc/extensions/ExtendedAtomics/README.md)
    has been updated [1084685]
  - Published [sycl_bitcast extension](doc/extensions/Bitcast/SYCL_INTEL_bitcast.asciidoc)
  - Published a [proposal](doc/extensions/StaticLocalMemoryQuery/SYCL_INTEL_static_local_memory_query.asciidoc)
    which adds ability to query max local size which is used by a specific
    kernel and a specific device.
  - Published [device_specific_kernel_queries](doc/extensions/DeviceSpecificKernelQueries/SYCL_INTEL_device_specific_kernel_queries.asciidoc)
    extension which rephrases work group queries as device-specific kernel
    queries [4c07ff8]
  - Added more information about the [plugin interface (PI)](doc/PluginInterface.md)
    [0614e9a]
  - [Contribution guidelines](../CONTRIBUTING.md) were simplified, now sign-off
    line is not required [7886fd8]
  - Added missing constructors and member functions in
    [reduction extension proposal](doc/extensions/Reduction/Reduction.md)
    [f695479]
  - Published [parallel_for simplification extension](doc/extensions/ParallelForSimpification/SYCL_INTEL_parallel_for_simplification.asciidoc) [856a777]
  - Added memory scope to [ExtendedAtomics extension](doc/extensions/ExtendedAtomics/SYCL_INTEL_extended_atomics.asciidoc) [f8e11e0]
  - Published [math array extension](doc/extensions/MathArray/SYCL_INTEL_math_array.asciidoc) [36c5041]
  - Added more comments that describe Scheduler design [ad441a0]
  - Published [extension mechanism proposal](doc/extensions/ExtensionMechanism/SYCL_INTEL_extension_api.asciidoc) [cf65794]

## Bug fixes
### SYCL Frontend and driver changes
  - Fixed bug in hierarchical parallelism implementation related to using a
    private address of the `parallel_for_work_group` lambda object by all work
    items in the work group [16f64b8]
  - Fixed a problem with errors that are reported when `-fsycl-device-only`
    option is used [7f924a8]
  - Fixed a crash which happened when a specialization constant is referenced
    twice [ce020c9]
  - Fixed a possible filename collision when `-save-temps` is used [99fa86f]
  - Fixed a hang which could happen on CUDA backend when
    `sycl::group::async_work_group_copy` is used [f836604]

### SYCL headers and runtime
  - Ignore exceptions that could happen during sycl objects destruction to avoid
    terminates [cbd8a72]
  - Fixed an issue with dependency vector invalidation in some cases [2551b19]
  - Resolved a conflict with `min`/`max` macro that can be defined by
    `windows.h` [e7d4537]
  - Fixed an issue with using wrong CUDA context when creating an event
    [6788713]
  - Fixed a crash which happened when no plugin is available [d15de0b]
  - Fixed float to half-type conversion for small numbers [3a9a1a22]
  - Fixed `sycl::intel::sub_group::broadcast` which was incorrectly mapped to
    SPIRV intrinsic [24471bc]
  - Fixed an issue when a sub-buffer was accessing incorrect memory [4c4054b]
  - Fixed race which could happen when `sycl::program` API is used from multiple
    threads [e88a611]
  - [686e32b]
  - Make `cl::sycl::event::get` method to be constant [03208c0]
  - Fixed a crash which could happen when host accessors are created in
    multiple threads [1d13f84]
  - Fixed a problem with `sycl::program::set_spec_constant` which set a value
    for all `sycl::program` objects rather than for an object it was called for
    only [c22e34b]

### Misc
  - Fixed an issue with gdb xmethod not working when `run` command is issued
    multiple times without terminating the session [042d981]
  - Fixed an issue with platform selection in opencl-aot tool that was caused
    by an identical supported platform name for CPU and GPU.

## Known issues
  - [new] A crash can happen in a multithreaded application if two threads call
    an API which implies waiting for an event. No known workaround.
  - The format of the object files produced by the compiler can change between
    versions. The workaround is to rebuild the application.
  - The SYCL library doesn't guarantee stable API/ABI, so applications compiled
    with older version of the SYCL library may not work with new one.
    The workaround is to rebuild the application.
    [ABI policy guide](doc/ABIPolicyGuide.md)
  - Using `cl::sycl::program` API to refer to a kernel defined in another
    translation unit leads to undefined behavior
  - Linkage errors with the following message:
    `error LNK2005: "bool const std::_Is_integral<bool>" (??$_Is_integral@_N@std@@3_NB) already defined`
    can happen when a SYCL application is built using MS Visual Studio 2019
    version below 16.3.0
    The workaround is to enable `-std=c++17` for the failing MSVC version.

# March'20 release notes

Release notes for the commit range e8f1f29..ba404be

## New features
  - Initial CUDA backend support [7a9a425]
  - [SYCL][FPGA] Implement IO pipes interface [c900248]
  - Added the implementation of [GroupAlgorithms extension](doc/extensions/GroupAlgorithms/SYCL_INTEL_group_algorithms.asciidoc)
    [8bfa107]
  - Added a partial implementation of [sub group algorithms extension](doc/extensions/SubGroupAlgorithms/SYCL_INTEL_sub_group_algorithms.asciidoc)
    [017af4e]
  - New attributes for Intel FPGA devices: `intelfpga::force_pow2_depth`,
    `intelfpga::loop_coalesce`, `intelfpga::speculated_iterations`,
    `intelfpga::disable_loop_pipelining`, `intelfpga::max_interleaving`
    [73dd705][a5b9804]
  - Added support for `intel::reqd_work_group_size` attribute [8eb588d]
  - Added support for specialization constants feature which is based on
    [SYCL Specialization Constant proposal](https://github.com/codeplaysoftware/standards-proposals/blob/master/spec-constant/index.md) [29abe37]

## Improvements
### SYCL Frontend and driver changes
  - Added a diagnostic on attempt to declare or use non-const static variable
    inside device code [7743e86] [1853516]
  - Relaxed requirements for kernel types even more. Now by default they should
    have trivial copy constructor and trivial destructor [17aac3c]
  - Changed `std::numeric_limits<sycl::half>` to constexpr functions [85d7a5e]
  - Added a diagnostic on attempt to use zero length arrays inside device code
    [e6ce614]
  - Added support for math functions 'fabs' and 'ceil' in device code [f41309b]
  - Added a diagnostic (warning) on attempt to append new device object to
    an archive which already contains an AOT-compiled device object [9d348eb]
  - Added a diagnostic on attempt to use functions which have no definition in
    the TU and are not marked with `SYCL_EXTERNAL` macro inside device code
    [a3b340b]
  - Added a diagnostic on attempt to use thread local storage inside device code
    [eb373c4]
  - Removed arch designator from the default output file name when compiling
    with `-fsycl-link` option. Now an output file has just a flat name based on
    the first input file [dc729a7]
  - The SYCL headers were moved from `lib/clang/11.0.0/include` to
    `include/sycl` to support mixed compilers [39501f6]
  - Added support for the GCC style inline assembly in the device code [6f4e007]
  - Improved fat static library support: the driver now consider for offloading
    static libraries which are passed on the command line as well as libraries
    passed as part of the linker options. This effectively negates the need to
    use `-foffload-static-lib` and `-foffload-whole-static-lib` options which
    are deprecated now.
  - The `SYCL_EXTERNAL` macro is now allowed to be used with class member
    functions [3baec18]
  - Set `aux-target-cpu` for the device compilation which sets AVX and other
    necessary macro based on a target [f953fda]

### SYCL headers and runtime
  - Changed `sycl::context` and `sycl::queue` constructors to be explicit to
    avoid unintended conversions [c220eb8][3b6799a]
  - Added a diagnostic on setting `SYCL_DEVICE_TYPE` environment variable to an
    incorrect value [0125496]
  - Improved error codes which are encoded in the SYCL exceptions [04ee17c]
  - Removed functions that use float type in the fallback library for fp64
    complex [6ccd84a0]
  - Added support for `RESTRICT_WRITE_ACCESS_TO_CONSTANT_PTR` macro which allows
    to enable diagnostic on writing to a raw pointer obtained from a
    `sycl::constant_ptr` object [c9ed5b2]
  - Added support for USM extension for CUDA backend [498d56c]

### Documentation
  - Refactored [USM specification](doc/extensions/USM/USM.adoc) [0438422]
  - Added [GroupAlgorithms extensions](doc/extensions/GroupAlgorithms/)
    as replacement of GroupCollectives extension [c181fdb][b18a566]
  - Doxygen documentation is now rendered to GitHub Pages. An initial
    implementation is available [online](https://intel.github.io/llvm-docs/doxygen/annotated.html)
    [29d9cc2]
  - More details have been added about the `-fintelfpga` option in the
    [Compiler User Manual](doc/SYCLCompilerUserManual.md) [4b03ddb]
  - Added [SYCL_INTEL_enqueue_barrier extension document](doc/extensions/EnqueueBarrier/enqueue_barrier.asciidoc)
    [6cfd2cb]
  - Added [standard layout relaxation extension](doc/extensions/RelaxStdLayout/SYCL_INTEL_relax_standard_layout.asciidoc)
    [ce53521]
  - Deprecated SubGroupNDRange extension [d9b178f]
  - Added extension for base sub-group class:
    [SubGroup](doc/extensions/SubGroup/SYCL_INTEL_sub_group.asciidoc) [d9b178f]
  - Added extension for functions operating on sub-groups:
    [SubGroupAlgorithms](doc/extensions/SubGroupAlgorithms/SYCL_INTEL_sub_group_algorithms.asciidoc)
    [d9b178f]
  - Added extension introducing group masks and ballot functionality:
    [GroupMask](doc/extensions/GroupMask/SYCL_INTEL_group_mask.asciidoc)
    [d9b178f]
  - The project has been renamed to "oneAPI DPC++ Compiler", all documentation
    has been fixed accordingly [7a2e75e]

## Bug fixes
### SYCL Frontend and driver changes
  - Fixed a problem with compiler not being able to find a dependency file when
    compiling AOT to an object for FPGA [7b58b01]
  - Fixed a problem with host object not being added to the partial link step
    when compiling from source and using `-foffload-static-lib` option [1a951cb]
  - Reversed `reqd_work_group_size` attribute to match SYCL behavior [1da6fbe]
  - Fixed dependency output location when `/Fo<dir>` is given [2b6f4f4]
  - Fixed a crash which happened when no kernel name is passed to the
    `sycl::handler::parallel_for` [fadaa59]

### SYCL headers and runtime
  - Fixed `sycl::queue::wait()` which was not waiting for event associated with
    USM operation [850fb9f]
  - Fixed problem with reporting wrong error message on the second attempt to
    build program if the first attempt failed [9a34a11]
  - Fixed an issue which could happen when `sycl::event::wait` is called from
    multiple threads [3da5473]
  - Aligned `sub_group::store` signature between host and device [b3a9426]
  - Fixed `sycl::program::get_compile_options` and
    `sycl::program::get_build_options` to return correct values [03326f7]
  - Fixed `sycl::multi_ptr`'s methods that were incorrectly enabled/disabled on
    device/host [401d174]
  - Fixed incorrect dependency handling when creating sub-buffers which could
    lead to data races [45e39bd]
  - Reversed reported max work-group size for a device to align with work-group
    size reversing before kernels launch [72b7dee]
  - Fixed incorrect handling of kernels that use hierarchical parallelism when
    `-O0` option is passed to the clang [fd8ae8a]
  - Changed names of SYCL internal variables to avoid conflict with commonly
    used macros: `SUCCESS`, `BLOCKED` and `FAILED` [0f7e361]
  - Fixed a bug when a host device was always included in the device list
    returned by `sycl::device::get_devices` [6cf590f]
  - Fixed a problem with passing `sycl::vec` object to
    `sycl::group::async_work_group_copy` [20aa83e]
  - Fixed behavior of sycl::malloc_shared to return nullptr for the allocation
    size of zero or less byte, and the behavior of sycl::free functions to
    ignore the deallocation request from nullptr [d596593]
  - Fixed a possible problem with selecting work-group size which is bigger than
    max allowed work-group [b48f08f]
  - Fixed an issue which causes errors when using sub-buffers [5d1d716]
  - Changed the implementation of the buffer constructor from a pair of
    iterators. Now, data is not written back to the host on destruction of the
    buffer unless the buffer has a valid non-null pointer specified via the
    member function set_final_data [fb72758]
  - Fixed a problem with incorrect acceptance of a lambda which takes an
    argument of the `sycl::id` type in the `sycl::handler::parallel_for` version
    which takes a `sycl::ndrange` object [0408899]
  - Resolved circular dependency between `sycl::event` and `sycl::queue`
    [8c71dcb]


## Known issues
  - The format of the object files produced by the compiler can change between
    versions. The workaround is to rebuild the application.
  - The SYCL library doesn't guarantee stable API/ABI, so applications compiled
    with older version of the SYCL library may not work with new one.
    The workaround is to rebuild the application.
  - Using `cl::sycl::program` API to refer to a kernel defined in another
    translation unit leads to undefined behavior
  - Linkage errors with the following message:
    `error LNK2005: "bool const std::_Is_integral<bool>" (??$_Is_integral@_N@std@@3_NB) already defined`
    can happen when a SYCL application is built using MS Visual Studio 2019
    version below 16.3.0
    The workaround is to enable `-std=c++17` for the failing MSVC version.

## Prerequisites
### Linux
  - Experimental Intel(R) CPU Runtime for OpenCL(TM) Applications with SYCL
    support from the release package https://github.com/intel/llvm/releases/
  - The latest version of Intel(R) Graphics Compute Runtime for OpenCL(TM) from
    https://github.com/intel/compute-runtime/releases/
### Windows
  - Experimental Intel(R) CPU Runtime for OpenCL(TM) Applications with SYCL
    support from the release package https://github.com/intel/llvm/releases/
  - The latest version of Intel(R) Graphics Compute Runtime for OpenCL(TM) from
    https://downloadcenter.intel.com/

Please, see the runtime installation guide [here](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md#install-low-level-runtime)



# February'20 release notes

Release notes for commit e8f1f29

## New features
  - Added `__builtin_intel_fpga_mem` for the FPGA SYCL device. The built-in is
    used to indicate the characteristics of the load-store unit (LSU) to be used
    when de-referencing the pointer [1e33c01]
  - Added support for the `intelfpga::no_global_work_offset` attribute
    [8bed533] [5a9058b]
  - Added basic xmethod implementation for `sycl::accessor::operator[]` to make
    it callable from gdb command line [d6be8ff]
  - Added device libraries for standard math like `std::cos` and `std::complex`
    type [7abd9d5]

## Improvements
### SYCL Frontend and driver changes
  - Added support for passing a non-type template parameter to the `loop_unroll`
    attribute [8d7a32a]
  - Relaxed the standard layout requirement for kernel arguments. Now by default
    they should be trivially copyable. The `-sycl-std=1.2.1` driver option turns
    standard layout requirement "on" [3adb4a5]
  - Added diagnostic on using `__float128` type in the device code [659efdf]
  - The `intelfpga::max_private_copies` has been renamed to
    `intelfpga::private_copies` [97a199f]
  - Prevented duplication of error diagnostics for `-fsycl` [3a0b62e]
  - Added diagnostic when attempting to use existing FPGA static library with
    additional device code [6431be6]
  - Added support for non-type template parameters for FPGA memory attributes
    [ffcad03]

### SYCL headers and runtime
  - The `SYCL_DEVICE_WHITE_LIST` control was renamed to `SYCL_DEVICE_ALLOWLIST`
    [4df18fa]
  - Added a leaf limit to the execution graph to avoid leaf bloat in
    applications that have an overwhelming number of command groups that can
    be executed in parallel [7c293e2]
  - Added `get_range()` method to the image and local accessors to align with
    the SYCL specification [8ed5566]
  - Added a diagnostic on attempt to create a context from devices that are
    bound to different platforms [8f354f7]
  - An ordered queue can now be created by passing
   `sycl::property::queue::in_order` property to the `sycl::queue` constructor
    [c855520]
  - Added a diagnostic on attempt to create an accessor with an unsupported type
    [306624e]
  - Made host device return `nullptr` for bad USM allocation functions
    (huge, 0, etc) [2a000d9]
  - Added templated forms of USM allocation functions [42cf5bf]
  - Added support for APIs that query properties of USM pointers [926e38e]
  - Added cleanup of finished command nodes of the execution graph in the
    situations when the `wait` for a command is called implicitly or explicitly
    [438dc49]
  - Added 2 `sycl::queue` constructors accepting `sycl::context` and
    `sycl::device` arguments [c81c1c5]

### Documentation
  - Added [documentation](doc/extensions/QueueShortcuts/QueueShortcuts.adoc) for
    simplification of the `sycl::queue` functions [478b7c0]
  - Added [documentation](doc/extensions/ReqdWorkGroupSize/SYCL_INTEL_reqd_work_group_size.asciidoc)
    for `reqd_work_group_size` extension [c2c416a]
  - The FAQ [document](doc/FAQ.md) was introduced [e42b40e]
  - Added Ordered Queue Property
    [proposal](doc/extensions/OrderedQueue/OrderedQueue_v2.adoc) [9fa878f]
  - Added device code split options documentation to the
    [user's manual](doc/UsersManual.md) [1355aa6]
  - Added documentation for [ExtendedAtomics extension](doc/extensions/ExtendedAtomics/SYCL_INTEL_extended_atomics.asciidoc) [4445462]
  - Removed old Ordered Queue proposal and make a note of deprecation [e8f1f29]

## Bug fixes
### SYCL Frontend and driver changes
  - Fixed variable sharing passed by value to `parallel_for_work_group`
    [d8ea63a]
  - Do not produce an error if some restricted feature (e.g. exceptions) is used
    by a function only referenced in unevaluated contexts [5eae571]
  - Fixed problem with not cleaning up temporary files when device code split
    feature is enabled [d86ee2f]
  - Fixed issue with emitting wrong 'typename' keywords in the integration
    header [c19372e]
  - The SYCL target image registration functions have been renamed to avoid
    conflicts with the OpenMP registration functions [82fd970]
  - Avoid using `std::memcpy` in the device code [f39f47e]
  - Fixed `-save-temps` option when used along with `-fsycl` [f7f4699]
  - Fixed link steps triggering for libraries specified through
    `-foffload-static-lib` when no source or object file is provided [360b25b]
  - Fixed output options behavior for `-fsycl-link` on Windows [67b24d46]

### SYCL headers and runtime
  - Fixed final result saturation in the host implementation of `sycl::mad_sat`
    [54dddb4]
  - Fixed a crash when a non-nullptr_t `0x0` value is passed to the
    `sycl::buffer::set_final_data` method [6a0e279]
  - Fixed an issue with copying sub-buffer between different contexts [0867a38]
  - Resolved the problem when local accessor is a temporary object [1eed329]
  - Fixed an issue with the event not being retained when a memory object is
    constructed using interoperability constructors [0aabe7e]
  - Fixed compilation of kernels which use `sycl::stream` for FPGA device
    [c4dbaa2]
  - Fixed execution graph cleanup on memory object destruction [7a75b54]

## Known issues
  - [new] Defining a `SUCCESS` or `FAILED` can break SYCL headers because such
    names are used internally
  - [new] The format of the object files produced by the compiler can change
    between versions. The workaround is to rebuild the application.
  - [new] The SYCL library doesn't guarantee stable API/ABI, so applications
    compiled with older version of the SYCL library may not work with new one.
    The workaround is to rebuild the application.
  - Using `cl::sycl::program` API to refer to a kernel defined in another
    translation unit leads to undefined behaviour
  - Linkage errors with the following message:
    `error LNK2005: "bool const std::_Is_integral<bool>" (??$_Is_integral@_N@std@@3_NB) already defined`
    can happen when a SYCL application is built using MS Visual Studio 2019
    version below 16.3.0
    The workaround is to enable `-std=c++17` for the failing MSVC version.

## Prerequisites
### Linux
- Experimental Intel(R) CPU Runtime for OpenCL(TM) Applications with SYCL
  support version
  [2020.10.3.0.04](https://github.com/intel/llvm/releases/download/2020-02/oclcpuexp-2020.10.3.0.04_rel.tar.gz)
  is the recommended OpenCL CPU RT prerequisite for the SYCL compiler
- The Intel(R) Graphics Compute Runtime for OpenCL(TM) version
  [20.06.15619](https://github.com/intel/compute-runtime/releases/tag/20.06.15619)
  is the recommended OpenCL GPU RT prerequisite for the SYCL compiler.
<!--### Windows-->
- Experimental Intel(R) CPU Runtime for OpenCL(TM) Applications with SYCL
  support version
  [2020.10.3.0.04](https://github.com/intel/llvm/releases/download/2020-02/win-oclcpuexp-2020.10.3.0.04_rel.zip)
  is the recommended OpenCL CPU RT prerequisite for the SYCL compiler
- The Intel(R) Graphics Compute Runtime for OpenCL(TM) version
  [26.20.100.7870](https://downloadcenter.intel.com/download/29426/Intel-Graphics-Windows-10-DCH-Drivers)
  is the recommended OpenCL GPU RT prerequisite for the SYCL compiler.

Please, see the runtime installation guide [here](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md#install-low-level-runtime)


# December'19 release notes

Release notes for commit 78d80a1cc628af76f09c53673ada906a3d2f0131

## New features
  - New attributes for Intel FPGA devices : `num_simd_work_items`, `bank_bits`,
  `max_work_group_size`, `max_global_work_dim`
  [61d60b6] [7becb9d] [9053642] [6441851]
  - The infrastructure for device standard C/C++ libraries has been introduced.
    `assert` can now be used in device code [0039ee0]
  - Add opencl-aot tool for AoT compilation for Intel CPU devices [400460b]
  - Added support for `cl::sycl::intel::experimental::printf` builtin [78d80a1]
  - Implemented device code split feature: compiler can now be instructed to
    split a single device code module into multiple via the
    `-fsycl-device-code-split` option.
    [9d7dba6] [cc93bc4] [5491486] [a339d4c] [9095749]

## Improvements
### SYCL Frontend and driver changes
  - Allowed applying `cl::reqd_work_group_size` and
    `cl::intel_reqd_sub_group_size` attributes to a lambda function [b06fc66]
  - Allowed applying SYCL_EXTERNAL to functions with raw pointer arguments or
    return value if `-Wno-error=sycl-strict` option is passed [2840458]
  - Added support for FPGA device libraries in fat static libraries [d39ab73]
  - Removed limitation of virtual types in SYCL [98754ff]

### SYCL headers and runtime
  - Added support for platform name and platform version in the device allowed
    list [7f2f668]
  - Made kernels and programs caches thread-safe [118dc82]
  - It's now possible to omit `cl` namespace specifier when using SYCL API.
    `sycl::buffer` can be used instead `cl::sycl::buffer` [67e4655]
  - Diagnostic on attempt to do several actions in one command group has been
    implemented [9f8ae50]
  - `queue::submit` now throws synchronous exceptions [6a83d14]
  - Enqueue pending tasks after host accessor destructor [80d17b2]
  - Implemented more "flat" kernel submission `cl::sycl::queue` methods
    [c5318c5]

### Documentation
  - Added support for generation of SYCL documentation with Doxygen [de418d6]
  - [Design document](doc/extensions/C-CXX-StandardLibrary/C-CXX-StandardLibrary.rst)
    which describes design of C/C++ standard library support has been added

## Bug fixes
### SYCL Frontend and driver changes
  - Fixed problem which prevented attaching multiple attributes to a SYCL kernel
    [b77e5b7]

### SYCL headers and runtime
  - Fixed a possible deadlock with host accessor' creation from multiple threads
    [d1c6dbe]
  - Compatibility issues that can happen when the SYCL library and SYCL
    application are compiled with different version of standard headers are
    fixed.  [d854643]
  - Fixed compile error for `handler::copy` with `-fsycl-unnamed-lambda`
    [e73d2ce]
  - Fixed error which happened when autogenerated (`-fsycl-unnamed-lambda`)
    kernel name contained `cl::sycl::half` type [514fc0b]
  - Fixed crash on submitting a kernel with zero local size on the host device
    [b6806ea]
  - Made `vec::load` and `vec::store` member functions templated to align with
    the SYCL specification [4bd76de]
  - Made exceptions that are thrown in the copy back routine during SYCL
    memory object destructor asynchronous [2d6dcd0]
  - Fixed creation of host accessor to sub-buffer leading to crash in some
    scenarios [f607520]
  - Fixed build error on Windows caused by RuntimeLibrary value inconsistency
    for LLVM and apps/tests linked with it [f9296b6]

## Known issues
- [new] The size of object file is increased compared to the older compiler
  releases due to the recent fat object layout changes in the open source LLVM.
- Using `cl::sycl::program` API to refer to a kernel defined in another
  translation unit leads to undefined behaviour
- Linkage errors with the following message:
  `error LNK2005: "bool const std::_Is_integral<bool>" (??$_Is_integral@_N@std@@3_NB) already defined`
  can happen when a SYCL application is built using MS Visual Studio 2019
  version below 16.3.0
  For MSVC version having the error the workaround is to use -std=c++17 switch.

## Prerequisites
### Linux
- Experimental Intel(R) CPU Runtime for OpenCL(TM) Applications with SYCL
  support version
  [2020.9.1.0.18_rel](https://github.com/intel/llvm/releases/download/2019-12/oclcpuexp-2020.9.1.0.18_rel.tar.gz)
  is recommended OpenCL CPU RT prerequisite for the SYCL compiler
- The Intel(R) Graphics Compute Runtime for OpenCL(TM) version
  [19.48.14977](https://github.com/intel/compute-runtime/releases/tag/19.48.14977)
  is recommended OpenCL GPU RT prerequisite for the SYCL compiler.
### Windows
- Experimental Intel(R) CPU Runtime for OpenCL(TM) Applications with SYCL
  support version
  [2020.9.1.0.18_rel](https://github.com/intel/llvm/releases/download/2019-12/win-oclcpuexp-2020.9.1.0.18_rel.zip)
  is recommended OpenCL CPU RT prerequisite for the SYCL compiler
- The Intel(R) Graphics Compute Runtime for OpenCL(TM) version
  [26.20.100.7463](https://downloadcenter.intel.com/download/29195/Intel-Graphics-Windows-10-DCH-Drivers)
  is recommended OpenCL GPU RT prerequisite for the SYCL compiler.

Please, see the runtime installation guide [here](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedWithSYCLCompiler.md#install-low-level-runtime)

# November'19 release notes

Release notes for commit e0a62df4e20eaf4bdff5c7dd46cbde566fbaee90

## New features

## Improvements
### SYCL Frontend and driver changes
  - Added more diagnostics on incorrect usage of SYCL options [f8a8785]
  - Added diagnostic on capturing values by reference in a kernel lambda
    [0e7639a]
  - Removed `--sycl` option which was an alias to `-fsycl-device-only`
  - Added support for using integral template parameter as an argument to
    `intelfpga::ivdep`, `intelfpga::ii` and `intelfpga::max_concurrency`
    attributes [6a283ae]
  - Improved diagnostic for using incorrect triple for AOT [6af65e1]

### SYCL headers and runtime
  - `cl::sycl::pipe` class was moved to `cl::sycl::intel` namespace [23af059]
  - Added new query `info::device::kernel_kernel_pipe_support` which indicates
    whether device supports SYCL_INTEL_data_flow_pipes extension [8acf36e]
  - Added support for copying accessor with arbitrary dimensions in
    `handler::copy` [ff30897]
  - Cache result of compilation and kernel creation of programs created via
    `cl::sycl::program::build_with_kernel_type` without build options
    [8c2e09a] [4ba499c] [d442364]
  - Added diagnostic (static_assert) for dimensions mismatch when passing
    `cl::sycl::vec` objects to SYCL builtins [eb3ac32]
  - Added diagnostic (exception) on attempt to create invalid sub-buffer
    [0bd58df]
  - Added `single_task` and `parallel_for` methods to
    `cl::sycl::ordered_queue::single_task`. These provide an alternative way to
    launch kernels [1d2f7ce]
  - Added forms of USM functions for allocations that take a queue reference
    [f57c2b6]
  - Added C++ deduction guides to
      - `cl::sycl::range`
      - `cl::sycl::id`
      - `cl::sycl::nd_range`
      - `cl::sycl::vec`
      - `cl::sycl::multi_ptr`
      - `cl::sycl::buffer`
    [644013e] [3546a78] [728b904] [dba1c20]
  - Added a new buffer constructor which takes a contiguous container as an
    argument [dba1c20]
  - Added support of 1 byte type to load and store method of
    `cl::sycl::intel::sub_group` class [c8cffcc]
  - Improved error reporting for kernel enqueue process [d27cff2]
  - Instance of memory objects are now preserved on host after host accessor
    destruction [16ae15a]
  - Added support for `SYCL_DEVICE_WHITE_LIST` control which can be used to
    specify devices that are visible to the SYCL implementation [4ad5263]

### Documentation
  - Updated build instructions for Windows and added instructions for setting
    up FPGA emulator device to [get started guide](doc/GetStartedWithSYCLCompiler.md)
    [6d0b326] [b2bb35b] [a7336c2]
  - Updated [SYCLCompilerAndRuntimeDesign](doc/SYCLCompilerAndRuntimeDesign.md)
    to use proper names of AOT related options [b3ee6a2]
  - Added [unnamed lambda extension](doc/extensions/UnnamedKernelLambda/SYCL_INTEL_unnamed_kernel_lambda.asciidoc)
    draft [47c4c71]
  - Added [kernel restrict all extension](doc/extensions/KernelRestrictAll/SYCL_INTEL_kernel_restrict_all.asciidoc)
    draft [47c4c71]
  - Added initial draft of [data flow pipes extension](doc/extensions/DataFlowPipes/data_flow_pipes.asciidoc)
    proposal [ee2b482]
  - [USM doc](doc/extensions/USM/USM.adoc) was updated with new version of
    allocation functions [0c32410]
  - Added draft version of [group collectives extension](doc/extensions/GroupCollectives/GroupCollectives.md)
    [e0a62df]
  - Added [deduction guides extension](doc/extensions/deduction_guides/SYCL_INTEL_deduction_guides.asciidoc)
    [4591d74]
  - Improved [environment variables controls documentation](doc/SYCLEnvironmentVariables.md)
    [6aae4ef]

## Bug fixes
### SYCL Frontend and driver changes
  - Fixed a problem with option duplication during offload [1edd217]
  - Modified the driver to unbundle files with no extension like objects
    [84992a5]
  - Fixed AOT compilation for FPGA target on Windows [6b789f9] [525da51]
  - Fixed FPGA AOT compilation when using FPGA AOCX device archives
    [ec738f2] [16c530a]
  - Fixed problem causing abort when `-fsycl` and `-lstdc++` passed
    simultaneously [e437fd4]
  - Fixed `-o` option for FPGA AOT on Windows
    [ddd24a3]
  - Default `-std=c++` setting for device compilations was removed to avoid
    `__cplusplus` differences between host and device compilation. [29cabdf]
  - Suppressed warning about cdecl calling convention [0d2dab4]
  - Fixed problem when two different lambdas with the same signature get the
    same mangled name [d6aa11b]
  - Removed the error being issued for pseudo-destructor expressions [3c9006e]
  - Removed the device linking step for cases with no valid objects being
    provided from the device compilation [25bbe42]

### SYCL headers and runtime
  - Implemented a workaround for the issue with conversion of one element
    swizzle to a scalar [f752698]
  - Fixed a UX problem with a sycl::stream mixing output of several work-items
    in some cases [377b3fa]
  - Fixed problem with linker options getting passed to the bundler instead of
    the linker when -foffload-static-lib is supplied [e437fd4]
  - Fixed problem with using wrong offset inside `cl::sycl::handler::copy`
    [5c8e81f]
  - Fixed `get_count` and `get_size` methods of `cl::sycl::accessor` class that
    used to return incorrect values for non-local accessor created with range
    [9dd68c5]
  - Fixed issue with asynchronous exceptions not being associated with the
    returned event [6c512d3]
  - Fixed issue with host accessor not always providing exclusive access to
    a memory object [16ae15a]
  - Fixed issue with ignoring `cl::sycl::property::buffer::use_host_ptr`
    property [16ae15a]

## Known issues
- [new] `cl::sycl::program` constructor creates unnecessary context which is
  bound to the device which is chosen by `cl::sycl::default_selector`. This can
  lead to unexpected behavior, for example, process may hang if default selector
  chooses Intel FPGA emulator device but target device is Intel OpenCL CPU one
- [new] Using cl::sycl::program API to refer to a kernel defined in another
  translation unit leads to undefined behaviour
- [new] -fsycl-unnamed-lambda doesn't work with kernel names that contain
  cl::sycl::half type
- The addition of the static keyword on an array in the presence of Intel
  FPGA memory attributes results in the empty kernel after translation
- A loop's attribute in device code may be lost during compilation
- Linkage errors with the following message:
  `error LNK2005: "bool const std::_Is_integral<bool>" (??$_Is_integral@_N@std@@3_NB) already defined`
  can happen when a SYCL application is built using MS Visual Studio 2019
  version below 16.3.0
  For MSVC version having the error the workaround is to use -std=c++17 switch.

## Prerequisites
### Linux
- Experimental Intel(R) CPU Runtime for OpenCL(TM) Applications with SYCL
  support version
  [2019.9.12.0.12_rel](https://github.com/intel/llvm/releases/download/2019-11/oclcpuexp-2019.9.12.0.12_rel.tar.gz)
  is recommended OpenCL CPU RT prerequisite for the SYCL compiler
- The Intel(R) Graphics Compute Runtime for OpenCL(TM) version
  [19.48.14977](https://github.com/intel/compute-runtime/releases/tag/19.48.14977)
  is recommended OpenCL GPU RT prerequisite for the SYCL compiler.
### Windows
- Experimental Intel(R) CPU Runtime for OpenCL(TM) Applications with SYCL
  support version
  [2019.9.12.0.12_rel](https://github.com/intel/llvm/releases/download/2019-11/win-oclcpuexp-2019.9.12.0.12_rel.zip)
  is recommended OpenCL CPU RT prerequisite for the SYCL compiler
- The Intel(R) Graphics Compute Runtime for OpenCL(TM) version
  [26.20.100.7463](https://downloadcenter.intel.com/download/29195/Intel-Graphics-Windows-10-DCH-Drivers)
  is recommended OpenCL GPU RT prerequisite for the SYCL compiler.

Please, see the runtime installation guide [here](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedWithSYCLCompiler.md#install-low-level-runtime)

# October'19 release notes

Release notes for commit 918b285d8dede6ab0561fccc622f71cb858849a6

## New features
  - `cl::sycl::queue::mem_advise` method was implemented [4828db5]
  - `cl::sycl::handler::memcpy` and `cl::sycl::handler::memset` methods that
     operate on USM pointer were implemented [d9e8467]
  - Implemented `ordered queue` [extension](doc/extensions/OrderedQueue/OrderedQueue.adoc)
  - Implemented support for half type in sub-group collectives: broadcast,
    reduce, inclusive_scan and exclusive_scan [0c78bc8]
  - Added `cl::sycl::intel::ctz` built-in. [6a96b3c]
  - Added support for SYCL_EXTERNAL macro.
  - Added support for passing device function pointers to a kernel [dc9db24]
  - Added support for USM on host device [5b0952c]
  - Enabled C++11 attribute spelling for clang `loop_unroll` attribute [2f1e243]
  - Added full support of images on host device
  - Added support for profiling info on host device [6c03c4f]
  - `cl::sycl::handler::prefetch` is implemented [feeacc1]
  - SYCL sub-buffers is mapped to OpenCL sub-buffers

## Improvements
### SYCL Frontend and driver changes
  - Added Intel FPGA Command line interface support for Windows [55ebcae]
  - Added support for one-step compilation from source with `-fsycl-link`
    [55ebcae]
  - Enabled additional aoc options for dependency files input and output report
    [55ebcae]
  - Suppressed warning `"_declspec attribute 'dllexport' is not supported"`
    when run with `-fsycl`. Emit error when import function is called in the
    sycl kernel. [b10bdbb]
  - Changed `-fsycl-device-only` to override `-fsycl` option [d429243]
  - Added user-friendly diagnostic for unsupported math built-in functions usage
    in kernel [0476352]
  - The linking stage is now skipped if -fsycl-device-only option is passed
    [93178d1]
  - When unbundling static libraries on Windows, do not extract the host section
    as it is not being used. This fixes possible disk usage issues when working
    with fat static libraries [93ab97e]
  - Passing `-fsycl-help` with `-###` option now prints the actual call to tool
    being made. [8b8bfa9]
  - Allow for `-gN` to override default setting with `-fintelfpga` [3b20615]
  - Update sub-group reduce/scan syntax [cd8194d]
  - Prevent libraries from being considered for unbundling on Windows [3438a48]
  - Improved Windows behaviors for calling `lib.exe` when creating an archive
    for Intel FPGA AOT [e7afcb1]

### SYCL headers and runtime
  - Removed suppression of exceptions thrown by async_handler from
    `cl::sycl::queue` destructor [61574d8]
  - Added the support for output operator for half data types [6a2cd90]
  - Improved efficiency of stream output of `cl::sycl::h_item` for Intel FPGA
    device [80e97a0]
  - Added support for `std::numeric_limits<cl::sycl::half>` [6edca52]
  - Marked barrier flags as constexpr to avoid its extra runtime translation
    [5635959]
  - Added support for unary plus and minus for `cl::sycl::vec` class
  - Reversed mapping of SYCL range/ID dimensions to OpenCL, to provide expected
    performance through unit stride dimension. The highest dimension in SYCL
    (e.g. r2 in cl::sycl::range<3> R(r0,r1,r2)) now maps to the lowest dimension
    in OpenCL (e.g. an enqueue of size_t[3] cl_R = {r2,r1,r0}). The same applies
    to range and ID queries, in kernels defined through OpenCL interop.
    [40aa3f9]
  - Added support for constructing `cl::sycl::image` without host ptr but with
    pitch provided [d1931fd]
  - Added `sycld` library on Windows which is compiled using `/MDd` option.
    This library should be used when SYCL application is compiled with `/MDd`
    option to avoid ABI issues [71a75c0]
  - Added driver and runtime support for AOT-compiled images for multiple
    devices.  This handles the case when the device code is AOT-compiled for
    multiple targets [0d4eb49] [bcf38cf]

### Documentation
  - Get started [guide](doc/GetStartedWithSYCLCompiler.md) was reworked
    [9050a98] [94ee028]
  - Added SYCL compiler [command line guide](doc/SYCLCompilerUserManual.md)
    [af63c6e]
  - New [document](doc/SYCLPluginInterface.md) describing the SYCL Runtime
    Plugin Interface [bffdbcd]
  - Updated interfaces in [Sub-group extension specification](doc/extensions/SubGroupNDRange/SubGroupNDRange.md)
    [cc6e4ae]
  - Updated interfaces in [USM proposal](doc/extensions/USM/USM.adoc)
    [a6d7e12] [d9e8467]

## Bug fixes
### SYCL Frontend and driver changes
  - Fixed problem with using aliases as kernel names [a784071]
  - Fixed address space in generation of annotate attribute for static vars and
    global Intel FPGA annotation [800c8c0]
  - Suppressed emitting errors for TLS declarations [ddc1a7f]
  - Suppressed device code link warnings that happen during linking `fat`
    and `non-fat` object files [b38a8e0]
  - Fixed pointer width on 64-bit version of Windows [63e2b19]
  - Fixed integration header generation when kernel name type is defined in cl,
    sycl or detail namespaces [5d22a8e]
  - Fixed problem with incorrect generation of output filename caused by
    processing of libraries in SYCL device toolchain [d3d9d2c]
  - Fixed problem with generation of depfile information for Intel FPGA AOT
    compilation [fbe951f]
  - Fixed generation of help message in case of `-fsycl-help=get` option passed
    [8b8bfa9]
  - Improved use of `/Fo` on Windows in offload situations so intermediate
    temporary files are not renamed [6984794]
  - Resolved problem with unnamed lambdas having the same name [f4d182f]
  - Fixed -fsycl-add-targets option to support multiple triple:binary arguments
    and to emit diagnostics for invalid target triples [21fa901]
  - Fixed AOT compilation for GEN devices [cd2dd9b]

### SYCL headers and runtime
  - Fixed problem with using 32 bits integer type as underlying type of
    `cl::sycl::vec` class when 64 bits integer types must be used on Windows
    [b4998f2]
  - `cl::sycl::aligned_alloc*` now returns nullptr in case of error [9266cd5]
  - Fixed bug in conversion from float to half in the host version of
    `cl::sycl::half` type [6a2cd90]
  - Corrected automatic/rte mode conversion of `cl::sycl::vec::convert` method
    [6a2cd90]
  - Fixed memory leak related to incorrectly destroying command group objects
    [d7b5c0d]
  - Fixed layout and alignment of objects of 3 elements `cl::sycl::vec` type,
    now they occupy memory for 4 elements underneath [32f0cd5] [8f7f4a0]
  - Fixed problem with reporting the same asynchronous exceptions multiple times
    [9040739]
  - Fixed a bug with a wrong success code being returned for non-blocking pipes,
    that was resulting in incorrect array data passing through a pipe. [3339c45]
  - Fixed problem with calling atomic_load for float types in
    `cl::sycl::atomic::load`. Now it bitcasts float value to integer one then
    call atomic_load. [f4b7b17]
  - Fixed crash in case incorrect local size is passed. Now an exception is
    thrown in such cases. [1865c79]
  - `cl::sycl::vec` types aliases are now aligned with the SYCL specification.
  - Fixed `cl::sycl::rotate` method to correctly handle over-sized shift widths
    [d2e6a26]
  - Changed underlying address space of `cl::sycl::constant_ptr` from constant
    to global to avoid casts between constant and generic address spaces
    [38c2960]
  - Aligned `cl::sycl::range` class with the SYCL specification by removing its
    default constructor [d3b6a49]
  - Fixed several thread safety problems in `cl::sycl::queue` class [349a0d3]
  - Fixed compare_exchange_strong to properly update expected inout parameter
    [627a137]
  - Fixed issue with host version of `cl::sycl::sub_sat` function [7865dfc]
  - Fixed initialization of `cl::sycl::h_item` object when
    `cl::sycl::handler::parallel_for` method with flexible range is used
    [ab3e71e]
  - Fixed host version of `cl::sycl::mul_hi` built-in to correctly handle
    negative arguments [8a3b7d9]
  - Fix host memory deallocation size of SYCL memory objects [866d634]
  - Fixed bug preventing from passing structure containing accessor to a kernel
    on some devices [1d72965]
  - Fixed bug preventing using types from "inline" namespace as kernel names
    [28d5931]
  - Fixed bug when placeholder accessor behaved like a host accessor fetching
    memory to be available on the host and blocking further operations on the
    accessed memory object [d8505ad]
  - Rectified precision issue with the float to half conversion [2de1379]
  - Fixed `cl::sycl::buffer::reinterpret` method which was working incorrectly
    with sub-buffers [7b2f630] [916c32d] [60b6e3f]
  - Fixed problem with allocating USM memory on the host [01869a0]
  - Fixed compilation issues of built-in functions. [6bcf548]

## Known issues
- [new] The addition of the static keyword on an array in the presence of Intel
  FPGA memory attributes results in the empty kernel after translation.
- [new] A loop's attribute in device code may be lost during compilation.
- [new] Linkage errors with the following message:
  `error LNK2005: "bool const std::_Is_integral<bool>" (??$_Is_integral@_N@std@@3_NB) already defined`
  can happen when a SYCL application is built using MS Visual Studio 2019
  version below 16.3.0.

## Prerequisites
### Linux
- Experimental Intel(R) CPU Runtime for OpenCL(TM) Applications with SYCL
  support version
  [2019.9.11.0.1106_rel](https://github.com/intel/llvm/releases/download/2019-10/oclcpuexp-2019.9.11.0.1106_rel.tar.gz)
  is recommended OpenCL CPU RT prerequisite for the SYCL compiler
- The Intel(R) Graphics Compute Runtime for OpenCL(TM) version
  [19.43.14583](https://github.com/intel/compute-runtime/releases/tag/19.43.14583)
  is recommended OpenCL GPU RT prerequisite for the SYCL compiler.
### Windows
- Experimental Intel(R) CPU Runtime for OpenCL(TM) Applications with SYCL
  support version
  [2019.9.11.0.1106_rel](https://github.com/intel/llvm/releases/download/2019-10/win-oclcpuexp-2019.9.11.0.1106_rel.zip)
  is recommended OpenCL CPU RT prerequisite for the SYCL compiler
- The Intel(R) Graphics Compute Runtime for OpenCL(TM) version
  [100.7372](https://downloadmirror.intel.com/29127/a08/1910.1007372.exe)
  is recommended OpenCL GPU RT prerequisite for the SYCL compiler.

Please, see the runtime installation guide [here](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedWithSYCLCompiler.md#install-low-level-runtime)


# September'19 release notes

Release notes for commit d4efd2ae3a708fc995e61b7da9c7419dac900372

## New features
- Added support for `reqd_work_group_size` attribute. [68578d7]
- SYCL task graph can now be printed to file in json format.
  See [SYCL ENV VARIABLES](doc/SYCLEnvironmentVariables.md) for information how
  to enable it. [c615566]
- Added support for
  [`cl::sycl::intel::fpga_reg`](doc/extensions/IntelFPGA/FPGAReg.md) and
  [`cl::sycl::intel::fpga_selector`](doc/extensions/IntelFPGA/FPGASelector.md)
  extensions. [e438d2b]

## Improvements
- `clCreateCommandQueue` or `clCreateCommandQueueWithProperties` is used
  depending on version of OpenCL implementation. [3511e3d]
- Added support querying kernel for `compile_sub_group_size`. [bb7cb34]
  This resolves intel/llvm#367
- If device image is in SPIRV format and SPIRV is not supported by OpenCL
  implementation exception is thrown. [09e328f]
- Added support for USM pointer to `cl::sycl::handler::set_arg` method.
  [df410a5]
- Added support for `-fsycl-help=arg` compiler option which can be used to emit
  help message from corresponding offline compiler. [0e44dd2]
- Added `-reuse-exe` compiler option which can be used to avoid recompilaton of
  device code (SPIRV) if it has not been changed from the previous compilation
  and all options are the same. This work for Intel FPGA AOT compilation.
  [2934fd8]
- SYCL math builtins now work with regular pointers. [24fa42b]
- Made SYCL specific options available when using clang-cl. [f5a7522]
- USM now works on pre-context basis, so now it's possible to have two
  `cl::sycl::context` objects with USM feature enabled. [e339962]
- `"-fms-compatibility"` and `"-fdelayed-template-parsing"` are not passed
  implicitly when `-fsycl` is used on MSVC. [9c3d98d]
- Implemented `cl::sycl::vec::convert` method, host device now supports all
  rounding modes while other devices support automatic rounding mode. [fe3bbf9]
- `cl::sycl::image` now uses internal aligned_allocator instead of standard one.
  [d7380f5]
- `cl::sycl::program` class now throws exception containing build log in case
  of build failure.

## Bug fixes
- Fixed issue that prevented from using lambda functions in kernels. [1b83ae9]
- Fixed crash happening when `cl::sycl::event` methods's are called for manually
  constructed (not obtained from `cl::sycl::queue::submit`) `cl::sycl::event`.
  [90333c3]
- Fixed problem with functions called from inializer list are being considered
  as device code. [071b581]
- Asynchronous error handler is now called in `cl::sycl::queue` destructor as
  the SYCL specification requires. [0182f72]
- Fixed race condition which could happen if multiple host accessors to the same
  buffer are created simultaneously. [0c61c8c]
- Fixed problem preventing from using template type with template parameters as
  a kernel name. [45194f7]
- Fixed bug with offset passed to `cl::sycl::handler::parallel_for` methods was
  ignored in case of host device or when item is an argument of a lambda.
  [0caeeae]
- Fixed crash which happened when `-fsycl` was used with no source file
  provided. [c291fd3]
- Resolved problem with using bool type as a kernel name. [07b4f09]
- Fixed crash which could happen if sycl objects are used during global objects
  destruction. [fff31fa]
- Fixed incorrect behavior of host version of `cl::sycl::abs_diff` function in
  case if `x - y < 0` and `x` and `y` are unsigned types. [35fc029]
- Aligned the type of exception being thrown if no device of requested type is
  available with the SYCL specification. Now it throws
  `cl::sycl::runtime_error`. [9d5faab]
- `cl::sycl::accessor` can now be created from `cl::sycl::buffer` which was
  constructed with non-default allocator. [8535b24]
- Programs that failed to build (JIT compile) are not cached anymore. [d4efd2a]
- Partial initialization of a constant static array now works correctly.
  [4e52d44]
- Fixed casting from derived class to base in case of multiple inheritance.
  [76e223c]
- Fixed compilation of relational operations with pointer. [4541a7f]
- Fixed representation of long long int in `cl::sycl::vec` class on Windows.
  [336cb82]
- `cl::sycl::stream` class has gained support for printing nan and inf values
  [b63a96f] This resolves intel/llvm#500
- Fixed bug that prevented from using aliases as non-type template parameters
  (such as int) and full template specialization in kernel names. [a784071]
- Crash when using stl allocator with `cl::sycl::buffer` and `cl::sycl::image`
  classes is fixed. [d7380f5]

## Prerequisites
### Linux
- Experimental Intel(R) CPU Runtime for OpenCL(TM) Applications with SYCL
  support version
  [2019.8.8.0.0822_rel](https://github.com/intel/llvm/releases/download/2019-09/oclcpuexp-2019.8.8.0.0822_rel.tar.gz)
  is recommended OpenCL CPU RT prerequisite for the SYCL compiler
- The Intel(R) Graphics Compute Runtime for OpenCL(TM) version
  [19.34.13890](https://github.com/intel/compute-runtime/releases/tag/19.34.13890)
  is recommended OpenCL GPU RT prerequisite for the SYCL compiler.
### Windows
- Experimental Intel(R) CPU Runtime for OpenCL(TM) Applications with SYCL
  support version
  [2019.9.9.0.0901](https://github.com/intel/llvm/releases/download/2019-09/win-oclcpuexp-2019.9.9.0.0901.zip)
  is recommended OpenCL CPU RT prerequisite for the SYCL compiler
- The Intel(R) Graphics Compute Runtime for OpenCL(TM) version
  [100.7158](https://downloadmirror.intel.com/29058/a08/igfx_win10_100.7158.exe)
  is recommended OpenCL GPU RT prerequisite for the SYCL compiler.

Please, see the runtime installation guide [here](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedWithSYCLCompiler.md#install-low-level-runtime)

# August'19 release notes

Release notes for commit c557eb740d55e828fcf74b28d2b686c928e45318.

## New features
- Support for `image accessor` has been landed.
- Added support for unnamed lambda kernels, so `parallel_for` works without
  specifying "Kernel Name" type. This can be enabled by passing
  `-fsycl-unnamed-lambda` option.
- Kernel to kernel blocking and non-blocking pipe feature is implemented.
- Added support for Unified Shared Memory ([USM](doc/extensions/USM)).

## Improvements
- Now OpenCL 1.2 clCreateSampler sampler is used for all version of OpenCL
  implementation.
- Added Intel FPGA specific command line interfaces for ahead of time
  compilation.
- Intel FPGA memory attributes are now supported for static variables.
- Hierarchical parallelism is improved to pass conformance tests.
- `private_memory` class has been implemented.
- sycl.lib is automatically linked with -fsycl switch on Windows.
- Added support for Windows getOSModuleHandle.
- Functions with variadic arguments doesn't trigger compiler diagnostic about
  calling convention.
- Added experimental support for building and linking SYCL runtime with libc++
  library instead of libstdc++.
- Adjusted array and vec classes so they are more efficient for Intel FPGA
  devices.
- Exception will be thrown on attempt to create image accessor for the device
  which doesn't support images.
- Use `llvm-objcopy` for merging device and host objects instead of doing
  partial linking.
- Check if online compiler is available before building the program.
- clCreateProgramWithILKHR is used if OpenCL implementation supports
  cl_khr_il_program extension.
- Reuse the pointer provided by the user in the `buffer` constructor (even if
  `use_host_ptr` property isn't specified) if its alignment is sufficient.
- `-Ldir` now can be used to find libraries with `-foffload-static-lib`
- `max_concurrency` Intel FPGA loop attribute now accepts zero.
- Ignore incorrectly used Intel FPGA loop attribute emitting user friendly
  warning instead of compiler error.
- Added `depends_on` methods of `handler` class which can be used to provide
  additional dependency for a command group.
- SYCL implementation can now be built on Windows using Visual Studio 2017 or
  higher.

## Bug fixes
- Fixed assertion failure when use `-save-temps` option along with `-fsycl`.
- Cached JITed programs are now released during destruction of `context` they
  are bound to, earlier release happened during a process shutdown.
- Fixed `get_linear_id` method of `group` class, now it calculate according to
  row-major ordering.
- Removed printing of error messages to stderr before throwing an exception.
- Explicit copy API of the handler class is asynchronous again.
- `fill` method of `handler` class now takes element size into account.
- Fixed bug with ignored several Intel FPGA loop attributes in case of argument
  is 1.
- Fixed bug which prevented Intel FPGA loop attributes work with infinite loops.
- Fixed problem which caused invalid/redundant memory copy operations to be
  generated.
- The commands created by Scheduler now cleaned up on destruction of
  corresponding `SYCL` memory objects (`buffer`, `image`).
- 1 dimensional sub buffer is passed as cl_mem obtained by calling
  clCreateSubBuffer when kernel is built from OpenCL C source.
- Now copy of entire memory is performed when data is needed in new context even
  if user requests accesses to only part of the memory.
- Fixed problem with one element `vec` objects relation/logical operation which
  was working like scalar.
- Type casting and conditional operator (ternary 'if') with pointers now working
  correctly.
- The problem with calling inlined kernel from multiple TUs is fixed.
- Fixed compiler warnings for Intel FPGA attributes on host compilation.
- Fixed bug with passing values of `vec<#, half>` type to the kernel.
- Fixed buffer constructor which takes host data as shared_ptr. Now it
  increments shared_ptr reference counter and reuses provided memory if
  possible.
- Fixed a bug with nd_item.barrier not respecting fence_space flag

## Prerequisites
- Experimental Intel(R) CPU Runtime for OpenCL(TM) Applications with SYCL
  support version [2019.8.7.0.0725_rel](https://github.com/intel/llvm/releases/download/2019-08/oclcpuexp-2019.8.7.0.0725_rel.tar.gz)
  is recommended OpenCL CPU RT prerequisite for the SYCL compiler
- The Intel(R) Graphics Compute Runtime for OpenCL(TM) version [19.29.13530](https://github.com/intel/compute-runtime/releases/tag/19.29.13530) is
  recommended OpenCL GPU RT prerequisite for the SYCL compiler.

# July'19 release notes

Release notes for commit 64c0262c0f0b9e1b7b2e2dcef57542a3fe3bdb97.

## New features
 - `cl::sycl::stream` class support has been added.
 - New attributes for Intel FPGA device are added: `merge`, `max_replicates`
   and `simple_dual_port`.
 - Initial support for new Plugin Interface (PI) layer is added to SYCL runtime
   library. This feature simplifies porting SYCL implementation to non-OpenCL
   APIs.
 - New address space handling rules are implemented in the SYCL device
   compiler. Raw pointers are allocated in generic address space by default and
   address space inference is supposed to be done by LLVM pass.  Old compiler
   behavior can be recovered by enabling `DISABLE_INFER_AS` environment
   variable.
 - Add basic implementation of hierarchical parallelism API.
 - Add new clang built-in function `__unique_stable_name`. SYCL compiler may
   use this built-in function to auto-generate SYCL kernel name for lambdas.

## Improvements
 - SYCL integration header is excluded from the dependency list.
 - Raw pointers capturing added to the SYCL device front-end compiler. This
   capability is required for Unified Shared Memory feature implementation.
 - SYCL device compiler enabled support for OpenCL types like event, sampler,
   images to simplify compilation of the SYCL code to SPIR-V format.
   `CXXReflower` pass used to make "SPIR-V friendly LLVM IR" has been removed.
 - Intel FPGA loop attributes were renamed to avoid potential name conflicts.
 - Old scheduler has been removed.
 - `sampler` type support is added to the `set_arg` methods.
 - Internal SYCL device compiler design documentation was improved and updated.
   Development process documentation has been updated with more details.
 - Initial support for `image` class (w/o accessor support).
 - Static variables are allocated in global address space now.
 - Made sub-group methods constant to enable more use cases.
 - Added `-fsycl-link` option to generate fat object "linkable" as regular host
   object.
 - Enable `set_final_data` method with `shared_ptr` parameter.
 - Enable using of the copy method with `shared_ptr` with `const T`.

## Bug fixes
 - Fixed argument size calculation for zero-dimensional accessor.
 - Removed incorrect source correlation from kernel instructions leading to
   incorrect profiling and debug information.
 - A number of issues were fixed breaking build of the compiler on Windows
 - Fixed unaligned access in load and store methods of the vector class.
 - `global_mem_cache_type` values were aligned with the latest revision of the
   SYCL specification.
 - Stubs for C++ standard headers were removed. This should fix compilation of
   <iostream> and <algorithm> with SYCL device compiler.
 - Unscoped enums were removed from global namespace to avoid conflicts with
   user defined symbols.
 - Explicit copy API of the handler class is blocking i.e. data is
   copied once the command group has completed execution.
 - Renamed `cl::sycl::group::get_linear` to `cl::sycl::group::get_linear_id`.
 - SYCL kernel constructor from OpenCL handle now retains OpenCL object during
   SYCL object lifetime.
 - Fixed forward declaration compilation inside a SYCL kernel.
 - Fixed code generation for 3-element boolean vectors.

## Prerequisites
 - Experimental Intel(R) CPU Runtime for OpenCL(TM) Applications with SYCL
   support is available now and recommended OpenCL CPU RT prerequisite for the
   SYCL compiler.
 - The Intel(R) Graphics Compute Runtime for OpenCL(TM) version 19.25.13237 is
   recommended OpenCL GPU RT prerequisite for the SYCL compiler.

## Known issues
 - New address space handling approach might degrade compilation time
   (especially for GPU device).
 - Some tests might fail on CPU device with the [first experimental CPU
   runtime](https://github.com/intel/llvm/tree/expoclcpu-1.0.0) due to new
   address space handling by the SYCL compiler. The workaround for this kind of
   issues while we wait for CPU runtime update is to set `DISABLE_INFER_AS`
   environment variable during compilation. See
   https://github.com/intel/llvm/issues/277 for more details.

# June'19 release notes

The release notes contain information about changes that were done after
previous release notes and up to commit
d404d1c6767524c21b9c5d05f11b89510abc0ab9.

## New Features
- New FPGA loop attributes supported:
    - `intelfpga::ivdep(Safelen)`
    - `intelfpga::ii(Interval)`
    - `intelfpga::max_concurrency(NThreads)`

## Improvements
- The new scheduler is implemented with the following improvements:
    - optimize host memory allocation for `buffer` objects.
    - optimize data transfer between host and device by using Map/Unmap instead
      of Read/Write for 1D accessors.
    - simultaneous read from a buffer is allowed: execution of two kernels
      reading from the same buffer are not serialized anymore.
- Memory attribute `intelfpga::max_concurrency` was renamed to
  `intelfpga::max_private_copies` to avoid name conflict with fresh added loop
  attribute
- Added support for const values and local accessors in `handler::set_arg`
  method.

## Bug Fixes
- The new scheduler is implemented with the following bug fixes:
    - host accessor now blocks subsequent operations(except RAR) with the buffer
      it provides accesses to until it is destroyed.
    - OpenCL buffers now released on the buffer destruction.
- `accessor::operator[]` like methods now take into account offset.
- Non-SYCL compilation(without `-fsycl`) was fixed, such application should work
  on host device, but fail on OpenCL devices.
- Several warnings were cleaned up.
- buffer constructor was fixed to support const type as template parameter.
- `event::get_profiling_info` now waits for event to be completed.
- Removed non-const overload of `item::operator[]` as it's not present in SYCL
  specification.
- Compiling multiple objects when using `-fsycl-link-targets` now creates proper
  final .spv binary.
- Fixed bug with crash in sampler destructor when sampler object is created
  using enumerations.
- Fixed `handler::set_arg`, so now it works correctly with kernels created using
  program constructor which takes `cl_program` or `program::build_with_source`.
- Now `lgamma_r` builtin works correctly when application is built without
  specifying `-fsycl` option to the compiler.

## Prerequisites
- Experimental Intel® CPU Runtime for OpenCL™ Applications with SYCL support is
  available now and recommended OpenCL CPU RT prerequisite for the SYCL
  compiler.
- The Intel(R) Graphics Compute Runtime for OpenCL(TM) version 19.21.13045 is
  recomended OpenCL GPU RT prerequisite for the SYCL compiler.

## Known issues
- Performance regressions can happen due to additional math for calculation of
  offset that were added to the `accessor::operator[]` like methods.
- Applications can hang at exit when running on OpenCL CPU device because some
  OpenCL handles allocated inside SYCL(e.g. `cl_command_queue`) are not
  released.

# May'19 release notes

## New Features
- Added support for half type.
- Implemented sampler class.

## Improvements
- Implemented several methods of buffer class:
    - buffer::has_property and buffer::get_property
    - buffer::get_access with range and offset classes
    - buffer::get_allocator as well as overall support for custom allocators.
- Implemented broadcasting vec::operator=.
- Added support for creating a sub-buffer from a SYCL buffer.
- Added diagnostic about capturing class static variable in kernel code.
- Added support for discard_write access::mode.
- Now SYCL buffer allocates 64 bytes aligned memory.
- Added support for case when object of accessor class is wrapped by some class.
- Added support for const void specialization of multi_ptr class.
- Implemented the following groups of SYCL built-in functions:
    - integers
    - geometric
- Support for variadic templates in SYCL kernel names.

## Bug Fixes
- Disabled several methods of buffer class that were available for incompatible
  number of dimensions.
- Added initialization of range field in buffer interoperability constructor.
- Fixed buffer constructor with iterators, now the data is written back to the
  input iterator if it's not const iterator.
- Fixed the problem which didn't allow using buffer::set_final_data and creating
  of host accessor from buffer created using interoperability constructor.
- Now program::get_*_options returns options correctly if the program is created
  with interoperability constructor.
- Fixed linking multiple programs compiled using compile_with_kernel_type.
- Fixed initialization of device list in program class interoperability
  constructor.
- Aligned vec class with the SYCL Specification, changed argument to multi_ptr
  with const type.
- Fixed queue profiling device information query to work with OpenCL1.2.
