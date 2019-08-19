# August'19 release notes

Release notes for commit c557eb740d55e828fcf74b28d2b686c928e45318.

## New features
- Support for `image accessor` has been landed.
- Added support for unnamed lambda kernels, so `parallel_for` works without
  specifying "Kernel Name" type. This can be enabled by passing
  `-fsycl-unnamed-lambda` option.
- Kernel to kernel blocking and non-blocking pipe feature is implemented.
- Added support for Unified Shared Memory ([USM](doc/extensions/usm)).

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
- Fixed buffer constructor which takes host data as shared_ptr. Now it increments
  shared_ptr reference counter and reuses provided memory if possible.
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
 - Experimental Intel(R) CPU Runtime for OpenCL(TM) Applications with SYCL support is
   available now and recommended OpenCL CPU RT prerequisite for the SYCL
   compiler.
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
- Added support for const values and local accessors in `handler::set_arg` method.

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
- Fixed bug with crash in sampler destructor when sampler object is created using
  enumerations.
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
