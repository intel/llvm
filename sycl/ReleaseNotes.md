# Nov'23 release notes
Release notes for commit range f4e0d3177338..f4ed132f243a

## New Features
### SYCL Compiler
- Added `-fpreview-breaking-changes` option (see ["Upcoming API/ABI breakages"](#upcoming-apiabi-breakages) section for more details) [08febcfcb64a] [99b97eef4278]
- Enabled early AOT abilities when creating objects [6f248088541c]
- Added support for `-faltmathlib` [32c3e642d644]
- Added support for `-ftarget-register-alloc-mode` [8e5275af77af]
- Added `-fsycl-add-default-spec-consts-image` command line option to Clang Driver [3847c7c6c5c7]

### SYCL Library
- Added experimental implementation of SYCL bindless images extension [58a8f2001d8b] [380453deeb51] [b1aab046f408]
- Added support for local accessors to `copy_from`/`copy_to` for ESIMD [474461cb2e1d]
- Added support for local accessors to atomic update for ESIMD [8974cb944b3c]
- Added support for local accessors to lsc for ESIMD [9dce2d227b31]
- Added new `raw_send` APIs moving compile time params to template params for ESIMD [1a5e53216b00]
- Added SYCLcompat library [bedd818eb281] [465aa56ee5b3] [75b31a4c24b4] [4dfaf20884c7] [47d84897a87b] [8c07a840311a] [ab578107b6f8] [ac5a90e0c1b5] [a733fa74b583] [c39c2138f5a0]
- Implemented `sycl_ext_oneapi_auto_local_range` [a6d4c2120b5c]
- Implemented [`sycl_ext_oneapi_sub_group_mask`](https://github.com/intel/llvm/blob/3bd09b9089c5221f72f19ab1adde19da909661a7/sycl/doc/extensions/supported/sycl_ext_oneapi_sub_group_mask.asciidoc) version 2 [3bd09b9089c5]
- Implemented [`sycl_ext_oneapi_queue_priority`](https://github.com/intel/llvm/blob/0c33fea5fac067066cc942ee6f98612700e05d57/sycl/doc/extensions/supported/sycl_ext_oneapi_queue_priority.asciidoc) [0c33fea5fac0]
- Implemented optional kernel feature for `joint_matrix` and `joint_matrix_mad` [946b1f236167] [c38dfcf64681]
- Implemented [`sycl_ext_oneapi_prefetch`](https://github.com/intel/llvm/blob/b5d69df53620a1c9d33f7575cd3e6dd1ade54508/sycl/doc/extensions/proposed/sycl_ext_oneapi_prefetch.asciidoc) [e7139b0f15d0] [0229456e19d5] [b5d69df53620]
- Implemented [`sycl_ext_oneapi_device_architecture`](https://github.com/intel/llvm/blob/1ad69e59eb13245cda5c58b04db0bb79f35a597b/sycl/doc/extensions/experimental/sycl_ext_oneapi_device_architecture.asciidoc) for HIP and CUDA devices [1ad69e59eb13]
- Improved error message when running ESIMD code on unsupported devices [a5f471ea1535]
- Added [`ext_oneapi_non_uniform_groups`](https://github.com/intel/llvm/blob/de92299c2c09d626dcbc633c83e259d828220d03/sycl/doc/extensions/proposed/sycl_ext_oneapi_non_uniform_groups.asciidoc) aspect [de92299c2c09]

### Documentation
- Moved [`sycl_ext_oneapi_complex`](https://github.com/intel/llvm/blob/ea6aea37c6dd777ef8df68e45a93646be2ad9558/sycl/doc/extensions/experimental/sycl_ext_oneapi_complex.asciidoc) from proposed to experimental [ea6aea37c6dd]
- Added [SYCLcompat documentation](https://github.com/intel/llvm/blob/180a92ad707bd35df9e98c1474dc52a1e9b3dead/sycl/doc/syclcompat/README.md) [180a92ad707b]
- Added [`sycl_ext_oneapi_forward_progress`](https://github.com/intel/llvm/blob/67461148b16938a318796f639ee857b02831c38d/sycl/doc/extensions/proposed/sycl_ext_oneapi_forward_progress.asciidoc) extension spec [67461148b169]
- Added [`sycl_ext_oneapi_launch_queries`](https://github.com/intel/llvm/blob/67461148b16938a318796f639ee857b02831c38d/sycl/doc/extensions/proposed/sycl_ext_oneapi_launch_queries.asciidoc) extension spec [67461148b169]
- Added [`sycl_ext_oneapi_kernel_compiler`](https://github.com/intel/llvm/blob/5e64f96d20c22a6c2a1600515160ddf248637641/sycl/doc/extensions/proposed/sycl_ext_oneapi_kernel_compiler.asciidoc) extension spec [5e64f96d20c2]
- Added [`sycl_ext_oneapi_kernel_compiler_opencl`](https://github.com/intel/llvm/blob/5e64f96d20c22a6c2a1600515160ddf248637641/sycl/doc/extensions/proposed/sycl_ext_oneapi_kernel_compiler_opencl.asciidoc) extension spec [5e64f96d20c2]
- Added [`sycl_ext_oneapi_enqueue_functions`](https://github.com/intel/llvm/blob/ebc3ddb9a60bad6f1f0e331da4044000e3a7fb60/sycl/doc/extensions/proposed/sycl_ext_oneapi_enqueue_functions.asciidoc) extension spec [ebc3ddb9a60b]
- Added [`sycl_ext_oneapi_graph_fusion`](https://github.com/intel/llvm/blob/8c4cad45faba815ad252de895a2a1434d8c0bd1a/sycl/doc/extensions/proposed/sycl_ext_oneapi_graph_fusion.asciidoc) extension spec [8c4cad45faba]
- Added [`sycl_ext_intel_fpga_datapath`](https://github.com/intel/llvm/blob/a162179381fd1b0ebab5cab8978e7cef9dfdac8c/sycl/doc/extensions/proposed/sycl_ext_intel_fpga_datapath.asciidoc) extension spec [a162179381fd]
- Added [`sycl_ext_intel_fpga_mem`](https://github.com/intel/llvm/blob/a162179381fd1b0ebab5cab8978e7cef9dfdac8c/sycl/doc/extensions/proposed/sycl_ext_intel_fpga_mem.asciidoc) extension spec [a162179381fd]
- Added [`sycl_ext_intel_cache_controls`](https://github.com/intel/llvm/blob/deb020bf740b1d34d71189f5900b1558934a09d7/sycl/doc/extensions/proposed/sycl_ext_intel_cache_controls.asciidoc) extension spec [deb020bf740b]
- Added [design doc](https://github.com/intel/llvm/blob/1713ed19f71c4401b2ff62dcbc32a8e71b336701/sycl/doc/design/CommandGraph.md) for `sycl_ext_oneapi_graph` extension [1713ed19f71c]

## Improvements
### SYCL Compiler
- Added diagnostics if SYCL-specific command line options are used without `-fsycl` [bf128c873e74]
- Enabled `INTEL_SYCL_OPAQUEPOINTER_READY` [a47705f2d865] [9dded21c5984]
- Supported multiple SYCL targets with `native_cpu` in the same compiler invocation [aab5d747c46e]
- Made `-fintelfpga` flag imply `-fsycl`. [ad2f959d8b88] [6d48bc13c456]
- Ensured debug format for SYCL device on Windows is the expected format [73bba79a3ffb]
- Disabled parallel for range rounding at `-O0` [2c117d7a6104]
- Fixed missing `optnone` attribute in `-O0` mode on non-FPGA target to improve debuggability [993fe5b31b43]

### SYCL Library
- Supported missing types for the `ldg` in the [sycl_ext_oneapi_cuda_tex_cache_read](https://github.com/intel/llvm/blob/aec8a35788c1953c6ae74ab1e6eb924fa09d5e4e/sycl/doc/extensions/experimental/sycl_ext_oneapi_cuda_tex_cache_read.asciidoc) [aec8a35788c1]
- Stopped throwing exception when passing empty accessor to `handler::require()` [7bf1f57bc26e]
- Moved `addc()` and `subb()` functions out of experimental namespace [9e3852bc1c2e]
- Moved BFN function from experimental namespace [c9da5c7ff7e8]
- Improved error message when `queue::memcpy` receives a null pointer by reporting source code info(file/line number) [626c4b5078c1]
- Improved accuracy of double `tanpi` host implementation [041a74c0cbda]
- Added support bitwise and/or/xor and logical and/or subgroup collective functions for AMD[288aeaef03a9]
- Updated `annotated_ptr` API to match spec changes [954730e7a823] [fbad42a398ef] [43336a15fcaf]
- Implemented info::event_profiling::command_submit support for platforms that use OpenCL version less than 2.1 [7cfa951303eb]
- Fixed required alignment in buffers [093dae1f7214]
- Improved error message when using `sycl_ext_oneapi_graph` features that are not yet implemented [8d8d3f4f52b0]
- Supported `get_multi_ptr` in ESIMD context [2f0f3814c541]
- Accept integral type in parallel_for with offsets [ecfc49c825c6]
- Added support for `assert()` in ESIMD kernels [545a043ebaf5]
- Added missing Intel archs to [`sycl_ext_oneapi_device_architecture`](https://github.com/intel/llvm/blob/c4db251179dcd1b30a1043d4780634dbe39b3e3a/sycl/doc/extensions/experimental/sycl_ext_oneapi_device_architecture.asciidoc) [c4db251179dc]
- Enabled range rounding for unnamed lambdas [4f30e6648d9e]
- Fixed loading of plugins when path to them contains non-ANSI symbols [5c30815d4821]
- Improved the error message when the command group function object passed to `queue::submit` has the wrong signature [5801970585b2]
- Improved `handler::fill` performance on 2D and 3D buffers [0249a55f810e]
- Added defaults for streaming and register_map fpga_kernel_properties [23a6f389c1e4]
- Moved GRF property conflict check to compile time [4fb92d35c7be]
- Added error when using invoke_simd with callables with struct arguments [03c13fd53a24]
- Added missing `rintf` for CUDA [3c327c736ced]
- Optimized kernel launch on CUDA by caching the max local mem size so that CUDA driver is called less [1c892f952cae]
- Made `sycl_ext_oneapi_graph` functions thread-safe [c8c64a673eeb]
- Improved diagnostic for `nd_item` passed to `parallel_for` with range [e600d7922489]
- Implemented missing exceptions defined by `sycl_ext_oneapi_graph` specification [77b794bcaad6]
- Implemented supported `make_device`, `make_queue`, `make_event` interop functions for HIP backend [5e9d07b14a47]
- Added missing `nearbyint` for CUDA [0ef26d3ee808]
- Implemented runtime query for matrix combinations [a09491f4e389]
- Supported basic kernel launches over 64-bit global ranges [192c3016e76b]
- Clarified runtime error for nested kernel submissions [7afc2d03fd00]
- Fix compiler crash when spec constants are structs with padding in the middle [292309c826a3]
- Added support for CUDA backend with `sycl_ext_oneapi_graph` [367b662a49ba]
- Added support for specialization constants in `sycl_ext_oneapi_graph` nodes [fca7f5e2e3fb]
- Improved `vec::convert` implementation to use a single instruction instead of per-element [f634cc9bf799]
- Made `annotated_arg<T>` implicitly device copyable if `T` is device copyable [9f65e61a2dbf]

### Documentation
- Updated [`sycl_ext_oneapi_local_memory`](https://github.com/intel/llvm/blob/457251feceb81845de962383f227e14522d4cf96/sycl/doc/extensions/supported/sycl_ext_oneapi_local_memory.asciidoc) [457251feceb8]
- Added get-coord API and general query example to [`sycl_ext_intel_matrix`](https://github.com/intel/llvm/blob/579f1dec0b5b2ecd177dfb95c3d1522f57079cc6/sycl/doc/extensions/experimental/sycl_ext_matrix/sycl_ext_intel_matrix.asciidoc) [579f1dec0b5b] [38ac2124c1e8]
- Updated design doc for new offloading model [115808f5128f]
- Updated description of the default value of the environment variable `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS` to reflect actual behavior [ad973f3383e2]
- Added overloads with `sycl::queue` for some functions in [`sycl_ext_oneapi_bindless_images`](https://github.com/intel/llvm/blob/4d0c552985bd0e4fc16740a0fc89393ef7fe3d04/sycl/doc/extensions/experimental/sycl_ext_oneapi_bindless_images.asciidoc) [4d0c552985bd]
- Specify an error in `sycl_ext_oneapi_graph` if an executable graph is submitted to a queue with a different device or context [2c53815c6c1e]

## Bug Fixes
### SYCL Compiler
- Fixed backend option passing issues with multiple targets [12ca2db0e30b]
- Fixed `-fgpu-rdc` option for CUDA [f7595ac7527e]
- Fixed triple in offload mismatch warning [0fd9a4eb17a2]
- Fixed a compilation issue when `-enable-global-offset=false` for AMDGU and NVPTX targets [00cf4c29740b]

### SYCL Library
- Fixed stack corruption in `sycl::modf` for `sycl::half` type [7c6541dc18e3]
- Fixed `nextafter(half)` precision on host [1f10f3588e8c]
- Fixed identity values for (sub)group collectives min/max for AMD and CUDA [248389371932]
- Corrected exception thrown when kernel is submitted with a large (unsupported) `reqd_work_group_size` attribute [25c3666dffa9]
- Implemented hip `prefetch`/`mem_advise` (noop) for rocm < 5.0 [7e4cc1113987]
- Adjusted `range` definition so it has a default constructor in accordance with the SYCL spec [0289d2ac94f9]
- Added missing return statement in the `multi_ptr<void..>` conversion operator [836469ba88e2]
- Optimized kernel fusion implementation so that some resources can be released earlier [a33b0a950ae6]
- Fixed an integer overflow that happened when sub-buffer size exceeds `INT_MAX` [2fb0b140ad3b]
- Fixed `std::enable_if` typo for `marray` and `multi_ptr` [0ef25ec3bdf8]
- Fixed an issue with `reqd_work_group_size` attribute values bigger than `INT_MAX` being incorrectly handled [cee07d3d2652]
- Fixed UB in group sort scratch memory access [ed4c01301a7c]
- Use primary context in HIP adapter [d1c92cb95b29]
- Fixed a crash when default-constructed accessor is passed to a kernel [fb8ddbf7b0ea]
- Stopped `sycl_ext_oneapi_graph` subgraph nodes from creating incorrect dependencies on subsequent usages of the subgraph [92ddf8d562d7]
- Fixed ambiguous `log2` call in host code [12fa0af28e88]

### Documentation
- Updated `device_image_scope` property  in `sycl_ext_oneapi_device_global` [be8e031cd078]

## API/ABI breakages
- Deprecated experimental functions for ESIMD: `quot`, `div`, `mod`, `dp*`, `line`, `lrp` [76976a22ba2e]
- Removed ESIMD Emulator plugin [1ece6da6b113]
- Deprecated `sycl::abs(genfloat)` [d15d4409b2cf]

## Upcoming API/ABI breakages
The following changes ared only in effect if the `-fpreview-breaking-changes` flag is set.
- Fixed sycl::vec ABI inconsistencies and made it trivially copyable [531aabfdee91] [f4ed132f243a] 
- Updated `sycl.hpp` to not include `<complex>` and `<cmath>` [6ed0ab887d18] [d22c9c58d2e7]

## Known Issues
- Having MESA OpenCL implementation which provides no devices on a
  system may cause incorrect device discovery. As a workaround such an OpenCL
  implementation can be disabled by removing `/etc/OpenCL/vendor/mesa.icd`.
- `-fsycl-dead-args-optimization` can't help eliminate offset of
  accessor even though it's created with no offset specified
- SYCL 2020 barriers show worse performance than SYCL 1.2.1 do. [18c80faa]
- When using fallback assert in separate compilation flow it requires explicit
  linking against `lib/libsycl-fallback-cassert.o` or
  `lib/libsycl-fallback-cassert.spv`
- Limit alignment of allocation requests at 64KB which is the only alignment
  supported by Level Zero. 7dfaf3bd
- User-defined functions with the name and signature matching those of any
  OpenCL C built-in function (i.e. an exact match of arguments, return type
  doesn't matter) can lead to Undefined Behavior.
- A DPC++ system that has FPGAs installed does not support multi-process
  execution. Creating a context opens the device associated with the context
  and places a lock on it for that process. No other process may use that
  device. Some queries about the device through `device.get_info<>()` also
  open up the device and lock it to that process since the runtime needs
  to query the actual device to obtain that information.
- The format of the object files produced by the compiler can change between
  versions. The workaround is to rebuild the application.
- Using `sycl::kernel_bundle` API to refer to a kernel defined
  in another translation unit leads to undefined behavior.
- Linkage errors with the following message:
  `error LNK2005: "bool const std::_Is_integral<bool>" (??$_Is_integral@_N@std@@3_NB) already defined`
  can happen when a SYCL application is built using MS Visual Studio 2019
  version below 16.3.0 and user specifies `-std=c++14` or `/std:c++14`.
- Printing internal defines isn't supported on Windows. [50628db1]
- The support of accessor and local_accessor for ESIMD is still limited comparing to SYCL.
- [`sycl_ext_oneapi_root_group`](https://github.com/intel/llvm/tree/743c35be2da7/sycl/doc/extensions/proposed/sycl_ext_oneapi_root_group.asciidoc) implementation has the following limitations: `max_num_work_group_sync` query always returns `1` and the implemented barrier has work group scope since all work items in a root group are currently in the same work group.


# Oct'23 release notes

Release notes for commit range [`cb91c232c661..f4e0d3177338`](https://github.com/intel/llvm/compare/cb91c232c661..f4e0d3177338)

## New features

### SYCL Compiler
- Enabled Pre-Compiled header (PCH) inclusion while performing host compilation with `-fsycl`. [7f12b287236f]
- Added support for `-ftarget-export-symbols` option which allows to export symbols for AOT. [c33ba5dafe91]
- Added support for `reqd_sub_group_size` to `sycl::is_compatible` and implemented device code split based on `reqd-sub-group-size` optional kernel feature. [841768794f7a]
- Added support for `-ftarget-compile-fast` for spir64_gen and JIT mode. [b6b01b48cfd9][92855bea87f4]
- Implemented adding C++ libraries to the linker with `-fsycl`. [da5e1b98f1c3]
- Added support for multiple call operators in a kernel functor. [84cc1e1eff04]
- Added support to propagate compile flags to device backend compiler.[f45fb51ba6cf]
- Added new FPGA loop attribute `enable_loop_pipelining`. [4147b8ded75f]

### SYCL Library
- Enabled [`sycl_ext_oneapi_annotated_arg`](https://github.com/intel/llvm/tree/80b7de1d63f4/sycl/doc/extensions/experimental/sycl_ext_oneapi_annotated_arg.asciidoc) and [`sycl_ext_oneapi_annotated_ptr`](https://github.com/intel/llvm/tree/80b7de1d63f4/sycl/doc/extensions/experimental/sycl_ext_oneapi_annotated_ptr.asciidoc) experimental extensions. [05d3be63f996][80b7de1d63f4][557204abf242]
- Implemented [`sycl_ext_intel_queue_immediate_command_list`](https://github.com/intel/llvm/tree/2f2560fe74e6/sycl/doc/extensions/supported/sycl_ext_intel_queue_immediate_command_list.asciidoc) extension. [2f2560fe74e6]
- Implemented [`sycl_ext_oneapi_copy_optimize`](https://github.com/intel/llvm/tree/65cc0cfe809f/sycl/doc/extensions/experimental/sycl_ext_oneapi_copy_optimize.asciidoc) experimental extension. [65cc0cfe809f]
- Added initial implementation of the experimental SYCL Graph extension [`sycl_ext_oneapi_graph`](https://github.com/intel/llvm/tree/f9d4830babca/sycl/doc/extensions/proposed/sycl_ext_oneapi_graph.asciidoc)  [35465da4824a][f9d4830babca][72341ee375d6][96a605095665]
- Added `dimensions` member to item/range-like types. [9c7bd9ea046f]
- Added support of `queue::priority_*` properties for OpenCL backend. [25fd689999f7]
- Implemented initial version of the SYCL Native CPU Plug-in designed in [`SYCLNativeCPU`](https://github.com/intel/llvm/tree/5b501eeca963/sycl/doc/design/SYCLNativeCPU.md). [b2d0837e9644] [5b501eeca963]
- Added `__imf_max/min/hadd/fast_*` functions to imf device libraries. [fe01366e8b08][d96e5075303a]
- Introduced and implemented new `sycl::ext::oneapi::experimental::info::device::architecture` device descriptor and `device::ext_oneapi_architecture_is(ext::oneapi::experimental::architecture)` host API as part of [`sycl_ext_oneapi_device_architecture`](https://github.com/intel/llvm/tree/d0b01b265d92/sycl/doc/extensions/experimental/sycl_ext_oneapi_device_architecture.asciidoc) extension (for Level Zero and OpenCL). [d0b01b265d92]
- Added experimental implementation of [`sycl_ext_intel_grf_size`](https://github.com/intel/llvm/tree/370aa2a01711/sycl/doc/extensions/experimental/sycl_ext_intel_grf_size.asciidoc) [370aa2a01711]
- Experimental [`sycl_ext_oneapi_device_global`](https://github.com/intel/llvm/tree/d812d1e13ab5/sycl/doc/extensions/experimental/sycl_ext_oneapi_device_global.asciidoc) extension is supported now. [d812d1e13ab5]
- Enabled media API that works with image accessors for ESIMD. Those accessors do not depend on stateful/stateless mode. [b95b2d88e446]
- Allowed implicit conversion from `std::experimental::simd_mask` to `ESIMD::simd_mask`. [de5b479a2de9]
- Added `host_task` image accessor support. [b6faefb08e23]
- Started to support 64-bit offsets with accessors in stateless mode for ESIMD APIs. [f28ae00bf6a3][d04ebb03c1c8][62955fc20e14]
- Added `__imf_llabs` to the imf device libraries. [3c19581f828c]
- Added explicit conversion of `multi_ptr<T>` to `multi_ptr<const T>`. [10d2f5b613f3]
- Added support for scalar logical operators with group algorithms. [11ac73003056][90406b2ba07c]
- Implemented device query for 64 bit atomic support in ESIMD emulator. [c40baa6db64b]
- Added support of 16 bit data for `lsc_atomic_update` and `lsc_slm_atomic_update` ESIMD API. [3028d82a75d2]
- Added initial implementation of [`sycl_ext_oneapi_root_group`](https://github.com/intel/llvm/tree/743c35be2da7/sycl/doc/extensions/proposed/sycl_ext_oneapi_root_group.asciidoc) extension. There are severe limitations: `max_num_work_group_sync` query always returns `1` and the implemented barrier has work group scope since all work items in a root group are currently in the same work group. [743c35be2da7]
- Added support for tf32 type using the unified interface for SYCL Matrix extension. [aba6d85f9f16]
- Implemented Host Pipes described in [`sycl_ext_intel_dataflow_pipes`](https://github.com/intel/llvm/tree/992ef064289f/sycl/doc/extensions/supported/sycl_ext_intel_dataflow_pipes.asciidoc) extension. [992ef064289f][5bd42eb6a7df]
- Added support for more math built-ins for `bfloat16` together with the extension specification [`sycl_ext_oneapi_bfloat16_math_functions`](https://github.com/intel/llvm/tree/c7759bb8cd0b/sycl/doc/extensions/experimental/sycl_ext_oneapi_bfloat16_math_functions.asciidoc) update [c7759bb8cd0b]
- Added decorated `async_work_group_copy` overloads. [d1f67ca228f2]
- Added initial implementation of the Unified Runtime plugin and routed to it with `SYCL_PREFER_UR`. [5a239c6a08e1]
- Added support for accessors to `atomic_update` ESIMD API. [59a1aa0841da]
- Enabled pretty-printing of reference objects by GDB. [e3fc16f56c0e]
- Added Xmethods that help GDB to support reading local_accessors on GPU from SLM. [ddfc5220afaa]
- Enabled passing local accessors to ESIMD kernel and via `invoke_simd()` API, enabled usage of `get_pointer()` and `operator[]` for accessors in ESIMD kernel. The support of accessor and local_accessor is still limited comparing to SYCL. [21ca00fb1cc5][82ae85d12bf0]
- Implement unpadding for 2d block load/store ESIMD API. [9be8a306d0eb]
- Added SYCL 2020 image classes. [9035cdb0898d][1a3e99307e4f]
- Added interfaces for SYCL 2020 image accessors, only the host accessors interfaces are actually functional. [a58d8e3e88c0]
- Added XPTI notifications for SYCL 2020 images as well as for the associated accessor classes. [8c535276785d][73d5c04d0305]
- Started to print device aspects in `sycl-ls --verbose`. [e32ab1759fed]
- Enabled group algorithms to be used with `tangle_group` and `opportunistic_group` arguments. [29e629e00e83]
- Implemented `info::device::backend_version` query. [df743a5c97e1]
- Added `fixed_size_group` support to algorithms. [01ac0338bfa9]
- Added simple `abs(int)` to imf libdevice. [2a3aaeea5be8]
- Added `ballot_group` support to algorithms. [70a35de4eacb]
- Implemented interoperability support for images for the Level Zero backend. [836ceec36b57]
- Added `marray` support for math built-in functions. [a7a6de2e40a2][c274f40588ff][2953d2548791]
- Enabled inline assembly support in ESIMD kernels. [f5a062dd057f][48abe38c0bee]
- Enhanced interop queue interface to choose standard or immediate commandlists. [0ff3ec40e241]
- Enabled double type for `atomic_update()` ESIMD API. [98c443543c72]
- Added support for `addc` and `subb` operations for ESIMD. [0a098933d9ba]
- Allowed zero-sized 3D accessors. [5cb8279ddb8e]

### Documentation
- Added experimental SYCL bindless images extension [`sycl_ext_oneapi_bindless_images`](https://github.com/intel/llvm/tree/99cbdd15fef2/sycl/doc/extensions/proposed/sycl_ext_oneapi_bindless_images.asciidoc) proposal. [74e83d5af06c][99cbdd15fef2]
- Added [`sycl_ext_oneapi_work_group_local`](https://github.com/intel/llvm/tree/1d6cff1a8e5a/sycl/doc/extensions/proposed/sycl_ext_oneapi_work_group_local.asciidoc) extension proposal. [1d6cff1a8e5a]
- Moved [`sycl_ext_oneapi_device_global`](https://github.com/intel/llvm/tree/d812d1e13ab5/sycl/doc/extensions/experimental/sycl_ext_oneapi_device_global.asciidoc) from proposed to experimental. [d812d1e13ab5]
- Added [`DeviceConfigFile`](https://github.com/intel/llvm/tree/3f4b778dd163/sycl/doc/design/DeviceConfigFile.md) design document. [3f4b778dd163]
- Added [`sycl_ext_oneapi_address_cast`](https://github.com/intel/llvm/tree/9cab5598bc82/sycl/doc/extensions/proposed/sycl_ext_oneapi_address_cast.asciidoc) extension. [9cab5598bc82]
- Added [`sycl_ext_intel_grf_size`](https://github.com/intel/llvm/tree/83d099758770/sycl/doc/extensions/proposed/sycl_ext_intel_grf_size.asciidoc) extension. [83d099758770]
- Added proposal for [`sycl_ext_oneapi_append_and_shift`](https://github.com/intel/llvm/tree/1a283acaac3c/sycl/doc/extensions/proposed/sycl_ext_oneapi_append_and_shift.asciidoc) extension. [1a283acaac3c]
- Updated description of accessor-based memory APIs in [`sycl_ext_intel_esimd`](https://github.com/intel/llvm/tree/9e5889918277/sycl/doc/extensions/experimental/sycl_ext_intel_esimd/sycl_ext_intel_esimd.md). [9e5889918277]
- Added draft of [`sycl_ext_oneapi_prefetch`](https://github.com/intel/llvm/tree/034ac0958662/sycl/doc/extensions/proposed/sycl_ext_oneapi_prefetch.asciidoc). [034ac0958662]
- Added draft of [`sycl_ext_oneapi_barrier`](https://github.com/intel/llvm/tree/32144edcf982/sycl/doc/extensions/proposed/sycl_ext_oneapi_barrier.asciidoc) [32144edcf982]
- Added legacy SYCL 1.2.1 image aspect [`sycl_ext_intel_legacy_image` extension](https://github.com/intel/llvm/tree/78b709dd5bb8/sycl/doc/extensions/supported/sycl_ext_intel_legacy_image.asciidoc). [78b709dd5bb8]
- Added [`sycl_ext_intel_data_flow_pipes_properties`](https://github.com/intel/llvm/tree/5b0a4615874b/sycl/doc/extensions/experimental/sycl_ext_intel_data_flow_pipes_properties.asciidoc) extension. [5b0a4615874b]
- Updated [`sycl_ext_intel_dataflow_pipes`](https://github.com/intel/llvm/tree/6b2d66b8d52f/sycl/doc/extensions/supported/sycl_ext_intel_dataflow_pipes.asciidoc) extension for FPGA host. [6b2d66b8d52f]

## Improvements

### SYCL Compiler
- Deprecated `-fsycl-link-huge-device-code` in favor of a new option, `-flink-huge-device-code`. The new option is identical in functionality but allowed with `-fopenmp-targets`. [79ea9b7aa52b]
- Optimized size of produced binaries when `device_global` is used. [f99335f5e71a]
- Started to emit an error when PCH is triggered in SYCL mode. [062446584d52]
- Improved FPGA archive device unbundling with AOCO. [4ed3676f6402]
- Moved imf `abs` to a separate device library for Deep Learning. [e4f074a6aabe]
- Fixed the bug report URL for DPC++. [98b7de88a7d9]
- Started to link with `bfloat16` related device libraries only when they are used. [663042b04b63]
- Started to properly pass `-fsycl-esimd-force-stateless-mem` to the host code compilation. [8b85b6400b82]

### SYCL Library
- Combined ADL-S and RPL-S device architectures. [f87be6f17428]
- Implemented `multi_ptr` default to be legacy to avoid code break with SYCL 1.2.1. [52edb3798749]
- Started to use aggregate initialization for `group_local_memory` arrays according to updated [`sycl_ext_oneapi_local_memory`](https://github.com/intel/llvm/tree/b50440ec76b9/sycl/doc/extensions/supported/sycl_ext_oneapi_local_memory.asciidoc).[b50440ec76b9]
- Deprecated ESIMD Emulator.  [177680e3e918]
- Deprecated `ext::oneapi::sub_group`. [0662e2a81014]
- Improved error message related to `ext_intel_free_memory` aspect. [c1bfcaf6d6ed]
- Removed non-necessary barrier after region copy in the Level Zero backend. [768e61e7966c]
- Fixed `get_info<device::free_memory>` to check its aspect. [dfb75675bc0a]
- Removed the workaround for release of auxiliary buffers. [fda9171fc218]
- Enabled optimization for read-only buffer accesses from multiple devices in a context. [dddaf7f5d54d]
- Removed old specialization constants extension and implementation. [e50ae05e28d3][833a9fe0a9cf]
- Improved `is_compatible` to check if specific target is defined with `-fsycl-targets` and change the result. [ef033238cfb5]
- Improved quotient approximation for host `sycl::remquo`. [d335fdf4afa2]
- Allowed `accessor` constructed with zero-size buffers. [0c0809590f37]
- Improved resources recycling for the Level Zero backend by cleaning up events on in-order queue wait. [260182a1ad75]
- Added `code_location` parameter to the rest of `sycl::queue` methods which allows to improve error reporting. [fdd609a5c724]
- Enabled `xpti::node_create` signal emit for parallel_for that bypasses graph. [4eaaaa963ca2]
- Improved the accuracy of host `sycl::cospi`. [37aa84b7afd5]
- Replaced usage of outdated memset OpenCL API `clEnqueueMemsetINTEL` with `clEnqueueMemFillINTEL`. [c5d0e1d4d9be]
- Added memory pointer to XPTI memory allocation event metadata to allow more informative XPTI memory tracing. [01ffe65da562]
- Implemented recycling of immediate command lists for queues in a context for the Level Zero backend. [2ddbf8c3b7a5]
- Optimized `ext_oneapi_submit_barrier()` for in-order queue for the Level Zero backend. [c21fd476c78a]
- Change the `SYCL_PI_LEVEL_ZERO_USM_RESIDENT` default to force device allocations only. [33874f76c59f]
- Started to report false for `aspect::image` on all devices. [5cf0f7cf5c21]
- Removed lambda parameter from "if_architecture_is" according to [`sycl_ext_oneapi_device_architecture`](https://github.com/intel/llvm/tree/71f745bcda7d/sycl/doc/extensions/experimental/sycl_ext_oneapi_device_architecture.asciidoc). [71f745bcda7d]
- Improved error reporting when `reqd_work_group_size` is not supported by a device. [084c02797084]
- Adjusted static restriction checks for `block_2d` APIs. [7902d6b35348]
- Disallowed local accessor misuse according to SYCL 2020. [d7a5ec048368]
- Implemented passing address space information to SPIR-V Joint Matrix Load/Store intrinsics. [854ab7eab5e4]
- Enabled immediate command lists by default for the Level Zero backend. [9e5b2ed1ddc3]
- Improved SYCL RT performance by removing extra map lookup for eliminated kernel arguments during enqueue of cached kernels. [7a1fa6c38e03]
- Changed the default to `SYCL_PI_LEVEL_ZERO_USM_RESIDENT=2`. [85435666d477]
- Added heuristics to reduce overhead from immediate command-list cleanup in the Level Zero backend. [ce69d6906e30]
- Renamed `cluster_group` to `fixed_size_group`. [62c091a0af17]
- Added error for invalid uniform arguments in InvokeSIMD. [487c1f8cb040]
- Reduced overhead from queue profiling by using `steady_clock::now()` to retrieve host time as `piGetDeviceAndHostTimer` has large overhead. [070598e433a7]
- Implemented graceful handling of the case that Level Zero runtime was already unloaded when we do cleanup. [928a919f15d9]
- Improved error messages for InvokeSIMD. [0c74bbb20e65]
- Updated `native_specialization_constant()` for the case when no specialization constants are present. [9eb1184686fb]
- Optimized re-use of command lists in the Level Zero backend. [24ae0153e8e8]
- Changed the behavior of `sycl::maximum` and `sycl::minimum` to be consistent with `std::max` and `std::min`. [77d648d949c3]
- Switched to sycl::exception instead of sycl::runtime_error in the majority of SYCL device headers. sycl::runtime_error is deprecated in SYCL 2020. [23a836f39af6]

### SYCL Tools
- Added USM calls parameter verification layer to `sycl-trace` which provides improved diagnostic on accessing invalid device memory under USM. [758bd264bb90]
- Started to print supported SG sizes in `sycl-ls --verbose`. [12515177ab03]

### Documentation
- Updated restrictions in [`sycl_ext_oneapi_device_architecture`](https://github.com/intel/llvm/tree/a69a54f38f2f/sycl/doc/extensions/experimental/sycl_ext_oneapi_device_architecture.asciidoc) extension. [a69a54f38f2f]
- Added `invoke_simd` SLM example and `simd_view` example to [`README`](https://github.com/intel/llvm/tree/f1986b40e04b/sycl/doc/extensions/supported/sycl_ext_intel_esimd/examples/README.md). [c99668ee5985][f1986b40e04b]
- Clarified ways to set sub-group sizes in [`sycl_ext_oneapi_named_sub_group_sizes`](https://github.com/intel/llvm/tree/3285e0fe5590/sycl/doc/extensions/proposed/sycl_ext_oneapi_named_sub_group_sizes.asciidoc) [3285e0fe5590]
- Added `invoke_simd` example to [`README`](https://github.com/intel/llvm/tree/e860b140d506/sycl/doc/extensions/supported/sycl_ext_intel_esimd/examples/README.md). [e860b140d506]
- Added [`README`](https://github.com/intel/llvm/blob/453ffdcf41dd50ec66b5cc9ea6304fadd10f18e8/sycl/plugins/unified_runtime/README.md) for the Unified Runtime directory. [453ffdcf41dd]
- Updated the versioning section in [`ABIPolicyGuide`](https://github.com/intel/llvm/tree/dfe6e4e542ab/sycl/doc/developer/ABIPolicyGuide.md). [dfe6e4e542ab]
- Added `invoke_simd` section to [`README`](https://github.com/intel/llvm/tree/52567e8cd428/sycl/doc/extensions/experimental/sycl_ext_intel_esimd/README.md) and updated [sycl_ext_intel_esimd](https://github.com/intel/llvm/tree/52567e8cd428/sycl/doc/extensions/experimental/sycl_ext_intel_esimd/sycl_ext_intel_esimd.md) [52567e8cd428]
- Updated [`sycl_ext_oneapi_device_architecture`](https://github.com/intel/llvm/tree/acfc905a994a/sycl/doc/extensions/experimental/sycl_ext_oneapi_device_architecture.asciidoc) and [`sycl_ext_oneapi_device_if`](https://github.com/intel/llvm/tree/acfc905a994a/sycl/doc/extensions/proposed/sycl_ext_oneapi_device_if.asciidoc) to allow captures in `if_device_has`, `if_architecture_is`. [acfc905a994a]
- Updated [`SPV_INTEL_joint_matrix`](https://github.com/intel/llvm/tree/adbbf0f39c25/sycl/doc/design/spirv-extensions/SPV_INTEL_joint_matrix.asciidoc) extension. [adbbf0f39c25]

## Bug fixes

### SYCL Compiler
- Fixed `uses_aspects` to be applied to function declarations and not only function definitions. [5d928976f9e2]
- Fixed `ivdep` attribute in template contexts. [008e3743929e]
- Fixed option restriction for device with Windows. [be3482ff3f0e]
- Fixed handling unsupported attributes for ESIMD. [8990c5503d47]
- Fixed optimization option processing for device options. [9a82d283ae2b]
- Enabled proper behavior of optional kernel features with `SYCL_EXTERNAL`. [1dce70f413e6]
- Fixed "SLM init call is supported only in kernels" message for `slm_init()` inside a ESIMD kernel when compiling with `-O0` option. [2056c669b122]
- Fixed memory leak in `Base64::decode`. [28113ec69167]
- Fixed crash when using libraries with `-fsycl`. [ef0d151e57a4]
- Added predefines for `/MDd` to fix certain build issues when using reductions on Windows. [1a602d77b7b2]
- Fixed compilation issue when kernel name is a `class` defined with `final` keyword. [9eeb7825ea2d]
- Fixed an issue with setting `VCSLMSize` attribute which was causing JIT compilation failure when using `slm_init()` in ESIMD kernel.  [f24a0bb343b2]
- Fixed option for undefined symbols for hostdep link to work with gold linker. [554ed2187d32]
- Fixed directory field of `DIFile`. [797187de2199]

### SYCL Library
- Fixed the bug in the Level zero backend where `zeModuleDestroy` is called although `ext::oneapi::level_zero::ownership::keep` is passed to `make_kernel_bundle` leading to a double free corruption. [1344fa29e6eb]
- Addressed specification mismatches for `multi_ptr`. [ed5941feb363]
- Fixed corner case when using `short` or `char` with exclusive scan. [aa31c1e2deb9]
- Updated the legacy `multi_ptr` construction from `local_accessor`s to conform with the construction method used by other accessors. [6314af860c8c]
- Made `get_pointer` `noexcept` to comply with SYCL 2020. [85d84114510c]
- Started to return the correct `sycl::errc` in case of invalid `sycl::reqd_work_group_size`. [d16a1fd39c08]
- Fixed `sycl::remquo` truncation error. [43fbde38e26f]
- Fixed linkage errors when trying to get `mem_channel` property or check it using `has_property()`. [9352172a9b81]
- Fixed dynamic loading of `ocloc` tool for online_compiler. [1854b263672b]
- Fixed global memory reporting for Arc. [0a99d87a7f31]
- Fixed `sycl::sub_group` to follow by-value semantics according to SYCL 2020. [064f332f156c]
- Fixed invalid value returned by `event.get_info<sycl::info::event::command_execution_status>()` for OpenCL backend. [5a91df97973a]
- Fixed assignment operator for ESIMD globals. [a3a9dd02d420]
- Fixed the Level Zero backend to not destroy build log on program build failure.[b7ffbc776b37]
- Fixed error appearing when validation layer is enabled for the Level Zero with `ZE_DEBUG=6`. [abfdb6306d7c]
- Allowed group algorithms to accept a function object with an explicit type. [78f09e6f31c6]
- Fixed `sycl::is_compatible()` for the case when vector of kernels ids is empty. [23fbba1e6b8b]
- Fixed `multi_ptr` ctor for extended address spaces. [a69e5152874e]
- Fixed trivially_copyable to device_copyable for `fill` and `copy`. [9c662e228fa4]
- Fixed range deductions in reduction `parallel_for`. [3024161f1b7b]
- Adjusted `multi_ptr` deduction guides to comply with SYCL 2020. [3df6fdc58b00]
- Fixed gather/scatter with accessors when passing scalar for ESIMD backend. [27755824d050]
- Fixed lost data during implicit conversion in local and host accessors. [7618dffd78ae]
- Fixed memory leak because of unnecessary kernel retain. [f110fd73f8e7]
- Fixed the pointer type contained by a read-only accessor to const. [fd4598035b7e]
- Fixed `max_sub_group_size` query for devices without sub groups.[98cdf4c3acda]
- Removed undefined behavior in host-side `abs_diff`. [a4105101e3c5]
- Fixed `vec::as<vec<bool, N>>()`. [72cffff649f5]
- Fixed segfault when using `ZE_DEBUG` and an exception is thrown on device selection phase. [05866c17ce87]
- Removed an unnecessary `multi_ptr` conversion which had conflict with an existing one resulting in ambiguity on implicit conversion. [aba3007fe71b]
- Fixed lost data in accessor implicit conversion. [10551294dcc7]
- Fixed stack smashing which happened in some device info queries on OpenCL backend. [f4577ff62ace]
- Stopped shipping the Level Zero loader and headers with the DPCPP toolchain installation. [6581bc013a46][ad8b7d447612]
- Corrected the PVC device id check in the Level Zero backend. [59a23020e1be]
- Fixed ambiguity for `bit_cast`. [834df4737a85]
- Fixed nan/inf handling in `sycl::stream` when in `-fast-math` mode. [757e4a72d58d]
- Allowed for different types in group algorithms `reduce`, `exclusive_scan`, and `inclusive_scan`. [8ad0e1df6e76]
- Fixed bug in get methods in the `config_2d_mem_access` class for 2D block stateless load/store API. [7e5f529db655]
- Fixed compilation error for `sycl::fabs` builtin in `-fast-math` mode. [43cb2c0e4867]
- Allowed `host_task` deduction tags in placeholder accessors. [21beef77542b]
- Added iterator operations for zero-dimension accessors to comply with SYCL 2020. [6eb8b4369b64]
- Fixed rounding issue in `__imf_vavgs`. [38d05361f06e]
- Fixed bug in `fill` operation for zero-dimensional accessor. [5b46e0dbcd2c]
- Fixed arguments passed to `ocloc` via `-ftarget-compile-fast`. [d8bb69e5b166]
- Implemented implicit conversion for local and host accessors to comply with SYCL 2020. [3d4da8fb5eee]
- Throw exception when empty accessor calls `require()` to comply with SYCL 2020. [aa08c7e9664a]
- Fixed `operator~` for `sycl::vec<bool, N>`. [58e64ce83be6]
- Fixed `sub_group` shuffle for vectors of `long long` and `half`.  [eeba87925d53]
- Fixed static destruction order issue in OpenCL extension fptr cache. [379a094d8726]
- Fixed leak of active barriers' events in the Level Zero backend. [86860693375c]
- Implemented proper queries for `aspect::ext_oneapi_srgb`. [23978b462fc7]
- Added `vec` assignment from scalar and more `vec` modulus overloads to comply with SYCL 2020. [68e13d41f9c3]
- Fixed handling of `mem_channel` buffer property. [909343107452]
- Fixed `operator&` and `operator[]` in `local_accessor<const T>`. [d5477c87c700]
- Fixed the third argugment type for `select` built-in. [528eba8660a9]
- Fixed `native_specialization_constant()` API implementation. [41344ed7da50]
- Moved Level Zero specific collectors to dynamic libraries loaded by request in sycl-trace tool to support the case when Level Zero driver is not available. [4c5250643b3c]
- Added missing `marray` relational functions: `any`, `all`, `bitselect`; fixed scalar `select` relational function; aligned scalar `abs` integer function with SYCL 2020 and fixed math functions which take `multi_ptr` argument. [8b0b210a95db]
- Added `operator[]` and `element_type` to swizzle vec to align with SYCL 2020. [9050d3325f2c]
- Fixed buffer range in `atomic_memory_order_acq_rel`. [9fb49f757f04]
- Started to throw for invalid `global_work_size` query. [61433888d4b5]
- Fixed the Level Zero backend to report events as submitted, not running, until they are completed. [502e0fdd3f3c]
- Allowed raw send ESIMD API to use non-standard types like `sycl::half`. [8022c6a4f6ec]
- Started to throw for invalid `info::kernel::num_args query` to comply with SYCL 2020. [93e8ca3ed2eb]
- Updated `group_broadcast` to support `vec` types. [09ae6434f55f]
- Fixed reductions to avoid implicit `atomic64` requirements. [e11b35835722]
- Added a partial profiling workaround for acc devices because queue profiling is no longer supported for OpenCL version < 2.1 after recent changes regarding command submit profiling info. [097d21c33599]
- Fixed program build API for the Level Zero backend to retain build-log when program build failed. [7d9f5ac5e5da]
- Implemented missing std::hash specializations for local_accessor and host_accessor to comply with SYCL 2020. [919246bacb82]
- Fixed `global_work_size` kernel query. [d666b95af56a]
- Fixed incorrect `sycl::vec<bool, N>` constructor behavior. [45aa49bb3f06]
- Fixed `weak_object` for `host_accessor` and `stream`. [225b3ecad28d]
- Fixed incorrect behaviors in some operations using `sycl::vec<bool, N>`. [61f6ce4579b7]
- Fixed integration footer for `specialization_id`. [dd96dfaf12da]
- Fixed compilation break occurring when `bfloat16` constructor is used in a kernel. [bc063ac7a9e1]
- Fixed crash in subgroup info queries when running on OpenCL backend which doesn't support sub-groups. [2408035da9c9]
- Added missing support for `target::host_task` specialised accessor constructor using `mode_target Tag`. [c01d8824b1aa]
- Fixed identityless reductions with unwritten reducers. [cb15f33e861a]
- Fixed dangling pointer issue in xpti tracing. [8a49660d4b44]
- Fixed PI event leak in memcpy2d device-host fallback. [24ec33c39c85]
- Fixed `weak_object` and `owner_less` for `device` objects. [b9b85136d492]
- Added `noexcept` specifier to `vec::byte_size` to comply with SYCL 2020. [cb6c0580dc30]
- Fixed undefined behaviour in vector printf specifier. [d190de65a4e1]
- Fixed `handler::fill` so it works even if pattern is not supported by a backend natively. [7f49367d234b]
- Fixed mechanism to throw exception when placeholder accessor passed to a command. (#10110) [93f477358d74]
- Fixed empty zero-dimensional accessor access range. [b4d2b2c966b1]
- Fixed incorrect write back in a case when a sycl::buffer is constructed with a `const T*` as host data. [d05818637be3][1b2b9ddc7f37]
- Fixed empty accessor default constructor to not create a placeholder to comply with SYCL 2020. [e753fae9fad7]

### Documentation
- Fixed link anchors in [`EnvironmentVariables`](https://github.com/intel/llvm/tree/ba1fc2e68d04/sycl/doc/EnvironmentVariables.md) documentation. [ba1fc2e68d04]
- Fixed InlineAsm example in [`sycl_ext_intel_esimd`](https://github.com/intel/llvm/tree/5ff169559712/sycl/doc/extensions/experimental/sycl_ext_intel_esimd/sycl_ext_intel_esimd.md). [5ff169559712]

## API/ABI breakages
- Removed deprecated `piclCreateProgramWithSource`. [42f560977a0a]
- Removed deprecated `barrier` API. [00533e0b1e7e]
- Removed deprecated `interop_task` [aff7fbb287d0]
- Removed deprecated `sycl::group_local_memory`. [1dc77a73b391]
- Removed deprecated `sycl::detail::bitcast` [61f3650c28ed]
- Removed deprecated `piEnqueueNativeKernel` [c3781336d0f1]
- Removed deprecated backend enum values: `level_zero`, `cuda`, `esimd_cpu`, `hip`. [81900499327a]
- Removed the workaround for release of auxiliary buffers. [fda9171fc218]
- Removed old specialization constants extension and implementation. [e50ae05e28d3][833a9fe0a9cf]
- Removed non-standard RT namespace from `sycl` namespace. [fd636ef3ab99]
- Removed deprecated ESIMD APIs. [91e8364164c6]
- Removed `DISABLE_SYCL_INSTRUMENTATION_METADATA` macro and `_CODELOC*` macro usage from API. [f4525e901cd6]
- Removed non-standard RT namespace from sycl namespace. [b3e0428ffef4]
- Removed `getOSModuleHandle` usage. [f73230d8a8ba]
- Removed deprecated `sycldevice` triple support. [01d7fc097ec6]
- Dropped support for `sycl_ext_oneapi_extended_atomics` extension. [17ea6d10f1d5]
- Dropped support for `sycl_ext_oneapi_group_algorithms` extension. [8b9d2106af0d]
- Removed unneeded backwards compatibility of `make_queue` and `get_native`. [c5f150a41b1d][5918e0c208a1]
- Removed support for binaries generated for triples with `-sycldevice` environment component. [5fc53134fcfc]
- Removed lambda parameter from `if_architecture_is` according to [`sycl_ext_oneapi_device_architecture`](https://github.com/intel/llvm/tree/71f745bcda7d/sycl/doc/extensions/experimental/sycl_ext_oneapi_device_architecture.asciidoc). [71f745bcda7d]
- Updated `sycl::exception` to be SYCL2020 compliant. [ce26ac08a383]
- Replaced deprecated `sycl::runtime_error` with SYCL 2020 compliant `sycl::exception` in SYCL RT. [f08dc9473d40]
- Renamed `win_proxy_loader` to `pi_win_proxy_loader`. [2648b7c5e1a4]
- Promoted the return type changes of SYCL relational builtins that changed between SYCL 1.2.1 and SYCL 2020 out from the guard of `SYCL2020_CONFORMANT_APIS`. [35fe4a21e1cb]
- Fixed `get_pointer` to return `T*` for `target::device` specialized accessor according to specification. [712cb4e4f75e]
- Fixed `max_work_item_sizes` return type from `id` to `range` according to SYCL 2020. [8f09d3e603da]
- Deprecated experimental `set_kernel_properties` API and `use_double_grf/use_large_grf` properties were removed. New API provided in the extension [`sycl_ext_intel_grf_size`](https://github.com/intel/llvm/tree/370aa2a01711/sycl/doc/extensions/experimental/sycl_ext_intel_grf_size.asciidoc) has to be used. [f363bb272b5a]

## Known issues
- Having MESA OpenCL implementation which provides no devices on a
  system may cause incorrect device discovery. As a workaround such an OpenCL
  implementation can be disabled by removing `/etc/OpenCL/vendor/mesa.icd`.
- `-fsycl-dead-args-optimization` can't help eliminate offset of
  accessor even though it's created with no offset specified
- SYCL 2020 barriers show worse performance than SYCL 1.2.1 do. [18c80faa]
- When using fallback assert in separate compilation flow it requires explicit
  linking against `lib/libsycl-fallback-cassert.o` or
  `lib/libsycl-fallback-cassert.spv`
- Limit alignment of allocation requests at 64KB which is the only alignment
  supported by Level Zero. 7dfaf3bd
- User-defined functions with the name and signature matching those of any
  OpenCL C built-in function (i.e. an exact match of arguments, return type
  doesn't matter) can lead to Undefined Behavior.
- A DPC++ system that has FPGAs installed does not support multi-process
  execution. Creating a context opens the device associated with the context
  and places a lock on it for that process. No other process may use that
  device. Some queries about the device through `device.get_info<>()` also
  open up the device and lock it to that process since the runtime needs
  to query the actual device to obtain that information.
- The format of the object files produced by the compiler can change between
  versions. The workaround is to rebuild the application.
- Using `sycl::kernel_bundle` API to refer to a kernel defined
  in another translation unit leads to undefined behavior
- Linkage errors with the following message:
  `error LNK2005: "bool const std::_Is_integral<bool>" (??$_Is_integral@_N@std@@3_NB) already defined`
  can happen when a SYCL application is built using MS Visual Studio 2019
  version below 16.3.0 and user specifies `-std=c++14` or `/std:c++14`.
- Printing internal defines isn't supported on Windows. [50628db1]
- The support of accessor and local_accessor for ESIMD is still limited comparing to SYCL.
- [`sycl_ext_oneapi_root_group`](https://github.com/intel/llvm/tree/743c35be2da7/sycl/doc/extensions/proposed/sycl_ext_oneapi_root_group.asciidoc) implementation has the following limitations: `max_num_work_group_sync` query always returns `1` and the implemented barrier has work group scope since all work items in a root group are currently in the same work group.


# March'23 release notes

Release notes for commit range [`ca54ea30..cb91c232`](https://github.com/intel/llvm/compare/ca54ea30...cb91c232)

## New features

### SYCL Compiler
- Added semi-dynamic SLM allocation - `sycl::ext::intel::experimental::esimd::slm_allocator`
[e7674019]
- Added support of compilation of the `.cu` files containing CUDA, and SYCL 
codes in the same translation unit [31608c24] [f5126d28]
- Added support of `amd-gpu-gfx1034` as an acceptable value for 
`-fsycl-targets` [5e86a41d]

### SYCL Library
- Implemented [`sycl_ext_oneapi_matrix`](doc/extensions/experimental/sycl_ext_oneapi_matrix/sycl_ext_oneapi_matrix.asciidoc) extension using new unified interface [f4a9ef1b]
- Partially implemented [`sycl_ext_codeplay_kernel_fusion`](doc/extensions/experimental/sycl_ext_codeplay_kernel_fusion.asciidoc) extension for Linux: Intel CPU and GPU devices are fully supported, except specialization constants, streams, images and reductions [89e82e3f] 
[5d7e86d4] [ab0c46bd] [b5d5aada] [325bc4e2] [cf245f62]
- Added default constructor to `sycl::local_accessor` [589824d7]
- Added support of `SYCL_PI_HIP_MAX_LOCAL_MEM_SIZE` environment variable for 
HIP backend [92f6d688]
- Aligned `sycl::vec` aliases with SYCL 2020 [1299e21c] [791ac603] [18f2cdab]
- Partially implemented [`sycl_ext_intel_fpga_kernel_interface_properties`](doc/extensions/proposed/sycl_ext_intel_fpga_kernel_interface_properties.asciidoc) extension [c5d6b89c]
- Added support for `sycl::ext::oneapi::experimental::if_architecture_is` to 
ESIMD kernels [7a811191]
- Implemented `sycl::opencl::cl_*` aliases [e98d732406cc]
- Implemented `annotated_arg` headers from [`sycl_ext_oneapi_annotated_arg`](doc/extensions/experimental/sycl_ext_oneapi_annotated_arg.asciidoc) extension [1649fe46]
- Added `bfloat16` support to imf libdevice functions [378678a3] [e85b51fc] 
[25a3813b] [2b00cf95]
- Added `sycl::device::get_info<info::device::aspects>` specialization 
[7869c17a]
- Added `rbegin()`, `rend()`, `crbegin()` and `crend()` methods to 
`sycl::accessor`, `sycl::host_accessor` and `sycl::local_accessor` [baa3d739]
- Added `accessor_ptr` alias to `sycl::accessor` and `sycl::local_accessor` 
[6b4cd9d0]
- Implemented `get_multi_ptr` method for `sycl::accessor` and 
`sycl::local_accessor` [f4185c75]
- Added `size_type` to `sycl::accessor` and `sycl::local_accessor` classes 
[0a17e16c]
- Implemented accessors implicit conversions [51b7f781]
- Implemented templated `sycl::kernel_bundle::has_kernel` functions [50e798f9]
- Implemented `sycl::kernel_bundle::get_kernel` function [cf018f16]
- Added support for basic atomic operations for HIP AMD backend [c3c5e923]
- Added the ability to set specialization constants in `sycl::compile` 
[33cd6b1a]
- Added the `value_type` and `binary_operation` member aliases and the 
`dimensions` value to the `sycl::reducer` class [8fdc68d4]
- Implemented `sycl::ext::intel::math::inv` free function [a90d0fd9]
- Added APIs that emulate the behaviors of CUDA SIMD intrinsic [b0bfdfa7]
- Added overloads on all binary operators with scalars as the left operand to 
`sycl::marray`, and allowed `half`, `float`, `double` in `&&` and `||` operators 
[a1787de5]
- Added support of Intel(R) Xe Matrix Extensions with SIMD8 capability 
[8316a9c8]
- Implemented device headers part of `sycl::any_device_has` and 
`sycl::all_devices_have` [81baa50b]
- Added `host_task` in `sycl::target` enum class [a8401952]
- Added support of `sycl::marray` to relational, common and some math functions 
[c3a40760] [c5c7ac2e] [d8715418]
- Implemented [`sycl_ext_oneapi_annotated_ptr`](doc/extensions/experimental/sycl_ext_oneapi_annotated_ptr.asciidoc) extension [3250db55]
- Implemented [`sycl_ext_oneapi_cuda_tex_cache_read`](doc/extensions/experimental/sycl_ext_oneapi_cuda_tex_cache_read.asciidoc) extension [5360825e650b]
- Introduced an API to get thread ID and sub-device ID for ESIMD extension 
[dd6145e8]
- Implemented `kernel_bundle::native_specialization_constant()` [4a412307]
- Implemented SYCL 2020 reductions without a known or specified identity 
[648845fb] [1404ff1d]
- Implemented basic functionality for the following group types: `ballot_group`, 
`cluster_group`, `tangle_group`, `opportunistic_group` as part of 
[`sycl_ext_oneapi_non_uniform_groups`](doc/extensions/proposed/sycl_ext_oneapi_non_uniform_groups.asciidoc) extension [86894205]
- Implemented `is_group` trait and changed the definition of `is_group_v` to be 
in line with SYCL 2020 [10e07a38]
- Implemented `atomic_memory_scope_capabilities` and 
`atomic_memory_order_capabilities` device queries for OpenCL and Level Zero 
backends [2816c14072f3] [b18e6ea83c62]
- Implemented `atomic_fence_scope_capabilities` and 
`atomic_fence_order_capabilities` device queries for OpenCL and Level Zero 
backends [632946ac1cbf]
- Implemented `sycl::ext::intel::experimental::esimd::bfn` function from ESIMD 
extension [25d0475f]
- Made `sycl::ext::oneapi::experimental::joint_reduce` work with 
`sycl::sub_group` [cb91c232]

### Documentation
- Added [`sycl_ext_oneapi_annotated_ptr`](doc/extensions/experimental/sycl_ext_oneapi_annotated_ptr.asciidoc), [`sycl_ext_intel_fpga_kernel_arg_properties`](doc/extensions/proposed/sycl_ext_intel_fpga_kernel_arg_properties.asciidoc), and 
[`sycl_ext_oneapi_kernel_arg_properties`](doc/extensions/proposed/sycl_ext_oneapi_kernel_arg_properties.asciidoc) extention specifications [b423e1db]
- Reintroduced `device_ptr` and `host_ptr` under SYCL namespace for 
[`sycl_ext_intel_usm_address_spaces`](doc/extensions/supported/sycl_ext_intel_usm_address_spaces.asciidoc) [ea8a0e56]
- Moved [`sycl_ext_oneapi_memcpy2d`](doc/extensions/supported/sycl_ext_oneapi_memcpy2d.asciidoc) to supported [751acf35]
- Updated documentation for `-fsycl-no-rdc` mode [a27e6fc5]
- Added an initial draft of non-uniform sub-groups proposal [cc7cbd28] 
[a5d62c8a]
- Added a draft of 16-bit atomics extension [1b973aed]
- Added an extension specification for USM allocation functions with properties 
support [3600f922]
- Added `sycl_ext_oneapi_peer_access` extention specification [b5bce772] 
[37960cda]

## Improvements

### SYCL Compiler
- Improved `sycl::reqd_work_group_size` for optional dimensions [b211d0df]
- Enabled DPC++ build with llvm-mingw toolchain on Windows [1715c36b]
- Deprecated usage of `-sycl-std=1.2.1` [8dd6fdff]
- Disabled force inlining of kernel call operator for FGPA [90e0e650]

### SYCL Library
- Added diagnostics that the joint_matrix API is only supported by the Intel and 
CUDA backends [ed596eca]
- Deprecated `sycl::ext::oneapi::leader(group)`, SYCL 2020 member function 
`group::leader` should be used instead [94434e78]
- Deprecated `get_access` for `sycl::host_accessor` in accordance with SYCL 2020 
[d145da8f]
- Deprecated `get_size()` and `get_max_statement_size()` member functions in the 
`sycl::stream` class, added replacements: `size()` and `get_work_item_buffer_size()` [aab0ba71]
- Deprecated legacy scalar and vector type aliases [e83d0ae0] [016d68b0]
- Empty platforms are no longer returned [5e8ae211]
- Started to throw an exception when selected device is not in context 
[262299d1]
- Changed signature of `sycl::host_accessor::get_pointer` in accordance with 
SYCL 2020 [a7473fd2]
- Started to ignore placeholder template parameter for accessors in accordance 
with SYCL 2020 [2b1d3205]
- Started to throw the correct exception when passing unbound accessor to 
command [22cefabf]
- Hid the members of `sycl::reducer` that are not mentioned in the SYCL 2020 
specification and introduced the identity member function [505aa7d1]
- Improved `group_local_memory_for_overwrite` (part of 
[`sycl_ext_oneapi_local_memory`](doc/extensions/supported/sycl_ext_oneapi_local_memory.asciidoc)) 
by default initializing the returned memory, if needed [37a43f93]
- Augmented `sycl::has_known_identity` to return true for `std::complex` and 
`std::plus` operators [2e733125]
- Improved diagnostic when `sycl::accessor` isn't bound to `sycl::handler` 
[3c87f55e]
- Added identities for arithmetic and integral operations on bool in accordance 
with the SYCL 2020 specification, added a check for the specification defined 
identities [091481d8]
- Renamed `raw_send{s}_{load/store}` APIs in ESIMD extension [cb280eab]
- Made `sycl::reducer` uncopyable and immovable in accordance with SYCL 2020 
[7c7efeee]
- Updated aspects list to be SYCL 2020 compliant [b9122b30]
- Updated `sycl::kernel_bundle::has_kernel(kernel_id, device)` to return `true` 
if the kernel is valid for a given device in accordance with SYCL 2020 
[a950450d]

### Documentation
- Added Explicit SIMD design and implementation notes [59dd3879]
- Removed `sycl_ext_oneapi_nd_range_reductions` docs, implementation had been 
removed in the middle of 2022 [a46efa0a]
- Clarified default device_global allocation behavior [15edb1bd]
- Corrected the specified return type for the UUID query [77d25cd8]
- Clarified the usage of Intel GPU drivers in WSL under the docker [1f39511c]

## Bug fixes

### SYCL Compiler
- Fixed `-f[no-]sycl-rdc` option in `clang-cl` [e39d0c18]
- Fixed emission of FPGA annotations [8199de8c]
- Cleaned up Windows defaultlib linking behaviors [aaae3348]
- Fixed an issue where the dimensions of the work-group hint were not correctly 
reversed [ba51bb68]
- Fixed a crash during generation of debug information with the option 
`-ffile-prefix-map` [723b80da]

### SYCL Library
- Fixed `operator-` for constant accessor iterator [890aa072]
- Fixed the case when sub-device free memory may exceed root-device free memory 
[9dc14a29]
- Fixed `sycl::handler::require()` to accept a non-placeholder argument 
[abdc0fe3]
- Fixed `sycl::ext::oneapi::experimental::joint_reduce`, it missed the case when 
work-group size is bigger than size of input data [23b68dbd]
- Added missing include of sycl/builtins.hpp to sycl/ext/intel/math.hpp 
[efe1074e]
- Fixed host compilation issue for lsc_atomic_update for store operation in 
ESIMD extension [fdf8ac77]
- Fixed event profiling for `sycl::info::event_profiling::command_submit` in 
Level Zero and other backends [71d77972]
- Fixed `sycl::group::get_local_linear_id()`, now it uses the correct range to 
linearize the local ID [c3849635]
- Fixed compilation issue when `sycl::stream::size` was called from device code [368b6605]
- Fixed `sycl::host_accessor` zero dimension constructors [23f107d4]
- Fixed atomic memory orders in reduction implementation [2aef0c94]
- Fixed accessor subscript ambiguity and reference type [8c244de9]
- Fixed an issue with missing explicit instantiations in sycl/properties/queue_properties.hpp [2864eac4]
- Fixed return type of `sycl::reducer::combine` [44a26c1e]
- Fixed an error when a scalar offset is provided as a parameter to the API 
[035ef2b2]
- Fixed return type of identity-less reduction as it used an unexpected value for 
the deprecated placeholder template parameter in accessor [c64f88ec]
- Fixed variadic `sycl::marray` constructor by accepting values that are 
indirectly convertible to the element type of the marray [becc391a]
- Fixed `sycl::atomic_ref` constructor to fix operator ambiguity [00c60624]
- Fixed handling of host-side memory in 2D memory operations [2dcc5817]
- Fixed constexpr initialization of `sycl::vec` for `sycl::half` [ba0ea790]
- Fixed default `sycl::host_accessor` iteration methods [f8ba0e48]
- Fixed `lsc_prefetch_2d` in ESIMD extension [10a9c14f]
- Fixed over-allocation in specialization constants [d58eb783]
- Fixed `sycl::handler::get_specialization_constant` segmentation fault that happened 
in certain cases likely due to strict aliasing violations [86c08b3c]
- Fixed a regression in ESIMD's `atomic_update` [1396da2e]
- Fixed return type of the `sycl::accessor::get_pointer` and `sycl::local_accessor::get_pointer` [3fdbfeb6]

## Known issues
- Having MESA OpenCL implementation which provides no devices on a
  system may cause incorrect device discovery. As a workaround such an OpenCL
  implementation can be disabled by removing `/etc/OpenCL/vendor/mesa.icd`.
- `-fsycl-dead-args-optimization` can't help eliminate offset of
  accessor even though it's created with no offset specified
- SYCL 2020 barriers show worse performance than SYCL 1.2.1 do. [18c80faa]
- When using fallback assert in separate compilation flow it requires explicit
  linking against `lib/libsycl-fallback-cassert.o` or
  `lib/libsycl-fallback-cassert.spv`
- Limit alignment of allocation requests at 64KB which is the only alignment
  supported by Level Zero. 7dfaf3bd
- User-defined functions with the name and signature matching those of any
  OpenCL C built-in function (i.e. an exact match of arguments, return type
  doesn't matter) can lead to Undefined Behavior.
- A DPC++ system that has FPGAs installed does not support multi-process
  execution. Creating a context opens the device associated with the context
  and places a lock on it for that process. No other process may use that
  device. Some queries about the device through `device.get_info<>()` also
  open up the device and lock it to that process since the runtime needs
  to query the actual device to obtain that information.
- The format of the object files produced by the compiler can change between
  versions. The workaround is to rebuild the application.
- Using `sycl::kernel_bundle` API to refer to a kernel defined
  in another translation unit leads to undefined behavior
- Linkage errors with the following message:
  `error LNK2005: "bool const std::_Is_integral<bool>" (??$_Is_integral@_N@std@@3_NB) already defined`
  can happen when a SYCL application is built using MS Visual Studio 2019
  version below 16.3.0 and user specifies `-std=c++14` or `/std:c++14`.
- Printing internal defines isn't supported on Windows. [50628db1]

# December'22 release notes

Release notes for commit range [`0f579bae..6977f1ac`](https://github.com/intel/llvm/compare/0f579bae...6977f1ac)

## New features

### SYCL Compiler
- Added support for per-object device code compilation under the option
`-fno-sycl-rdc`. This improves compiler performance and reduces memory usage,
but can only be used if there are no cross-object dependencies. [f884993d]
- Added support for per-aspect device code split mode. [9a2c4fe5]
- Extended support for the large GRF mode to non-ESIMD kernels. [9994934b]
[ab2a42c6]
- Implemented the [`sycl_ext_oneapi_device_architecture`](doc/extensions/experimental/sycl_ext_oneapi_device_architecture.asciidoc)
extension. [0e32a28d] [b59d93cc] [5bd5c871] [e5de913f]
- Implemented the [`sycl_ext_oneapi_kernel_properties`](doc/extensions/experimental/sycl_ext_oneapi_kernel_properties.asciidoc)
experimental extension. [332e4ee0] [27454de6] [70ee3d56] [430c7221]
- Added support for generic address space atomic built-ins to CUDA libclc.
[d6a8fd1f]

### SYCL Library
- Implemented accessor member functions `swap`, `byte_size`, `max_size` and
`empty`. [f1f907a2]
- Implemented SYCL 2020 default accessor constructor. [04928f9f]
- Implemented SYCL 2020 accessor iterators. [5b9fd3ca] [c7b1a000]
- Changed `value_type` of read-only accessors to `const` in accordance with
SYCL 2020. [227614ca]
- Implemented SYCL 2020 `multi_ptr` and `address_space_cast`. [8700b768]
[483984a9] [4a9e9a0e]
- Implemented SYCL 2020 `has_extension` free functions. [7f1a6efd]
- Implemented SYCL 2020 `aspect_selector`. [c0a4a565]
- Implemented new SYCL 2020 style FPGA selectors. [04176516]
- Implemented SYCL 2020 default `async_handler` behavior. [cd93d8f6]
- Implemented SYCL 2020 `is_compatible` free function. [67f6bba2]
- Implemented queue shortcut functions with placeholder accessors. [5ee066e4]
- Added support for creating a kernel bundle with descendent devices of the
passed context's members. [a7827795]
- Implemented non-blocking destruction and deferred release of memory objects
without attached host memory. [894ce256]
- Implemented the [`sycl_ext_oneapi_queue_priority`](doc/extensions/supported/sycl_ext_oneapi_queue_priority.asciidoc)
extension. [cdb09dca]
- Implemented the [`sycl_ext_oneapi_user_defined_reductions`](doc/extensions/experimental/sycl_ext_oneapi_user_defined_reductions.asciidoc)
extension. [8311d790]
- Implemented the [`sycl_ext_oneapi_queue_empty`](doc/extensions/supported/sycl_ext_oneapi_queue_empty.asciidoc)
extension proposal. [c4932957]
- Implemented the [`sycl_ext_oneapi_weak_object`](doc/extensions/supported/sycl_ext_oneapi_weak_object.asciidoc)
extension. [d9484274] [9297f63e]
- Implemented the [`sycl_ext_intel_cslice`](doc/extensions/supported/sycl_ext_intel_cslice.asciidoc)
extension. The old behavior that exposed compute slices as sub-sub-devices is
now deprecated. For compatibility purposes, it can be brought back via the
`SYCL_PI_LEVEL_ZERO_EXPOSE_CSLICE_IN_AFFINITY_PARTITIONING` environment
varible. [5995c618]
- Implemented the [`sycl_ext_intel_queue_index`](doc/extensions/supported/sycl_ext_intel_queue_index.asciidoc)
extension. [d2ec9645] [7179e830]
- Implemented the [`sycl_ext_oneapi_memcpy2d`](doc/extensions/supported/sycl_ext_oneapi_memcpy2d.asciidoc)
extension. [516d4112]
- Implemented device ID, memory clock rate and bus width information queries
from the [`sycl_ext_intel_device_info`](doc/extensions/supported/sycl_ext_intel_device_info.md)
extension. [1d993446] [4f7787c8]
- Implemented `ext::oneapi::experimental::radix_sorter` from the
[`sycl_ext_oneapi_group_sort`](doc/extensions/proposed/sycl_ext_oneapi_group_sort.asciidoc)
extension proposal. [86ba1809]
- Implemented a new unified interface for the [`sycl_ext_oneapi_matrix`](https://github.com/intel/llvm/blob/7dab76e1d33341b1e6bf339ab933552281abb3e2/sycl/doc/extensions/Matrix/dpcpp-joint-matrix.asciidoc)
extension for CUDA. [166bbc36]
- Added support for sorting over sub-groups. [168767c6]
- Added C++ API wrappers for the Intel math functions `ceil`, `floor`, `rint`,
`sqrt`, `rsqrt` and `trunc`. [1b7582bb]
- Implemented a SYCL device library for `bfloat16` Intel math function
utilities. [fc136d69]
- Added support for range reductions with any number of reduction variables.
[572bc503]
- Added support for reductions with kernels accepting `item`. [5d5e9f4e]
- Enabled sub-group masks for 64-bit subgroups. [10d50ed3]
- Implemented the new non-experimental API for DPAS. [55bf1a0d] [1e7a8eaa]
- Added 8/16-bit type support to `lsc_block_load` and `lsc_block_store` ESIMD
API. [f9d8059d]
- Implemented atomic operation support in the ESIMD emulator. [a6a0dea7]
- Added various trivial utility functions for the `half` type. [b4ce7c08]
- Added type cast functions between `half` and `float`/integer types to
libdevice. [599b1b96]
- Implemented the `ONEAPI_DEVICE_SELECTOR` environment variable that, in
addition to supporting `SYCL_DEVICE_FILTER` syntax, allows to expose GPU
sub-devices as SYCL root devices and supports negative filters.
`SYCL_DEVICE_FILTER` is now deprecated. [28d0cd3a] [b21e74ef] [77b6f34f]
[6bd5f9ce] [6aefd631]
- Added the `SYCL_PI_LEVEL_ZERO_SINGLE_ROOT_DEVICE_BUFFER_MIGRATION`
enviornment variable. [bd03e0d3]

### Documentation
- Added the [`sycl_ext_oneapi_device_architecture`](doc/extensions/experimental/sycl_ext_oneapi_device_architecture.asciidoc)
 extension specification. [7f2b17ed]
- Added the [`sycl_ext_oneapi_memcpy2d`](doc/extensions/supported/sycl_ext_oneapi_memcpy2d.asciidoc)
extension specification. [296e9c3a]
- Added the [`sycl_ext_oneapi_user_defined_reductions`](doc/extensions/experimental/sycl_ext_oneapi_user_defined_reductions.asciidoc)
extension specification. [cd4fd8c7]
- Added the [`sycl_ext_oneapi_weak_object`](doc/extensions/supported/sycl_ext_oneapi_weak_object.asciidoc)
extension specification. [d9484274]
- Added the [`sycl_ext_oneapi_prod`](doc/extensions/proposed/sycl_ext_oneapi_prod.asciidoc)
extension proposal. [ed7cb4b0]
- Added the [`sycl_ext_codeplay_kernel_fusion`](doc/extensions/experimental/sycl_ext_codeplay_kernel_fusion.asciidoc)
extension proposal. [be3dfbd7]
- Added the [`sycl_ext_intel_queue_index`](doc/extensions/supported/sycl_ext_intel_queue_index.asciidoc)
extension proposal. [f5fb7599]
- Added the [`sycl_ext_intel_cslice`](doc/extensions/supported/sycl_ext_intel_cslice.asciidoc)
extension proposal. [5777e1fe]
- Added the [`sycl_ext_oneapi_group_sort`](doc/extensions/proposed/sycl_ext_oneapi_group_sort.asciidoc)
extension update proposal that introduced sorting functions with fixed-size arrays. [c6d1cafd]
- Added device ID, memory clock rate and bus width device information queries to the
[`sycl_ext_intel_device_info`](doc/extensions/supported/sycl_ext_intel_device_info.md)
extension. [1d993446][4f7787c8]

## Improvements

### SYCL Compiler
- Added the `InferAddressSpaces` pass to the SPIR/SPIR-V compilation pipeline,
reducing the size of the generated device code. [a3ae0dd1]
- Redesigned pointer handling so that it no longer decomposes kernel argument
types containing pointers. [3916d3b6] [d55e9c28] [9b025068]
- Kernel lambda operator is now always inlined in the device code entry point
unless `-O0` is used. [b91b7327] [2359d948]
- Improved entry point handling in the `sycl-post-link` tool. [53d9c7b6]
- The `reqd_work_group_size` attribute now works with 1, 2 or 3 operands.
[4ff42c3c]
- Enabled using `-fcf-protection` option with `-fsycl`, which results in it
being applied only to host code compilation and producing a warning. [b6f61f62]
- Linux based compiler driver on Windows now pulls in the `sycld` debug library
when `msvcrtd` is specified as a dependent library. [ebf6c597]
- Added `/Zc:__cplusplus` as a default option during host compilation with MSVC.
[e7ed8607]
- Improved the `ESIMDOptimizeVecArgCallConv` optimization pass to cover more IR
patterns. [49264549]
- Added support for more types in ESIMD lsc functions. [d9e40ecd]
- Added error diagnostics for using
`sycl::ext::oneapi::experimental::annotated_arg/ptr` as a nested type.
[321c733c]
- The status of `bfloat16` support was changed from experimental to supported.
[7b47ebb6]

### SYCL Library
- Updated `online_compiler` with Gen12 GPU support. [adfb1c10]
- `get_kernel_bundle` and `has_kernel_bundle` now check that the kernels are
compatible with the devices. [91b15156]
- Waiting for an event associated with a kernel that uses a stream now also
waits for the stream to be flushed. [1db0e81f]
- Added the requested device type to the message of the exception thrown when no
such devices are found. [6b83ad7c]
- Optimized `operator[]` of `host_accessor`. [01e60f76]
- Improved reduction performance on discrete GPUs. [99bdc823]
- Added `invoke_simd` support for functions with `void` return type. [3fd08509]
- The Level Zero plugin now creates every event as host-visible by default.
[f3d245d6]
- Added Level Zero plugin support for global work sizes greater than
`UINT32_MAX` as long as they are divisible by some legal work-group size and
the resulting quotient does not exceed `UINT32_MAX`. [62dd13dd]
- Improved native Level Zero event handling in the immediate command list mode
by removing excessive status queries. [179ffa16]
- Removed an uninitialized buffer migration copy in the Level Zero plugin
[b0c34046]
- Implemented an optimization that reuses discarded Level Zero events in the
plugin. [b1533c5e]
- The host device is now inaccessible. [5b13d5b6]
- Removed deprecated `make_queue` API. [9fc1d684]
- Deprecated `group::get_global_range()`. [95338719]

### Documentation
- Updated the [`sycl_ext_oneapi_device_if`](doc/extensions/proposed/sycl_ext_oneapi_device_if.asciidoc)
extension proposal to allow chaining `if_device_has`, `else_if_device_has` and
`else_device` calls. [7f2b17ed]
- Updated the [`sycl_ext_intel_fpga_device_selector`](doc/extensions/supported/sycl_ext_intel_fpga_device_selector.asciidoc)
extension to use SYCL 2020 style selectors. [04176516]
- Updated the [`sycl_ext_intel_fpga_kernel_interface_properties`](doc/extensions/proposed/sycl_ext_intel_fpga_kernel_interface_properties.asciidoc)
extension proposal to allow the compiler to determine the initiation interval.
[1a1fd8d1]
- Updated the [`sycl_ext_intel_usm_address_spaces`](doc/extensions/supported/sycl_ext_intel_usm_address_spaces.asciidoc)
extension to adhere to SYCL 2020 `multi_ptr`. [4a9e9a0e]
- Added a new matrix use parameter to `joint_matrix` from the
[`sycl_ext_oneapi_matrix`](https://github.com/intel/llvm/blob/f2983fc0d8fcd7bd6022a7006ad489c591838041/sycl/doc/extensions/experimental/sycl_ext_oneapi_matrix/sycl_ext_oneapi_matrix.asciidoc)
extension specification. [52f34fd5]
- Removed `queue::size` and `queue::get_wait_list` functions from the
`sycl_ext_oneapi_queue_status_query` extension due to performance overhead
implications and renamed it to
[`sycl_ext_oneapi_queue_empty`](doc/extensions/supported/sycl_ext_oneapi_queue_empty.asciidoc). [b540f819]
- Clarified ESIMD emulator device selection behavior in
[`sycl_ext_intel_esimd`](doc/extensions/experimental/sycl_ext_intel_esimd/esimd_emulator.md).
[9b5f2884]
- Updated the [`sycl_ext_oneapi_device_architecture`](./doc/extensions/experimental/sycl_ext_oneapi_device_architecture.asciidoc)
extension to include NVIDIA and AMD architectures. [c6091dfa] [e5de913f]
- Updated [get started guide](doc/GetStartedGuide.md) to mention CUDA specific
driver options, some of which are necessary for making use of newer hardware
features. [4e5d276f] [f48f96eb]

## Bug fixes

### SYCL Compiler
- Fixed a crash when attempting to compile code that uses a function object
without a defined call operator as a kernel. [490ee550]
- Fixed a crash that occurred during compilation of device code with a captured
structured binding. [0e455c91]
- Fixed the `work_group_size_hint` attribute not being applicable to lambda
functions using non-conforming syntax. [c2a0db08]
- Fixed integration header parameter kind information for annotated types.
[b8f35cf3]
- Fixed an issue with offload dependencies when using `-fsycl-force-target`.
[54777c00]
- Fixed debug information generation when an integration footer is present.
[83febf94]
- Fixed a `__builtin_printf` related error when compiling device code with
`_GLIBCXX_ASSERTIONS=1`. [d8fd9bc8]
- Fixed a compiler error that occurred during archive generation when using
`-fsycl-link` for FPGA. [3159db53]
- Fixed memory corruption caused by the `ESIMDOptimizeVecArgCallConv` pass.
[86f709c3]
- Fixed a crash during ESIMD intrinsic generation. [20b1bea5]
- Fixed libclc function mangling. [e6c4c154] [f32d34f3]
- Fixed an issue with specifying the offload architecture for AMD and NVIDIA
targets. [41898580]
- Fixed incorrectly issued warnings about CUDA or HIP targets missing from a
linked library. [72d9b052]

### SYCL Library
- Fixed an issue where the in-order queue property was not respected when
submitting USM commands and host tasks. [067d3b39]
- Fixed a memory leak when enqueueing a barrier to a `discard_events` queue.
[968f9e76]
- Fixed a memory leak related to submitting host tasks without memory object
dependencies. [c44050a9]
- Fixed an invalid event error when handling cross-queue no-op dependencies.
[6c9a380d]
- Fixed an error when setting a specialization constant in a command group with
no kernel. [b333cee9]
- Fixed an issue where submitting a kernel that explicitly depends on a host
task was a blocking call that waited for the host task. [c44050a9]
- Removed `noexcept` from some of `usm_allocator` member functions to align with
the specification. [7b026976]
- Fixed `ext::intel::experimental::atomic_update` with the `fcmpwr` operation.
[52923e6d]
- Fixed memory leak issues when constructing a SYCL `kernel`/`kernel_bundle`
using interoperability. [b083e054] [a32021ba] [3c22764f]
- Fixed an error where the native handle returned by `get_native` from a default
constructed event was unusable. [72021733]
- Fixed an issue where reinterpreting a buffer to a `const` type changed the
corresponding `buffer_allocator` type to `const`. [3aabd264]
- Fixed `handler::set_arg` with `local_accessor`. [ef792c67]
- Added the missing default template argument for
`sycl::info::device::max_work_item_sizes`. [3a4e797d]
- Fixed an issue where some aspects could be incorrectly reported as
unsupported by a device. [f90d2b4c]
- Fixed return type of scalar versions of relational functions. The fix
requires defining `SYCL2020_CONFORMANT_APIS` macro. [45d516cc]
- Fixed an issue where device code cache was not used if the compilation was
triggered by different paths. [a0254c98] [1c77f9a7]
- Fixed a use-after-move bug when caching device code built for multiple
devices. [97c0c997]
- Removed the unintended requirement of `fp64` support from stream and ESIMD
float `fmod` implementations. [a67807ac] [cc189044]
- Fixed several complex math operations failing on devices that don't support
`fp64`. [6b24fdc6]
- Aligned host side float-to-half mantissa rounding with device side.
[4cbd459a]
- Fixed float-to-half conversion of the `half` minimum subnormal value on host.
[514708b7]
- Fixed `marray` math function implementation. [73a992b3]
- Fixed an out-of-bounds write in the group operations implementation.
[0fa7542b]
- Fixed a reduction performance regression caused by using the wrong
implementation for the `float` type. [97725f19]
- Fixed header deprecation warnings to work properly on Linux. [2cefad19]
- Fixed deprecation of SYCL 1.2.1 device selectors. [a6222ba6]
- Fixed multiple issues in GDB xmethods scripts. [e49aa083]
- Fixed an issue with `sycl-prof` JSON output. [37a74c7b]
- Fixed compilation errors on Windows when using the ESIMD API. [3aa48db2]
[77e92cea]
- Fixed invalid calculation in the ESIMD `tanh` function. [cda6680e]
- Fixed `kernel_bundle` errors when using ESIMD emulator devices. [9baa9d92]
- Fixed an issue where ESIMD emulator was picked by the default selector even in
the presence of other devices. [44d7926c]
- Fixed an error when querying an ESIMD emulator device for sub-group sizes.
[acca6089]
- Fixed invalid behavior of the maximum sub-group size query on some OpenCL
systems. [5998d7c7]
- Fixed an issue where the OpenCL plugin checked whether a program is supported
on a device by looking up platform version/extensions rather than device
ones. [9f892472]
- Fixed the result of the free device memory query with the Level Zero backend.
[c191fb0a]
- Fixed an issue with `ext_oneapi_barrier` not working when using the Level Zero
backend. [1f8d90fc]
- Fixed a hang after submitting a barrier to a Level Zero in-order queue.
[dd5a191c]
- Fixed an issue that occurred when submitting a barrier to a Level Zero queue
with no prior submissions. [5c9e5434]
- Fixed a memory leak when tracking indirect access in the Level Zero plugin.
[1b79491d]
- Fixed an invalid read issue that occurred during Level Zero event release.
[c49eeda8]
- Fixed a synchronization issue when using device scope Level Zero events.
[9811ef24]
- Fixed an issue that occurred when using `get_native` on a newly constructed
Level Zero queue. [5d0d4ef9]
- Fixed a segmentation fault related to events recycling in immediate command
list mode in the Level Zero plugin. [a3e93e07] [5b021a22]
- Fixed an issue where an invalid maximum of compute units was reported for
Level Zero sub-sub-devices. [b9f49196]
- Fixed a segmentation fault when using Level Zero sub-sub-devices with the
immediate command lists mode. [ed3d35c5]
- Reverted the Level Zero plugin change that preferred using copy engine for
memory read/write operations due to functional regressions. [44aa363a]
- Added the missing `fp16` case of `FMulKHR` libclc function. [4372915c]
- Fixed several bugs in the barrier implementation in the CUDA and HIP plugins.
[1c3d5987] [ce7c594f]
- Fixed `get_native` not working for CUDA devices. [998fd91e]
- Fixed a crash when submitting a kernel with a range of 0 to a CUDA device.
[a3958865]
- Fixed an issue where `make_device` produced duplicate CUDA devices.
[75302c53]
- Fixed an issue where the HIP backend always reported the `fp64` aspect as
missing. [cd832bff]

## Known issues
- Having MESA OpenCL implementation which provides no devices on a
  system may cause incorrect device discovery. As a workaround such an OpenCL
  implementation can be disabled by removing `/etc/OpenCL/vendor/mesa.icd`.
- Compilation may fail on Windows in debug mode if a kernel uses
  `std::array`.  This happens because debug version of `std::array` in
  Microsoft STL C++ headers calls functions that are illegal for the device
  code. As a workaround the following can be done:
   1. Dump compiler pipeline execution strings by passing `-###` option to the
      compiler. The compiler will print the internal execution strings of
      compilation tools. The actual compilation will not happen.
   2. Modify the (usually) first execution string (it should have
      `-fsycl-is-device` option) by adding
      `-D_CONTAINER_DEBUG_LEVEL=0 -D_ITERATOR_DEBUG_LEVEL=0` options to the
      end of the string.  Execute all string one by one.
- `-fsycl-dead-args-optimization` can't help eliminate offset of
  accessor even though it's created with no offset specified
- SYCL 2020 barriers show worse performance than SYCL 1.2.1 do. [18c80faa]
- When using fallback assert in separate compilation flow it requires explicit
  linking against `lib/libsycl-fallback-cassert.o` or
  `lib/libsycl-fallback-cassert.spv`
- Limit alignment of allocation requests at 64KB which is the only alignment
  supported by Level Zero. 7dfaf3bd
- User-defined functions with the name and signature matching those of any
  OpenCL C built-in function (i.e. an exact match of arguments, return type
  doesn't matter) can lead to Undefined Behavior.
- A DPC++ system that has FPGAs installed does not support multi-process
  execution. Creating a context opens the device associated with the context
  and places a lock on it for that process. No other process may use that
  device. Some queries about the device through `device.get_info<>()` also
  open up the device and lock it to that process since the runtime needs
  to query the actual device to obtain that information.
- The format of the object files produced by the compiler can change between
  versions. The workaround is to rebuild the application.
- Using `sycl::kernel_bundle` API to refer to a kernel defined
  in another translation unit leads to undefined behavior
- Linkage errors with the following message:
  `error LNK2005: "bool const std::_Is_integral<bool>" (??$_Is_integral@_N@std@@3_NB) already defined`
  can happen when a SYCL application is built using MS Visual Studio 2019
  version below 16.3.0 and user specifies `-std=c++14` or `/std:c++14`.
- Printing internal defines isn't supported on Windows. [50628db1]

# September'22 release notes

Release notes for commit range [`4043dda3..0f579bae`](https://github.com/intel/llvm/compare/4043dda3...0f579bae)

## New features

### SYCL Compiler

- Added ability to enforce stateless memory accesses for ESIMD. [18111623]
- Added support for `-fsycl-force-target` compiler option. [1d95f2ec]
- Added support for `[[intel::max_reinvocation_delay`]] loop attribute. [90fa5bb0]
- Added support for `-fsycl-huge-device-code` compiler option, which allows
  linking object files larger than 2GB. [f963062c]
- Added support for compiling `.cu` files with SYCL compiler. [e76ad72f]
- Added support for `assert` on HIP backend. [ade1870a]
- Enabled CXX standard library functions for CUDA backend. [1fe92c54]
- Implemented group collective built-in functions for more integral types. [d4933b6f]

### SYCL Library

- Implemented SYCL 2020 callable device selectors. [64f0db7a]
- Implemented SYCL 2020 standalone device selectors. [bfc7e984]
- Added SYCL 2020 property interfaces for `local_accessor`, `usm_allocator`,
  `accessor` and `host_accessor` classes. [1136b403] [da7dcf82]
- Added support for `fpga_simulator_selector`. [9bef890d]
- Added support for `local_accessor`. Deprecated `target::local`. [e4423ef4]
- Added support for querying free device memory on Level Zero backend. [0eeef2b3]
- Added support for querying free device memory on CUDA and HIP backends. [436f0d89]
- Implemented `bfloat16` conversions from/to `float` for host. [2a383f1c]
- Added support for `ext::oneapi::property::queue::discard_events` to
  Level Zero PI plugin. [13721204]
- Added `lsc_atomic` support on ESIMD emulator. [0c051a89]
- Added `dpas` support on ESIMD emulator. [3d506a34]
- Added C++ API for `imf` libdevice built-ins. [830916a3]
- Implemented `make_queue` for CUDA backend. [89460e81]
- Implemented `has_native_event` and `make_event` for CUDA backend. [74369c84]
- Added support of CUDA XPTI tracing. [0cd04144]
- Introduced predicates for ESIMD `lsc_block_store/load`. [f44edce3]
- Added experimental `set_kernel_properties` API and `use_double_grf` property
  for ESIMD. [9a55da53]
- Added "eager initialization" mode to Level Zero PI plugin. It might result
  in an unnecessary work done by the plugin, but ensures the fastest possible
  execution on hot and reportable paths. [c1459598]
- Added full support of element wise operations on `joint_matrix` on CUDA
  backend including `bfloat16` support. [0a1d751b]
- Implemented `group::get_linear_id(int)` method [6e83c127]

### Documentation

- Added stateful to stateless memory access conversion
  [design document](doc/design/ESIMDStatelesAccessors.md). [3e03f300]
- Added [`sycl_ext_oneapi_complex`](doc/extensions/proposed/sycl_ext_oneapi_complex.asciidoc)
  extension proposal. [01589da5]
- Updated [`sycl_ext_intel_fpga_device_selector`](doc/extensions/supported/sycl_ext_intel_fpga_device_selector.asciidoc)
  extension to add `fpga_simulator_accessor`. [9bef890d]
- Added [`sycl_ext_intel_fpga_kernel_interface_properties`](doc/extension/proposed/sycl_ext_intel_fpga_kernel_interface_properties.asciidoc) extension proposal. [4b6bd14b]
- Updated [`sycl_ext_oneapi_complex_algorithms`](doc/extensions/proposed/sycl_ext_oneapi_complex_algorithms.asciidoc)
  extension to include `sycl::complex` as supported type for algorithms. [07c5b48f]
- Clarified sub-group size calculation in [`sycl_ext_oneapi_invoke_simd`](doc/extensions/experimental/sycl_ext_oneapi_invoke_simd.asciidoc) extension spec. [9b33ad0f]
- Updated [`sycl_ext_oneapi_accessor_properties`](doc/extensions/supported/sycl_ext_oneapi_accessor_properties.asciidoc)
  to mark `has_property` API as `noexcept`. [7805aa3f]
- Updated [`sycl_ext_intel_device_info`](doc/extensions/supported/sycl_ext_intel_device_info.md)
  to support querying free device memory. [0eeef2b3]
- Updated [`sycl_ext_oneapi_matrix`](https://github.com/intel/llvm/blob/770f540d8b600c8c16df12dfccbf38fa780cf77a/sycl/doc/extensions/experimental/sycl_ext_oneapi_matrix.asciidoc)
  with description of new matrix features. [770f540d]
- Moved [`sycl_ext_oneapi_invoke_simd`](doc/extensions/experimental/sycl_ext_oneapi_invoke_simd.asciidoc)
  extensions specification from `proposed` to `experimental` because
  implementation is available. [6bee3440]

## Improvements

### SYCL Library

- Ensured that a correct `errc` thrown for an unassociated placeholder
  accessor. [4f9935ad]
- Removed dependency on OpenCL ICD Loader from the runtime. [90e8b5ef]
- Added support for `ZEBIN` format to persistent caching mechanism. [34dcf83d]
- Added identification mechanism for binaries in newer `ZEBIN` format. [f4dee549]
- Switched to use `struct` information descriptors in accordance with SYCL 2020.
  Removed some deprecated information queries. [b3cbda57]
- Updated `kernel_device_specific::max_sub_group_size` query to match SYCL 2020
  spec. Deprecated the old variant. [7842d056]
- Deprecated SYCL 1.2.1 device selectors. [c058380f]
- Improved error messages reported for unsupported device partitioning. [1c9ddbaa]
- Made `device` and `platform` default to `default_selector_v`. [b32dd41a]
- Deprecated `address_space::constant_space`. [351b123a]
- Marked `sycl::exception::has_context` as `noexcept`. [ad923c9a]
- Improved range reductions performance on CPU. [3323da65]
- Made `sycl::exception` `nothrow` copy constructible. [289e33da]
- Marked `has_property` methods as `noexcept`. [417b5a29]
- Improved `sycl::event::get_profiling_info` exception message when `event` is
  default constructed. [2e86cd459]
- Added a diagnostic (in form of `static_assert`) about kernel lambda size
  mismatch between host and device. [d278c671] [ec179b7a] [f417a88e]
- Updated `pipes` class to throw exceptions if used on host. [eab29696]
- Updated ESIMD Emulator PI plugin to report support for `cl_khr_fp64`
  extension. [398571a5]
- Updated Level Zero plugin to prefer copy engine for memory read/write
  operations. [65c3ea29]
- Optimized some memory transfers. [92d35cd1]
- Enabled event caching in Level Zero PI plugin. [a41b33c3]
- Optimized some reductions with `parallel_for` accepting `sycl::range`
  for discrete GPUs. [c22a5d3f]
- Improved performance of event synchronization on CUDA backend. [c4f326aa]
- Added ability to use descendent devices of context members within that
  context. Not supported with OpenCL backend yet. [a0c8c503] [78a483c7]
- Added support for querying `atomic64` device capability with HIP backend. [cb190fc2]
- Enabled FTZ operations for CUDA/PTX backend via
  `-fcuda-flush-denormals-to-zero`. [e8e7ae83]
- Improved error message about incorrect kernel argument types with CUDA backend. [2542e6a8]
- Limited allowed argument types for `rol/ror` ESIMD functions to better
  represent HW capabilities. [b05f256c]
- Implemented `mem_advise` reset and managed memory checks for CUDA backend. [fe18839c]
- Added concurrent memory check to `mem_advise` on CUDA backend. [33746d8d]
- Enabled multiple HIP streams per SYCL queue. [e0c40a9f]
- Implemented lazy mechanism of setting context for default-constructed events. [ed92c4ca]
- Improved performance for multi-dimensional accessors with multiple accesses
  in a kernel. [7c58b9a6]

### SYCL Compiler

- Increased max `_Bitint` size to 4096 for FPGA target. [db5f72a8] [3f06cad0]
- Removed deprecation message for `[[intel::disable_loop_pipelining]]` attribute. [07201f56]
- Allowed `__builtin_assume_aligned` to be called from device code. [24937eac]
- Improved link step performance when `per_kernel` device code split is used. [84de9d6d]
- Added support for `SYCL_EXTERNAL` on `device_global` variables. [8b958f67]
- Updated `__builtin_intel_fpga_mem` to accept more parameters. [231338dc]
- Updated `ivdep` attribute to allow `safelen = 0`. [558b3ba4]
- Improved linking with `sycl.lib` on Windows. [404d2816]
- Implemented more diagnostics about incorrect `device_global` usages. [12657218]
- Improved library resolution for `libsycl.so`. [4ce19d69]
- Improved diagnostics when linking with mismatched objects. [0e0202ee]
- Added a warning for floating-point size changes after implicit conversions. [e4f5d55f]
- Made `invoke_simd` convert its argument to appropriate types. [038764fd]

### Documentation

- Removed explicit `cl` namespace references. [433ea5c7]
- Added a short guideline on using CMake with SYCL compiler. [fa603c3e]

## Bug fixes

### SYCL Library

- Fixed a compilation issue where it wasn't possible to pass an initializer list
  for dependency events vector in `queue` shortcuts with `offset`
  parameter. [f4f83d95]
- Fixed `sycl::get_pointer_device` throwing an exception when it passed a
  descendent device (sub-device) instead of a root device. [26d5d98b]
- Fixed memory leak happening when kernel bundles are linked. [980677d9]
- Fixed USM free throwing an exception when it passed a context created for
  a descendent device. [c49d4944]
- Fixed `accessor`'s CTAD for `g++` host compiler. [57aabe7e]
- Fixed a compilation issue when using multi-dimensional `accessor`'s subscript
  operator. [22e3fc56]
- Fixed "definition with the same mangled name" error happening when used
  multiple buffer reductions in a kernel. [a0a4d721]
- Fixed a compilation issue with SYCL math built-ins when GCC < 11.1 is used as
  a host compiler. [c786894f]
- Fixed a compilation issue with SYCL math built-ins (such as `sycl::modf`,
  for example) not accepting pointers to `half`. [e2861665]
- Fixed an issues with `reduction`s when MSVC is used as host compiler. [94c4b80a]
- Fixed a compilation issue when fully specialized `sycl::span` is initialized
  from an array. [2b50820b]
- Fixed a crash in Level Zero PI plugins caused by specialization constants not
  being used on device side, but present in a program. [9500875f]
- Fixed event leak in Level Zero plugin. [6d04aa64]
- Fixed an issue with sub-sub-devices in Level Zero plugin. [4b1b01bd]
- Fixed an issue with incorrect `half` conversion on ESIMD emulator. [6143e55a]
- Fixed a compilation issue with `abs` ESIMD function. [c72a85dd]
- Fixed some warning coming out of SYCL headers when compiled in C++20 mode. [12ac4c36]
- Fixed a compilation issue when using multiple bitwise shift operations
  in ESIMD. [40d08c23]
- Fixed a crash in Level Zero PI plugin which occurs when the runtime tries to reset
  a command list which does not have a synchronization fence associated with it. [a61ac7a0]
- Fixed a performance issue with excessive streams synchronization
  on CUDA backend. [5352b423]
- Fixed a compilation issue with
  `sycl::get_native<sycl::backend::ext_oneapi_cuda>(sycl::device)` free
  function (intel/llvm#6653). [4d69c297]
- Fixed synchronization issue for explicit dependencies (`depends_on` usage)
  which is blocked by host task or host accessor. [346a6c53]
- Fixed an issue in Level Zero plugin which could cause barriers to not be
  correctly applied for an entire queue. [d01371b3]
- Fixed a synchronization issue on CUDA backend. [e848c15f]
- Fixed an issue with a context not properly retained with even interop on
  CUDA backend. [2baf1de5]
- Fixed `accessor` so gdb can parse its template parameters correctly. [372cc948]
- Fixed uses of common macro names in the implementation's header files. [e87adfd2]
- Fixed a performance regression in Level Zero backend related to command list. [8a4777d0]

### SYCL Compiler

- Fixed cleanup of temporary files produced by unbundling archives. [7ef4b0a3]
- Fixed optimizing out `device_global` variables with internal linkage. [5b2cfe21]
- Fixed an issue when compiling and linking with different optimization levels
  could cause runtime errors. [0cc7540e]
- Fixed description of `-f[no-]sycl-unnamed-lambda` compiler option. [616ecf75]
- Fixed an issue when building SYCL programs in Debug mode with
  `Windows-Clang.cmake`. [084f34c1]
- Fixed CUDA fat-binaries filename extensions in SYCL toolchain
  (`.o` -> `.cubin`). [2c217cfd]
- Fixed an issue causing incorrect conversions involving unsigned types in
  ESIMD. [2a1151b8]
- Fixed a crash in applications containing a mix of unnamed ESIMD and non-ESIMD
  kernels. [3d54d7ef]
- Fixed type casting of image coordinates in built-ins implementation for CUDA
  backend. [338b4edc]
- Fixed a linking issue when targeting AMD/HIP. [9829897a]
- Fixed an issue when `op[]` was called with typedef argument under gdb. [b7bfe391]

## API/ABI breakages

- Removed deprecated `kernel::get_work_group_info` [94883470]
- Removed deprecated `get_native` class method [282e7744]
- Removed support for `intel::fpga_pipeline` attribute [6fe90d20]
- Added `MAJOR_VERSION` to the name of the SYCL library on Windows. [c8427d6d]
- Removed `sycl::program` class. [85a6833d]
- Removed `ext::oneapi::reduction`. [7fa607fb]
- Removed deprecated `address_space` enum values. [351b123a]
- Removed `event::get` method. [da296ba3]
- Removed `using namespace experimental` inside `ext::intel`. [7ddd8a97]
- Made intel-specific device info descriptors namespace qualified. [0f4a0f3b]
- Removed deprecated `make_queue` API. [26b7762a]
- Aligned return types of `sycl::get_native` and `interop::get_native_mem`
  functions to be conformance with SYCL 2020 spec. [78275902]
- Aligned `sycl::buffer_allocator` interface with SYCL 2020 spec. [78275902]
- Removed `cl` namespace from `sycl/sycl.hpp` header. [3d2b25e1]
- Dropped support for compiling SYCL in less than C++17 mode. [43e713c3]
- Many other ABI-breaking changes resulting from internal refactoring

# Known issues

- This release is not backwards compatible with previous releases, which means
  that existing SYCL applications won't work with the newer runtime without
  re-compilation.
- Having MESA OpenCL implementation which provides no devices on a
  system may cause incorrect device discovery. As a workaround such an OpenCL
  implementation can be disabled by removing `/etc/OpenCL/vendor/mesa.icd`.
- Compilation may fail on Windows in debug mode if a kernel uses
  `std::array`.  This happens because debug version of `std::array` in
  Microsoft STL C++ headers calls functions that are illegal for the device
  code. As a workaround the following can be done:
   1. Dump compiler pipeline execution strings by passing `-###` option to the
      compiler. The compiler will print the internal execution strings of
      compilation tools. The actual compilation will not happen.
   2. Modify the (usually) first execution string (it should have
      `-fsycl-is-device` option) by adding
      `-D_CONTAINER_DEBUG_LEVEL=0 -D_ITERATOR_DEBUG_LEVEL=0` options to the
      end of the string.  Execute all string one by one.
- `-fsycl-dead-args-optimization` can't help eliminate offset of
  accessor even though it's created with no offset specified
- SYCL 2020 barriers show worse performance than SYCL 1.2.1 do. [18c80faa]
- When using fallback assert in separate compilation flow it requires explicit
  linking against `lib/libsycl-fallback-cassert.o` or
  `lib/libsycl-fallback-cassert.spv`
- Limit alignment of allocation requests at 64KB which is the only alignment
  supported by Level Zero. 7dfaf3bd
- On the following scenario on Level Zero backend:
   1. Kernel A, which uses buffer A, is submitted to queue A.
   2. Kernel B, which uses buffer B, is submitted to queue B.
   3. `queueA.wait()`.
   4. `queueB.wait()`.
  DPCPP runtime used to treat unmap/write commands for buffer A/B as host
  dependencies (i.e. they were waited for prior to enqueueing any command
  that's dependent on them). This allowed Level Zero plugin to detect that
  each queue is idle on steps 1/2 and submit the command list right away.
  This is no longer the case since we started passing these dependencies in an
  event waitlist and Level Zero plugin attempts to batch these commands, so
  the execution of kernel B starts only on step 4. The workaround restores the
  old behavior in this case until this is resolved. [2023e10d] [6c137f87]
- User-defined functions with the name and signature matching those of any
  OpenCL C built-in function (i.e. an exact match of arguments, return type
  doesn't matter) can lead to Undefined Behavior.
- A DPC++ system that has FPGAs installed does not support multi-process
  execution. Creating a context opens the device associated with the context
  and places a lock on it for that process. No other process may use that
  device. Some queries about the device through `device.get_info<>()` also
  open up the device and lock it to that process since the runtime needs
  to query the actual device to obtain that information.
- The format of the object files produced by the compiler can change between
  versions. The workaround is to rebuild the application.
- Using `sycl::kernel_bundle` API to refer to a kernel defined
  in another translation unit leads to undefined behavior
- Linkage errors with the following message:
  `error LNK2005: "bool const std::_Is_integral<bool>" (??$_Is_integral@_N@std@@3_NB) already defined`
  can happen when a SYCL application is built using MS Visual Studio 2019
  version below 16.3.0 and user specifies `-std=c++14` or `/std:c++14`.
- Printing internal defines isn't supported on Windows. [50628db1]

# June'22 release notes

Release notes for commit range f34ba2c..4043dda

## New features
### SYCL Compiler
  - Added `-fcuda-prec-sqrt` frontend compiler option which enables higher presision version of `sqrt` in the device code [ebf9ea8]
  - Added support for local memory accessors for the HIP backend. [58508ba]
  - Added initial support of `-lname` processing when searching for fat static libraries. [35e32d8] [a33f9c8]
  - Added `-fsycl-fp32-prec-sqrt` flag which enables correctly rounded `sycl::sqrt`. [5c8b7e7]
  - Added support for `[[intel::loop_count()]]` attribute. [c536e76]
  - Added support for passing driver options to JIT compiler and linker. [1c93bfe]
  - Added default argument support for `work_group_size_hint` attribute. [0cff80e]
  - Added support for float and double exchange and compare exchange atomic operations in CUDA libclc. [1d84c99]
  - Added `--ffast-math` support for CUDA libclc. [0f0c5d1]
  - Added support for software atomics (except for the ones using system scope) for lower sm versions of CUDA architecture. Enabled `SYCL_USE_NATIVE_FP_ATOMICS` by default. [7bc8447]
  - Added support for the global offset for AMDGPU. [2dc3c06]
  - Added support for asynchronous barrier for CUDA backend sm 80+. [6770421]
  - Added `-f[no-]sycl-device-lib-jit-link` option to control JIT linking of SYCL device libraries. [dfb37a8] [c946286]
  - Added support for the new FPGA attribute `[[intel::fpga_pipeline(N)]]` for loop pipelining. [92aadf3]
  - Added `assert` support for Windows NVPTX. [f29b498]
  - Added support for [`sycl_ext_oneapi_properties` extension](doc/extensions/experimental/sycl_ext_oneapi_properties.asciidoc). [87f60f6][1984e74][a2583ec][cdf561a][d2982c6][35c2e00]

### SYCL Library
  - Added support for Nvidia MMA for `bf16`, mixed precision int `((u)int8/int32)`, and mixed precision float `(half/float)`. [5373362]
  - Added a mode for the Level Zero plugin where only last command in each batch yields a host-visible event. Enabled this mode by default. [c6b7b8e]
  - Added an option to query for atomic scope capabilities for the CUDA backend. Updated returns for atomics memory order capabilties. [43a4192]
  - Added support for an experimental Level Zero API for host pointer import into USM. The feature can be enabled using `SYCL_USM_HOSTPTR_IMPORT` environment variable. [844d7b6]
  - Added support for the `wi_element` for `bf16` type. [9f2b7bd]
  - Added complex support for the reduce and scan group algorithms. [90a4dc7]
  - Added support for SYCL 2020 methods in the `group` class.  [73d59ce]
  - Added `SYCL_RT_WARNING_LEVEL` environment variable which allows to control amount of warnings and performance hints the runtime library may print. [2741010]
  - Added `tanh` (for floats/halfs) and `exp2` (for halfs) native definitions for CUDA backend. [250c498]
  - Added `bf16` builtins for `fma`, `fmin`, `fmax` and `fmax` on CUDA backend. [62651dd]
  - Added support for USM buffer location properties which allows to specify at what memory location the device usm allocation should be in. [12c988a]
  - Added support for `buffer_location` property to the `sycl::buffer`. [9808525]
  - Added `single_task` support for ESIMD_EMULATOR backend. [2331160]
  - Added support for SVM 1,2,4-elements gather/scatter for ESIMD. [e200720]
  - Added support for `bf16` builtins operating on storage types for CUDA backend. [413a9ef]
  - Added support for `backend_version` device property for CUDA backend. [4b1a4bc]
  - Added support for round-robin submissions to multiple compute CCS for the Level Zero backend. Disabled by default, can be controlled using `SYCL_PI_LEVEL_ZERO_USE_COMPUTE_ENGINE`. [a836c87]
  - Added support for buffer migration for contexts with multiple devices in the Level Zero plugin. [7baf152]
  - Added mode where the Level Zero plugin uses immediate command-lists instead of standard command-lists. This mode is disabled by default, can be enabled using `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS` environment variable. [b9cb1d1]
  - Added support for `sycl::get_native(sycl::buffer)` for OpenCL and CUDA backends. [8b3c8c4]
  - Added reduction overloads accepting `span`. [863383b]
  - Added LSC support for ESIMD_EMULATOR backend. [b78bf00]
  - Added `half` type support for `__esimd_convertvector_to/from`. [0bfffd6]
  - Added `buffer_allocator` SYCL 2020 conformant variant. [53430c8]
  - Added support for the USM buffer location property in `malloc_shared`. [6e89821] [9f61c8e][8c4d9a5]
  - Added support for the USM buffer location property in `malloc_host`. [2c7caab]
  - Added experimental context and device interoperability support for CUDA. [f0df89a]
  - Added support for memory intrinsics for the ESIMD_EMULATOR plugin. [1a8f501]
  - Added support for named barrier APIs for ESIMD. [1df0038]
  - Added support for DPAS API for ESIMD. [5881938]
  - Added support for LSC memory access APIs for ESIMD. [4bd50e7]
  - Added support for the `invoke_simd` feature. [4072557][8471ff3][8c7bb45][62afb59][3e1c1bf]
  - Added support for `info::device::atomic64` for OpenCL and Level Zero backends. [8feb558]
  - Added support for [`sycl_ext_oneapi_usm_device_read_only` extension](doc/extensions/supported/sycl_ext_oneapi_usm_device_read_only.asciidoc) [644c614][58c9d3a]
  - Added support for mapping/unmapping operations for ESIMD_EMULATOR plugin. [bc0579a]
  - Added support for `make_buffer` API for the Level Zero backend. [7c49984]
  - Added interoperability support for HIP backend. [e06d1b5]
  - Added missing `+-*/` operations for `half`. [059efbc]
  - Introduced new environment variable `SYCL_PI_CUDA_MAX_LOCAL_MEM_SZ` to control the max local memory allowed to be allocated per kernel on CUDA backend. [2e24304]
  - Added `ext_intel_global_host_space` in accordance with [`sycl_ext_intel_usm_address_spaces` extension](doc/extensions/supported/sycl_ext_intel_usm_address_spaces.asciidoc). [7a2f44b]
  - Added aspect for `bfloat16`. [f84fc32]
  - Introduced "Intel math functions" device library with support of type cast util functions for float, double and integer type. [a310952]
  - Added `bfloat16` support for `joint_matrix` [6ac62ab]

### Documentation
  - Added [`sycl_ext_oneapi_complex_algorithms` extension](doc/extensions/proposed/sycl_ext_oneapi_complex_algorithms.asciidoc) [7ae7ca8]
  - Added a design document for [`sycl_ext_oneapi_device_global` extension](doc/design/DeviceGlobal.md) [8c22ef1]
  - Added a design document for [`sycl_ext_oneapi_properties` extension](doc/design/CompileTimeProperties.md) [912572f]
  - Added new [`sycl_ext_oneapi_free_function_queries` proposal](doc/extensions/proposed/sycl_ext_oneapi_free_function_queries.asciidoc). [7a93a49]
  - Added [`sycl_ext_oneapi_group_load_store` extension](doc/extensions/experimental/sycl_ext_oneapi_group_load_store.asciidoc). [85ccdc0]
  - Added validation rules to the SPIR-V extension [`SPV_INTEL_global_variable_decorations`](doc/design/spirv-extensions/SPV_INTEL_global_variable_decorations.asciidoc). [dfaa070]
  - Added [`SYCL_INTEL_buffer_location` extension](doc/extensions/supported/sycl_ext_intel_buffer_location.asciidoc) to support `buffer_location` property for USM allocations. [962417d] [36a9ee2]
  - Added [`sycl_ext_oneapi_named_sub_group_sizes` extension](doc/extensions/proposed/sycl_ext_oneapi_named_sub_group_sizes.asciidoc) proposal which aims to simplify the process of using sub-groups. [4f3d7e1]
  - Added experimental latency control API into [`SYCL_INTEL_data_flow_pipes`](doc/extensions/supported/sycl_ext_intel_dataflow_pipes.asciidoc). [5224f78]
  - Added [`sycl_ext_oneapi_auto_local_range` extension](doc/extensions/proposed/sycl_ext_oneapi_auto_local_range.asciidoc) proposal. [cb4e702]
  - Added [SYCL 2020 spec constants](doc/design/SYCL2020-SpecializationConstants.md) design doc. [8ec9755]
  - Added [`sycl_ext_oneapi_queue_status_query` extension](doc/extensions/proposed/sycl_ext_oneapi_queue_status_query.asciidoc) proposal. [b6143e5]
  - Added initial version of [`sycl_ext_oneapi_invoke_simd`](doc/extensions/proposed/sycl_ext_oneapi_invoke_simd.asciidoc) and [`sycl_ext_oneapi_uniform`](doc/extensions/proposed/sycl_ext_oneapi_uniform.asciidoc) extenions proposal. [a37ca84]
  - Added the [`sycl_ext_oneapi_annotated_arg` extension](doc/extensions/proposed/sycl_ext_oneapi_annotated_arg.asciidoc) proposal for applying properties on kernel arguments. [caa696f]
  - Added [`sycl_ext_oneapi_cuda_async_barrier` extension](doc/extensions/experimental/sycl_ext_oneapi_cuda_async_barrier.asciidoc) for CUDA backend. [6770421]
  - Added `bfloat16` support to the `fma`, `fmin`, `fmax` and `fabs` SYCL floating point math functions into [`sycl_ext_oneapi_bfloat16` extension](doc/extensions/experimental/sycl_ext_oneapi_bfloat16.asciidoc). [c76ef5c]
  - Added initial version of [`sycl_ext_oneapi_root_group` extension](doc/extensions/proposed/sycl_ext_oneapi_root_group.asciidoc) proposal. [b59cd43]

### Tools
  - Implemented property set generation for device globals in the sycl-post-link. Added the `--device-globals` command-line argument for lowering and generating information about device global variables. [88123c1]
  - Introduced XPTI-based tools for SYCL applications: sycl-trace, sycl-prof, sycl-sanitize. [789d138]
  - Add support for tracing Level Zero API calls using XPTI and updated sycl-trace tool to be able to display both PI and Level Zero calls simultaneously. [fc9cf52]

## Improvements
### SYCL Compiler
  - Added a diagnostic on attempt to use zero length arrays in the device code [52e8f58]
  - Added support for consuming fat objects containing SPIR-V [c878063]
  - Added support for generating SPRIV based fat objects [1e94ef3]
  - Added support for group collective functions for HIP backend [106882c]
  - Added a diagnostic on attempt to use `-fsycl` and `-static-libstdc++` together. This combination is not supported due to the runtime dependence with libsycl.so [bb0055c]
  - Added support for atomic loads and stores with various memory orders and scopes [e15ac50] [6b2635e]
  - Improved performance of accessing memory pointed by `sycl::accessor` for FPGA device [fbab374]
  - Added support for setting CUDA installation directory using `CUDA_PATH` environment variable [b0c145a]
  - Improved deferred diagnostics for usages within function templates in device code. [d0efca5]
  - Added support for `sycl_special_class` attribute to mark SYCL classes/struct that need the additional compiler handling. [8ba9c79]
  - Improved driver to do device section checking only when offloading is enabled. [3742b93]
  - Allowed calls to constant expression function pointers in device code. [e84c952]
  - Disabled the passing code coverage and profiling options to device compilation.
  - Added clang support of code location information for kernels. [96d2e17]
  - Disallowed explicit casts between mismatching address spaces. [1cee960]
  - Added support of `[[sycl::device_has]]` attribute on kernel. [aa2162c]
  - Added a warning on explicit cast from default address space to named. [9adb25b]
  - Added a warning for converting 'c' input to 'c++' in SYCL mode. [5b62ee0]
  - Silenced unknown attribute warnings on host compilation. [2d359df]
  - Added a diagnostic on attemp to use `accessor::operator[]` in ESIMD code. [9d7a651]
  - Expanded driver's ability to discover fat static archives after `/link` option on Windows. [271ef40]
  - Added support for saving user specified names for lambda captures in kernel lambda object for FPGA target. [af29982] [5ffb2ee]
  - Adjusted the compilation when preprocessing is requested to allow for the device compilation to fail and continue to perform the preprocessing steps. [7f2e99c]
  - Added the ability to detect a kernel size mismatch in the case when the host and device compilers are different. [ef90e6a]
  - Improved handling of specialization constants by backends. [e62b5aa]
  - Improved support of `-mlong-double` options. [6083920]
  - Improved `-save-temps` to allow optimization when performing a SYCL device compilation. [05fe5ae]
  - Removed warning diagnostic on host compilation when using `__attribute__((sycl_device))`. [49e595e]
  - Introduceed multiple streams in each queue for CUDA backend to improve concurrent execution. [dd41845]
  - Improved compiler to collect information for optimization record only if optmization record is saved by user (i.e. `-fsave-optimization-record` or `-opt-record-file` is passed). [cb94c80]
  - Improved `[[intel::max_concurrency()]]` attribute support. [b6aa4cc]
  - Added the new `kernel_arg_exclusive_ptr` metadata which guarantees that the kernel pointer argument, or pointers that derive from it, will not be dereferenced outside current invocation of the kernel. [e03c4ed]
  - Added a warning for the case when invalid subgroup size is used on kernel for CUDA backend. [6dab69f]
  - Improved deprecation messaging for options. [f0b65a1]
  - Improved diagnostic behavior for `-fsanitize` with `-fsycl`. [9397cbc]

### SYCL Library
  - Added support for `sycl::ctz` API [d5eb769]
  - Improved the diagnostic for unresolved symbols in the device code for Level Zero backend [33cfb9f]
  - Added several arithmetic operations for `sycl::ext::oneapi::experimental::wi_element` [fe2e35e]
  - Added support for `sycl::queue::submit_barrier` for HIP backend [53f1cce]
  - Added `sycl::property_list` APIs  to `sycl::stream` [4b39793]
  - Defined `sycl::access::decorated` in the SYCL headers. [d183807]
  - Improved performance by allowing batching for wait with a barrier commands for Level Zero backend [bd1ed6a]
  - Disabled image support for CUDA backend since the support is not complete [90c8f05]
  - Avoided JITing unnecessary device code when using `sycl::handler::set_specialization_constant` [5746906]
  - Updated image accessor constructor to make it possible to use const references in parallel_for. [3070b95]
  - Relax the mutex lock duration in queue finish for the Level Zero backend to allow working with the queue from other threads. [8573935]
  - Added XPTI instrumentation for USM allocations. [7df3923]
  - Extended XPTI information with buffer constructor data. [fc0d28a]
  - Added error handling for `sycl::event::get_profiling_info()`. [ec74a5c]
  - Eliminated recursion and duplicated dependencies in leaf buffers handling in the scheduler. [6f40376]
  - Improved runtime to emit program build logs when `SYCL_RT_WARNING_LEVEL` is set to 2 or higher. [008519a]
  - Improved the error message at command execution failure. [930ecbf]
  - Improved runtime to build program for root device only and re-use the binary for sub-devices to avoid "duplicate" builds. [13a7455]
  - Improved `sycl::kernel::get_kernel_bundle` performance. [4817b3f]
  - Changed USM pooling parameters for the Level Zero backend to boost performance. [57f8a44][b000db8]
  - Exposed `value_type` and `min_capacity` from SYCL pipes extension class. [e1619fa]
  - Improved thread-safety of the Level Zero plugin by guarding access to the PI objects. [3321141] [a37c10b] [bd80f34] [8f97fe2]
  - Optimized half builtins for `fma`, `fmin`, `fmax` and `fmax` on CUDA backend. [62651dd]
  - Improved runtime to redirect warning from using `SYCL_DEVICE_FILTER` with `sycl-ls` to `std::cerr`. [70593d6]
  - Use new SPIR-V group operations within uniform control flow instead of non-uniform operations in SYCL headers. [9b84dd8]
  - Enabled online linking of the device libraries. [9fcab29]
  - Improved esimd-verifier logic for detecting valid SYCL calls. [eaf8b42]
  - Extended XPTI information with the kernel info. [4b9eef3]
  - Improved error message for exceeding CUDA grid limits. [ed877c2]
  - Added overload for `sycl::select(a, b, c)` where `c` is a bool. [7ae8fd3]
  - Fixed batching related thresholds to improve performance. [c6313bd]
  - Added always_inline for libdevice functions to enable which allows an underlying runtime to do inlining. [dfc87cc]
  - Improved performance by caching the result of `zeKernelGetName` in the Level Zero plugin. [40cece3]
  - Updated the experimental latency control API to use property list and made the template argument approach is deprecated. [273034a]
  - Renewed and synced the `queue::parallel_for()` with SYCL2020 . [e59fb89]
  - Improved runtime to ignore CUDA prefetch hint if not supported on the system and emit an optional warning in this case depending on warning level set using `SYCL_RT_WARNING_LEVEL`. [082929a]
  - Enabled pooling of small USM allocations for the Level Zero backend to improve performance. [6244efe]
  - Added managed memory check to enqueue prefetch and made it to ignore the prefetch hint and emit a warning if the memory provided is not managed. [0fe322c]
  - Enabled device code instrumentation by default. [53fc8e4]
  - Improveed performance of `queue::wait()` on CUDA backend. [8b85a3c]
  - Optimized host event wait. [f3a0970]
  - Improved default selector to filter devices based on available device images. [0e67db8]
  - Enabled caching of native OpenCL and Level Zero executable binaries. [f0283fc]
  - Deprecated `sycl::ext::intel::ctz` extension function,`sycl::ctz` from the core specification must be used instead. [50435a6]
  - Deprecated `cl::sycl::atomic` in SYCL 2020 mode. [52fad5a]
  - Deprecated `ext_intel_host_device_space` which is replaced by `ext_intel_global_host_space`. [7a2f44b]
  - Added CUDA-specific USM memory advice values. [e2e71a9]

### Tools
  - Added an option `--enable-esimd-emulator` to enable esimd emulator build using configure.py. [ddbcbb1]
  - Added an ability to build plugins separately. [c16412b]
  - Added `--enable-all-llvm-targets` switch to configure.py. [5e6642a]
  - Added PI tracing support for `loadOsLibrary`. [647249c]

### Documentation
  - Clarified the interaction between the [`sycl_ext_oneapi_invoke_simd` extension](doc/extensions/proposed/sycl_ext_oneapi_invoke_simd.asciidoc) and `SYCL_EXTERNAL` functions. [ac3e816]
  - Removed extensions specifications which were adopted to SYCL 2020. Please refer to [extensions/removed/README](doc/extensions/removed/README.md) for the list of removed extensions. [ae7e3d8] [57c9017]
  - Structured sycl/doc directory:
    - Moved the supported extensions to the [sycl/doc/extensions/supported](doc/extensions/supported). [2521592]
    - Moved the experimental extensions to the [sycl/doc/extensions/experimental](doc/extensions/experimental). [eae965c]
    - Moved the proposed extensions to the [sycl/doc/extensions/proposed](doc/extensions/proposed). [eb350ee]
    - Moved the deprecated extensions to the [sycl/doc/extensions/deprecated](doc/extensions/deprecated). [09be931]
    - Moved internal design documentations and BKMs to [sycl/doc/developer](doc/developer). [edbfc99]
    - Moved SPIR-V and OpenCL extensions to [doc/design](doc/design). [64e92cb]
  - Updated [CompileTimeProperties](doc/design/CompileTimeProperties.md) design documentation. [2359e81][f6420c7]
  - Clarified which SPIR-V decorations the `sycl-post-link` tool generates for each device global variable. [3c3b485]
  - Updated the design for device global variables for variables that are "shadowed" in an unnamed namespace. [e211d73]
  - Clarified the specification that device global with `SYCL_EXTERNAL` is allowed. [0d95d6f]
  - Add an overview README for the extensions directory. [d54708a]
  - Clarified the C++ attribute `[[sycl_detail::uses_aspects()]]` usage in the documentation. [27cc930]
  - Added a template document to use when creating new SYCL extension specifications. [4bd6d20]
  - Added new rule for naming C++ identifiers in the SYCL project. [1445528]
  - Updated [GettingStartedGuide](doc/GetStartedGuide.md) to recommend cuda 11.6. [0b456ce]
  - Added ESIMD_EMULATOR to `SYCL_DEVICE_FILTER` description. [526ad0c]
  - Clarified availability of `get_property()`.  [a2c5e90]
  - Deprecated extended atomics extension. [7581741]
  - Improved [GettingStartedGuide](doc/GetStartedGuide.md) for non-standard HIP installations. [09c3b46]
  - Added description of ESIMD_EMULATOR backend to [sycl_ext_intel_esimd/README](doc/extensions/experimental/sycl_ext_intel_esimd/README.md). [b8e6d23]

# Bug fixes
### SYCL Compiler
  - Fixed a crash happened if an overloaded `new` operator is used in a recursive function in the device code [2085978] [2085978]
  - Fixed wrong address space of `event_t` which could lead to builtins like barriers work incorrectly when using HIP backend. [22532c2]
  - Fixed an issue with certain macros being unavailable when using a custom host compiler. [652417b]
  - Fixed an issue with device code linking when one of the targets is not spir64 based. [1f8874f]
  - Disabled part of SimplifyCFG optimizations in SYCL mode which was resuling in invalid optimizations in some cases. [8b29220]
  - Silenced "unknown attribute" warning emitted during host part of full `-fsycl` compilation when it saw `[[intel::device_indirectly_callable]]` attribute. [718c0b1]
  - Removed incorrect assertion for use of `-fopenmp-new-driver` for multiple inputs. [6e0f6d1]
  - Fixed the issue caused by using the `nvvm_reflect` function in the nvptx backend with `-O0`. [537e51b]
  - Fixed a regression in cases where function pointers were captured as kernel arguments. [b19e2e4]
  - Fixed the libclc remangler to clone functions rather than aliasing to enable DPC++ for CUDA with `-O0`. [7b2fb02]
  - Fixed the error "Explicit load/store type does not match pointee type of pointer operand" caused by incorrect address space. [e688fa5]
  - Fixed incorrect diagnostic for `__spirv` calls when the `reqd_sub_group_size` attribute is applied on a sycl kernel. [1df7b59]
  - Fixed alignment of emulated specialization constants. [0cec3c6]
  - Fixed the group collective implementation for AMDGCN. Fixed the `shuffleUp` and `shuffleDown` functions for the AMDGCN builtins and SYCL headers. [d99e957]
  - Removed `llvm.nvvm.suq.depth` instruction which was causing `CUDA_ERROR_NOT_FOUND` or `CUDA_ERROR_NOT_SUPPORTED` errors if present in the fatbin. [ec29322]

### SYCL Library
  - Fixed alignment of kernel local arguments in the CUDA backend. [ebb1281]
  - Fixed a crash which could happen when bulding program for multiple devices. [64c2d35]
  - Fixed max constant value query for the HIP backend. [1e55cf3]
  - Fixed ambiguity error with `sycl::oneapi::experimental::this_nd_item`. [8aad52dd6027]
  - Fixed a performance issue caused by unnecessary command batching in the Level Zero plugin. [4d031a4]
  - Fixed an issue with `sycl::get_pointer_device` API working incorrectly for CUDA and HIP backends [8fa17b4]
  - Fixed an issue which might result in JITing for only one device while context is accosiated with multiple devices for Level Zero backend [7068457]
  - Fixed namespace ambiguity in `this_id`, `this_item`, and `this_group`. [19369b6]
  - Workarounded two bugs in the Level Zero driver related to static linking extension. [2930a94]
  - Fixed return type of `get_nativesycl::backend::opencl(event)` from `cl_event` to `vector<cl_event>`. [a2189c6]
  - Modified Level Zero plugin support for copy engines to address scenario when main copy engine is not available. [478a576]
  - Fixed support for query of USM capabilities. [5941394]
  - Fixed memory leak in the USM prefetch functionality. [5d4573f]
  - Fixed host device local accessor alignment. [08b14da]
  - Fixed `sycl::errc values` for exceptions per SYCL 2020. [270e78d]
  - Fixed bug with `constexpr_recurse` usage. [bd15de9]
  - Fixed `max_work_group_size` and `reqd_work_group_size` attribute arguments check. [7f37250]
  - Fixed iterator debug level mistmatch error on Widnows when building programs with `/MDd` when `libsycl-fallback-cassert.obj` is involved. [93b573a]
  - Fixed `get_native()` for `sycl::event` per requirements of the specification. [8878962]
  - Fixed device enumeration for the next platforms when current platform doesn't have devices. [0272ec2]
  - Fixed thread-safety issue in the scheduler which can appear if command gets cleaned up by another thread while adding a host accessor. [62ca43a]
  - Fixed reported device name for HIP backend. [68b089f]
  - Fixed `SYCL_PROGRAM_COMPILE_OPTIONS` and `SYCL_PROGRAM_LINK_OPTIONS` to override compile and link options respectively. [7a8fa1a]
  - Fixed incorrect handling of queue indexing for Level Zero backend. [1f4c9df]
  - Fixed memory leak in the reductions that require additional resources (such as buffers) [9aefea0]
  - Defined `get_property/has_property` in the queue for `property::queue::in_order`. [ca9fea6]
  - Fixed memory leak in the scheduler for `run_on_host_intel` commands. [86cf56a]
  - Fixed thread-safety issue caused by parallel access to the command list cache in the Level Zero plugin. [1f531c0]
  - Fixed device code outlining for static local variables to avoid invalid device code generation. [66e207e]
  - Fixed dynamic batching in the Level Zero plugin. [d6f115c]
  - Fixed unsigned long warning in fallback cstring on Windows. [5fbe02b]
  - Fixed sync of host task vs kernel for in-order queue. [c7ba937]
  - Fixed include dependency in `fpga_lsu.hpp` and `pipes.hpp` headers. [ac6a4f5]
  - Fixed kernel timestamp calculation in the Level Zero plugin. [c228f12] [7efb3e6] [76a3898]
  - Fixed usage of copy-engines in the Level Zero interoperability queue. [847f8b6]
  - Fixed kernel execution hang under large memory consumption by workarounding a bug in the Level Zero runtime. [b831bd0]
  - Fixed the Level Zero plugin to honor `property::queue::enable_profiling`. [fb27c65]
  - Fixed memory leak which existed when program build failed for the Level Zero backend. [beb7277]
  - Fixed buffer creation from rvalue iterator. [a905a27]
  - Fixed `queue::device_has()` to private. [279ef0d]
  - Fixed crash for case when a device image has no kernels. [279ef89]
  - Fixed dependency between host/device actions for unbundled FPGA specific archives. [9699575]
  - Fixed PI CUDA plugin to avoid linking against libsycl which may cause issues on some systems. [6821e66]
  - Fixed support of `bfloat16` for CUDA. [5231fe4]
  - Fixed `interop_handle::get_native_mem` so that it can work with accessors that use non-empty accessor_property_list. [5452a5d]
  - Fixed sub-device count calculation for numa partitioning. [ae284f1]
  - Fixed `SYCL_ENABLE_PLUGINS` to enable both the OpenCL and the Level Zero PI plugins if it is unset. [ff384bb]
  - Fixed BDF format on PCI query for the Level Zero backend. [ad6253e]
  - Fixed `sycl::queue` XPTI instrumentation. [ec57cd7]
  - Fixed interoperability return type for `sycl::buffer` to `std::vector<cl_mem>` per SYCL 2020. [8b3c8c4]
  - Fixed `SYCL_DUMP_IMAGES` handling to also dump when spec constants are on. [211ccda]
  - Fixed failure in case of using zero-size local accessor on some backends. [1292532]
  - Fixed flaky bug which might appear in multi-threaded applications with simultaneous access to the cache of device lib programs. [92cfd53]
  - Fixed compfail issue with `-ffast-math` on CUDA backend. [90ac3ee]
  - Fixed make_queue interoperability API for the Level Zero to accept device argument to properly associate queue with the right device. [29a5369]
  - Fixed invalid handler issue by updating OpenCL ICD loader from community.
  - Fixed "undefined symbol" error for `ldexpf`, `hypotf`, `frexpf` on SYCL GPU device when using 3rd-party math headers instead of MSVC math headers on Windows.[476a351]
  - Fixed memory leak for interop events created from native handle. [3c1d342]
  - Fixed alignment of the memory returned from USM allocation functions. [3114f02]
  - Fixed Vendor ID for AMD devices. [a1b42aa]
  - Fixed sporadic failure of in-order queue due to non-closed batch on the Level Zero backend. [e8bff05]
  - Fixed possible deadlock in case of having dependent events from different queues in a multi-threaded application. [4c619e9]
  - Fixed issue with delivery of assert message before abort'ing [6a32706]
  - Fixed default value for the `Alignment` template parameter of the usm_allocator. [3a91cec]
  - Fixed API to get maximum width/height/depth of an image for the Level Zero backend. [9ecc74b]
  - Fixed runtime for CUDA backend to ignore usm mem advise and warn if device doesn't meet requirements.  [ccaaa99]

### Tools
  - Fixed sycl-post-link tool to properly handle the offset in specialization constant descriptors. [12d7c1f]
  - Fixed sycl-post-link tool to properly handle the padding at the end of composite types. [12d7c1f]
  - Fixed translation of `Vector[Extract/Insert]Dynamic` instructions in llvm-spirv [bf43d7b]
  - Fixed unconditional debug info generation for `libsycl_profiler_collector.so`  [45784cd]
  - Fixed sycl-post-link failure caused by incorrect removal of `llvm.used` in the case when specialization constant has 2+ users. [80e9148]

### Documentation
  - Removed documentation for `[[intel::reqd_work_group_size]]`. [28ffda1]
  - Removed extension to set kernel cache configuration. [159a516]
  - Disallowed `[[sycl_detail::uses_aspects()]]` attribute on type aliases in [OptionalDeviceFeatures](doc/design/OptionalDeviceFeatures.md). [df83271]
  - Moved `properties` and property-related APIs into `sycl::ext::oneapi::experimental`. [`sycl_ext_oneapi_properties`](doc/extensions/experimental/sycl_ext_oneapi_properties.asciidoc) specification was updated to revision 2. [33fdc58][aacf541]
  - Updated [`sycl_ext_oneapi_kernel_properties` extension](doc/extensions/proposed/sycl_ext_oneapi_kernel_properties.asciidoc). 
  - Fixed documentation of `__builtin_sycl_unique_stable_id` in [clang/docs/LanguageExtensions](https://github.com/intel/llvm/blob/sycl/clang/docs/LanguageExtensions.rst). [72ca49c]
  - Aligned [`sycl_ext_intel_kernel_args_restrict` extension](doc/extensions/supported/sycl_ext_intel_kernel_args_restrict.asciidoc) extension with SYCL 2020.  [4a794df]

# API/ABI breakages
  - Removed deprecated API from ESIMD headers. [ec0385d]
  - Renamed `wi_slice` to `wi_data`. [f364e18]
  - Renamed `nbarrier_*` API to `named_barrier_*` for ESIMD. [5023657]
  - Moved a part of ESIMD APIs outside of experimental namespace. [c557d78][b2ee289]
  - Moved `bfloat16` from `intel` namespace to `oneapi` namespace. [5231fe4]

# Known issues
 - Having MESA OpenCL implementation which provides no devices on a
   system may cause incorrect device discovery. As a workaround such an OpenCL
   implementation can be disabled by removing `/etc/OpenCL/vendor/mesa.icd`.
 - Compilation may fail on Windows in debug mode if a kernel uses
   `std::array`.  This happens because debug version of `std::array` in
   Microsoft STL C++ headers calls functions that are illegal for the device
   code. As a workaround the following can be done:
    1. Dump compiler pipeline execution strings by passing `-###` option to the
       compiler. The compiler will print the internal execution strings of
       compilation tools. The actual compilation will not happen.
    2. Modify the (usually) first execution string (it should have
       `-fsycl-is-device` option) by adding
       `-D_CONTAINER_DEBUG_LEVEL=0 -D_ITERATOR_DEBUG_LEVEL=0` options to the
       end of the string.  Execute all string one by one.
 - `-fsycl-dead-args-optimization` can't help eliminate offset of
   accessor even though it's created with no offset specified
 - SYCL 2020 barriers show worse performance than SYCL 1.2.1 do [18c80fa]
 - When using fallback assert in separate compilation flow it requires explicit
   linking against `lib/libsycl-fallback-cassert.o` or
   `lib/libsycl-fallback-cassert.spv`
 - Limit alignment of allocation requests at 64KB which is the only alignment
   supported by Level Zero[7dfaf3bd]
 - On the following scenario on Level Zero backend:
    1. Kernel A, which uses buffer A, is submitted to queue A.
    2. Kernel B, which uses buffer B, is submitted to queue B.
    3. `queueA.wait()`.
    4. `queueB.wait()`.
   DPCPP runtime used to treat unmap/write commands for buffer A/B as host
   dependencies (i.e. they were waited for prior to enqueueing any command
   that's dependent on them). This allowed Level Zero plugin to detect that
   each queue is idle on steps 1/2 and submit the command list right away.
   This is no longer the case since we started passing these dependencies in an
   event waitlist and Level Zero plugin attempts to batch these commands, so
   the execution of kernel B starts only on step 4. The workaround restores the
   old behavior in this case until this is resolved [2023e10d][6c137f87].
 - User-defined functions with the name and signature matching those of any
   OpenCL C built-in function (i.e. an exact match of arguments, return type
   doesn't matter) can lead to Undefined Behavior.
 - A DPC++ system that has FPGAs installed does not support multi-process
   execution. Creating a context opens the device associated with the context
   and places a lock on it for that process. No other process may use that
   device. Some queries about the device through device.get_info<>() also
   open up the device and lock it to that process since the runtime needs
   to query the actual device to obtain that information.
 - The format of the object files produced by the compiler can change between
   versions. The workaround is to rebuild the application.
 - Using `sycl::program`/`sycl::kernel_bundle` API to refer to a kernel defined
   in another translation unit leads to undefined behavior
 - Linkage errors with the following message:
   `error LNK2005: "bool const std::_Is_integral<bool>" (??$_Is_integral@_N@std@@3_NB) already defined`
   can happen when a SYCL application is built using MS Visual Studio 2019
   version below 16.3.0 and user specifies `-std=c++14` or `/std:c++14`.
 - Printing internal defines isn't supported on Windows [50628db]

# December'21 release notes

Release notes for commit range 23ca0c2..27f59d8

## New features
### SYCL Compiler
 - Added support for `-fgpu-inline-threshold` which allows controlling inline
   threshold of the SYCL device code [5f7b607]
 - Added experimental support for CUDA backend on Windows [8aa3513]
 - Added support for experimental option `-fsycl-max-parallel-link-jobs=<N>`
   which can be used specify how many processes the compiler can use for
   linking the device code [c2221f0]
### SYCL Library
 - Added support for [default context extension](doc/extensions/supported/sycl_ext_oneapi_default_context.asciidoc)
   on Linux [315593d]
 - Added experimental support for [group sorting algorithm](doc/extensions/experimental/sycl_ext_oneapi_group_sort.asciidoc)
   [932ae56]
 - Added support for [sub-group mask extension](doc/extensions/supported/sycl_ext_oneapi_sub_group_mask.asciidoc)
   [78a3e77]
 - Added `sycl::ext::intel::experimental::esimd::simd_mask` as a replaced for
    `sycl::ext::intel::experimental::esimd::mask_type_t` to represent Gen
    predicates [01351f1]
 - Added stripped PDBs for SYCL libraries [6e5dd483]
 - Added support ESIMD emulator backend [f4ad3c11]
 - Added support for [FPGA DSP control](https://github.com/KhronosGroup/SPIRV-Registry/blob/main/extensions/INTEL/SPV_INTEL_fpga_dsp_control.asciidoc) extension [790aa8ba]
 - Implemented [discard_events extension](doc/extensions/supported/sycl_ext_oneapi_discard_queue_events.asciidoc)
   [9542e288]
 - Extended XPTI notifications with information about SYCL memory management
   [a068b154] [8f9d0d2b]
 - Added support for `sycl::device::info::ext_intel_gpu_hw_threads_per_eu`
   [2e798df8]
 - Added `sycl::ext::oneapi::no_offset` property for
   `sycl::ext::oneapi::accessor_property_list` [308e5ad5a378]
 - Implemented SYCL 2020 property traits [3b17da60b590]
 - Implemented [MAX_WORK_GROUP_QUERY extension](doc/extensions/MaxWorkGroupQueries/max_work_group_query.md)
   [2fdf94006672]
 - Added function wrapper for function-level loop fusion attribute
   (`[[intel::loop_fuse(N)]]` and `[[intel::loop_fuse_independent(N)]]`) for
   FPGA [e8ac5a076a20]
 - Added querying number of registers for kernel with `ext_codeplay_num_regs`
   kernel info query [97d33b7815e8]
 - Added backend macros `SYCL_BACKEND_OPENCL` and `SYCL_EXT_ONEAPI_*` for CUDA,
   Level Zero, ESIMD emulator, HIP [2b0ebab376dc]
 - Added support for `sycl::ext::intel::experimental::esimd_ballot` function
   [0bbb091c1baa]
 - Added initial support for [Tensor Cores matrix extension](https://github.com/intel/llvm/blob/f2983fc0d8fcd7bd6022a7006ad489c591838041/sycl/doc/extensions/experimental/sycl_ext_oneapi_matrix/sycl_ext_oneapi_matrix.asciidoc)
   [711ba58c30a8]

### Documentation
 - Added [device global extension specification](doc/extensions/proposed/sycl_ext_oneapi_device_global.asciidoc)
   [d3e70d4]
 - Added [property list extension specification](doc/extensions/proposed/sycl_ext_oneapi_properties.asciidoc)
   [a7da8b4]
 - Added extension specification for [discard queue events](doc/extensions/supported/sycl_ext_oneapi_discard_queue_events.asciidoc)
   [23ca24bb]
 - Added [KernelProperties](doc/extensions/proposed/sycl_ext_oneapi_kernel_properties.asciidoc)
   extension [64f5e707daed]

## Improvements
### SYCL Compiler
 - Added diagnostics on attempt to pass an incorrect value to
   `-fsycl-device-code-split` [631fd69]
 - Improved output of `-fsycl-help`[2404d02]
 - Allowed `::printf` builtin for CUDA backend only [0c55d3a6]
 - Implemented `nextafter` for `sycl::half` on CUDA backend [53c32682]
 - Added atomics with scopes and memory orders for CUDA backend
   [2ebde5f6] [00f43b34]
 - Added support for missing mathematical builtins in CUDA backend
   [789ec8ba] [f0747747] [390e1055]
 - Added diagnostic for non-forward declarable kernel name types [653bae90]
 - Added `group_ballot` intrinsic for CUDA backend [0680e5c5]
 - Added support for device side `assert` for CUDA backend [5a87b8cd]
 - Turned on `-fsycl-dead-args-optimization` by default [5983dfd8]
 - Improved compilation time by removing free function queries calls detection
   [e4791d11b]
 - Reduced memory consumption of device code linking [62668203]
 - Improved UX of `sycl::ext::oneapi::experimental::printf` by allowing format
   string to reside in a non-constant address space [2d62e51d]
 - Improved barrier and sync instructions to use full mask when targeting NVPTX
   [5ce99b89]
 - Added match for default SPIR device architecture with host architecture i.e.
   `x86_64` matches `spir64` and `i686` matches `spir` [f4d01cd4437b]
 - Set default device code split mode to off for FPGA [bea72e65a954]
 - Improved diagnostic for invalid SYCL kernel names
   [455dce89e3f0] [df1ff7ae3a43]
 - Made `Xsycl-target-frontend=` to accept device tripple aliases [7fa05699b5e9]
 - Improved diagnostic messages for `-fsycl-libspirv-path` [c54c6051df4c]
 - Made implied default device to force emulation for FPGA builds [074944e44f5a]
 - Added support for `sycl::ext::oneapi::sub_group::get_local_id` for HIP
   backend [7a9335d258f1]
 - Added a diagnostic of indirect implicit capture of `this` for kernel lambda
   [dce4c6a68220]
### SYCL Library
 - Updated joint matrix queries to report if unsigned int variants of mad
   matrix instruction are supported [dd7ebce]
 - Reduced overhead of device code assert implementation [b94f23a] [58ac74e]
 - Added a diagnostic on attempt to call `sycl::get_kernel_id` with an invalid
   kernel [9dd1ea3]
 - Reduced overhead on kernel submission for CUDA backend [b79ae69]
 - Reduced overhead on kernel submission in backend independent part of runtime
   [e292aa5]
 - Aligned Level-Zero Interoperability API with SYCL 2020 specification
   [dd7f82c] [e662166]
 - Made `sycl::half` default constructor constexpr [d32a444]
 - Changed CUDA and HIP backends to report each device in a separate platform
   [8dddb11]
 - Added initial support for SYCL2020 exceptions [15e0ab1]
 - Added SYCL 2020 `sycl::target::device` enumeration value [f710886]
 - Added a diagnostic on attempt to print `std::byte` using `sycl::stream`
   [dd5e094]
 - Added possibility to specify ownership of `ze_module_handle_t` when creating
   a `sycl::kernel_bundle` from it [e3c9c92]
 - Improve performance of `sycl::nd_item::get_group_range()` [0cd7b7e]
 - Deprecated `sycl::target::global_buffer`
 - Made `device_num` which can be passed to `SYCL_DEVICE_FILTER` unique
   [7aa5be0]
 - Added a diagnostic on using mutually exclusive `sycl::handler` methods
   [6f620a4]
 - Added support for `std::byte` to `sycl::vec` class [8fa04fe]
 - Added `sycl::make_kernel` interoperability support for Level-Zero backend
   [98896fd]
 - Optimized work with events in the Level Zero backend [973aee9]
 - Added support for `sycl::ext::oneapi::experimental::matrix::wi_slice` and
  `sycl::ext::oneapi::experimental::matrix::joint_matrix_fill`
   [97127ebf] [cbad428d]
 - Enabled code location information when `NDEBUG` is not defined in XPTI
   notifications [e9f2d640] [9ca7ceac]
 - Added a diagnostic on attempt to pass a command group function object to
   `sycl::queue::single_task` [2614d4d8]
 - Enlarged the maximum batch size to 64 for Level Zero backend to improve
   performance [596f6938]
 - Reduced kernel submission overhead for CUDA backend [35729a7f]
 - Improved translation of Level Zero error codes [6699a5d0], [5d9a04bac]
 - Added support for an arbitrary number of elements to
   `sycl::ext::intel::experimental::esimd::simd::copy_from/to` methods
   [2bdc4c42]
 - Added HIP support to `sycl::ext::oneapi::filter_selector`
   [7224cb2a], [b7cee067]
 - Added support for batching copy commands for Level Zero backend [4c3e6992]
 - Reduced `sycl::queue::submit` overhead by enabling post-enqueue execution
   graph cleanup [6fd6098c]
 - Added support for classes implicitly converted from `sycl::item` in
   `sycl::handler::parallel_for` parameter to align with the SYCL 2020
   specification [34b93bf8]
 - Removed direct initialization constructor from
   `sycl::ext::intel::experimental::bfloat16` class [81154ece]
 - Added `sycl::vec` and `sycl::marray` support to `sycl::known_identity` type
   trait [8fefb257]
 - Added minimal support for the generic space address space to match
   `sycl::atomic_ref` class definition in specification [e99f298e]
 - Improved cache of command-lists in the context to be per-device for Level
   Zero backend [ca457d96]
 - Extended group algorithms to support broadened types [3205368f67e0]
 - Added support for alignement flags in
   `sycl::ext::intel::experimental::esimd::simd::copy_from/copy_to` operations
   [27f5c12674fb]
 - Made `sycl::ext::oneapi::atomic_ref` available in `sycl` namespace
   [2cdcbedaf0a6]
 - Renamed `cuda` and `hip` enum values and namespaces to `ext_oneapu_cuda` and
   `ext_oneapi_hip` to align with SYCL 2020 specification [97f916e596fc]
 - Improved performance of kernel submission process [535ad1efc865]
 - Eliminated build of unwanted kernels when creating one with `make_kernel`
   [53ea8b9bc1e9]
 - Removed duplicate devices on submission to `kernel_bundle` API functions
   [c222497c52b5]
 - Deprecated `sycl::aspects::int64_base_atomics` and
   `sycl::aspects::int64_extended_atomics` [554b79cf7c17]
 - Made backend specific headers be included implicitly [bc8a00ac79ff]
 - Removed program class and related API [e7cc7b08ba5d]
 - Excluded current working directory from DLL search path when looking for
   runtime dependencies [0a65cb4b6e2f]
 - Enabled persistent device code cache for kernel bundles [810d67a21646]
 - Removed SYCL 1.2.1-style OpenCL interoperability API [bbafe08fd490]
 - Added a diagnostic on attempt to pass incorrect local size for HIP backend
   [c56499c2a72b]
 - Added a diagnostic on attempt to create an object of `sycl::buffer`with
   non-device copyable underlying date type [61f1ae6a9ef8]
 - Added a diagnostic(exception) on attempt to pass an empty `sycl::device`
   list to `sycl::link` [e95c1849a8fb]
 - Disabled fallback assert mechanism for objects of `sycl::kernel`created
   using interoperability API [f8014f57360c]
 - Added a diagnostic(exception) on attempt to pass a `sycl::kernel_bundle` to
   `sycl::handler::use_kernel_bundle` when kernel bundle and handler are bound
   to different contexts [49eb2d7967d7]
### Tools
 - Improved `sycl-ls` device indexing to handle new backends [0cee18e]
### Documentation
 - Updated [Level-Zero backend extension](doc/extensions/supported/sycl_ext_oneapi_backend_level_zero.md)
   to align with the SYCL 2020 specification [8bbebf5]
 - Updated [ITT instrumentation documentation](doc/design/ITTAnnotations.md) [9c0508b]
 - Extended [FPGA lsu](doc/extensions/proposed/sycl_ext_intel_fpga_lsu.md)
   and [SYCL_INTEL_data_flow_pipes](doc/extensions/proposed/sycl_ext_intel_dataflow_pipes.asciidoc)
   extensions to with latency control feature [5ab3cd3a]
 - Updated OpenCL CPU and FPGA emulator runtimes configuration on Windows to
   use OpenCL ICD registry records instead of `OCL_ICD_FILENAMES`. That is
   done to use the latest OpenCL ICD loader which ignores `OCL_ICD_FILENAMES`
   configuration in the administrative console [92bcb419]
 - Improved [group sorting spec](doc/extensions/experimental/sycl_ext_oneapi_group_sort.asciidoc) [f1c3506b]
 - Clarified restrictions on [device global variables](doc/extensions/DeviceGlobal/sycl_intel_device_global.asciidoc)
   [589d48844edb]
 - Deprecated [group algorithms](doc/extensions/deprecated/sycl_ext_oneapi_group_algorithms.asciidoc)
   and [subgroup extensions](doc/extensions/deprecated/sycl_ext_oneapi_sub_group.asciidoc)
   as it's superseded by SYCL2020 [7dc82742b5ae] [d2a4f502b558]
 - Removed `SubGroupAlgorithms` extension [cd5417b13a27]
 - Added proposal to add [query](doc/extensions/supported/sycl_ext_intel_device_info.md)
   for number of hardware threads per execution unit [5b771a9c4eca]
 - Updated the experimental matrix interface to match new AMX/DPAS JIT
   implementation [6495575258a8]
 - Updated test section of the contribution guide [145b9e782258]
 - Updated CUDA documentation with Windows support details [1cf024ac32a3]
 - Added FPGA properties to [device global](doc/extensions/proposed/sycl_ext_oneapi_device_global.asciidoc)
   specification [fd2bd6e1eac7]
 - Added clarifications in environment variables documentation [d5ba0dbb4912]

## Bug fixes
### SYCL Compiler
 - Fixed a problem which might affect headers searching [aad223f]
 - Fixed a problem which prevented submitting kernels from a static library
   when using CUDA backend [665a0a2]
 - Fixed backwards compatibility for libraries in archive format [53308b1]
 - Fixed crash on attempt to compile a kernel which uses templated global
   variable [226679a0]
 - Fixed a possible hang caused by recursive calls to `memcpy` or `memset` on
   CPU device [cb5e8ae54]
 - Allowed to have SYCL headers included in extern `C++` block [fcd7e317]
 - Disabled the warning when fsycl-host-compiler is used [756c2e8f]
 - Fixed handling of specializations constants when compiling for AOT and
   SPIR-V images [f82ddf41]
 - Fixed `nan` builtin for doubles in libclc [cc0d67f0]
 - Fixed out-of-bound behavior for none addressing mode on CUDA [fceb10e5]
 - Fixed memory leak which happened during device code compilation [bac0a25c]
 - Fixed compiler crash when specialization constant is value dependent
   [6dcc988eadaa]
 - Updated the device triple population logic to also incorporate usage of
   `-fsycl-link-targets` [e877e3bf8798]
 - Fixed `-Xsycl` option triple check with multiple `-fsycl-targets` to only
   use the last value of the option [c623223a78e0]
 - Fixed a bug with default value generation for specialization constant
   [4f5fa0eb59f2]
### SYCL Library
 - Fixed `sycl::is_device_copyable` type trait which could conflict with range
   rounding optimization [2d28cd4]
 - Aligned `sycl::id` and `sycl::range` implementations with the specification.
   The conversion operator from `sycl::id` to `sycl::range` has been deprecated
   and will be removed in future versions [560a214]
 - Fixed a problem with some non-users kernels showing up in kernel bundles
   [49e1e74] [4112cbc]
 - Fixed `sycl::event::get_wait_list()` which could return incorrect result
   [02852c5] [5bb3ab9e]
 - Fixed conversion of zero dimension `sycl::accessor` with
   `sycl::access::address_space::global_device_space` to an object of
   `sycl::atomic` type [ce7725d]
 - Fixed possible problem which could happen if there is no active CUDA context
   during a kernel submission in CUDA backend [0d3cc99]
 - Fixed memory leak which could happen in case of unsuccessful `sycl::stream`
   construction [4ceba5b]
 - Fixed a problem which could lead to memory corruption during graph cleanup
   [42c6f44]
 - Fixed a problem with `sycl::accessor::get_range` returning incorrect results
   in case of 3 dimensional image accessor [266515a]
 - Fixed `sycl::ext::oneapi::sub_group_mask::insert_bits` [f88a19e]
 - Fixed `sycl::ext::intel::experimental::esimd::simd_mask` behavior for
   sub-group sizes less than 32 [c855fd17]
 - Fixed a problem with `sycl::handler::fill` for HIP backend [cee76d9]
 - Fixed a problem resulting in an exception when trying to allocate a big
   `sycl::image` on the host [3061bc7]
 - Fixed behavior of `sycl::handler::parallel_for` which takes a `sycl::kernel`
   to ignore a `sycl::kernel_bundle` which may be set earlier [0b97344]
 - Made `sycl::kernel_bundle::get_native` const [91fef67]
 - Fixed logic, arithmetic and compare operators for
   `sycl::ext::intel::experimental::esimd::simd_view` with scalar values
   [7e11e48] [b88c455]
 - Aligned `sycl::device::create_sub_devices` exceptions with SYCL 2020
   specification [d2252e6]
 - Added a diagnostic on attempt to submit a kernel which is missing from a
   `sycl::kernel_bundle` set for a handler [7dbb6fb]
 - Disabled fallback assert for CUDA, HIP and FPGA devices [b0411f8] [09ece34c]
 - Fixed a race which could happen when submitting a host task to the in-order
   `sycl::queue` [20d9346]
 - Fixed memory leak when constructing a `sycl::queue` using interoperability
   API with Level Zero [299f5068]
 - Fixed a problem with releasing USM memory while there are not finished
   kernels that may access this memory for GPU devices of OpenCL backend
   [c4a72908]
 - Fixed a hang which could happen when cross queues dependencies are used
   [23e180bc]
 - Fixed a memory leak in `ext::intel::experimental::online_compiler`
   [2af6ccdb]
 - Fixed alignment of `sycl::vec` class on windows platform [826c5693]
 - Fixed an issue with incorrect arguments setting in case of multiple device
   images [5ca3628b]
 - Made `accessor::get_pointer` always return base pointer, even for accessors
   with offsets [59fcb82d]
 - Fixed problems with selection of the correct binary [a346c081]
 - Fixed sycl::span type deduction [f5d08c50]
 - Made initialization of host task thread pool thread-safe [2caf555c]
 - Fixed infinite loop when `sycl::handler::parallel_for` range exceeds
   `INT_MAX` [64720d87], [fd0b1082]
 - Fixed platform query in USM allocation info for HIP backend [2bc3c92a]
 - Fixed dangling pointer in `sycl::device_event` [8a3bd1c5]
 - Widen `(u)int8/16` to `(u)int32` and `half` to `float` in
   `sycl::group_broadcast` to workaround missing support of `(u)int8/16` and
   `half` on OpenCL CPU [1f3f9b9b]
 - Fixed an issue with `sycl::make_kernel` [13443a96]
 - Fixed a memory leak of specialization constant buffer [ae711ab9]
 - Fixed a memory leak in in-memory program cache [0b8cff452158]
 - Fixed a problem with invalid `sycl::kernel_boundle` being associated with a
   `sycl::kernel` created using `sycl::make_kernel` [7431afae7eb3]
 - Disabled use of `std::byte` when `__HAS_STD_BYTE=0` definition supplied on
   MSVC [347b114579ed]
 - Removed `half` class from global namespace [c9128e68ef39]
 - Defined missing feature test macros [504ac975d68e]
 - Fixed a problem with using `sycl::multi_ptr` with `std::byte` as an
   underlying type [cb388aae4086]
 - Fixed an issue with `sycl::queue::[memcpy|memset]` APIs breaking dependency
   chain when used for USM memory and zero size is specified [1a08b7e24864]
 - Fixed device aspect check in `sycl::has_kernel_bundle` [88cc5247ce8c]

## API/ABI breakages
 - The following classes have been removed:
    - `sycl::vector_class`
    - `sycl::string_class`
    - `sycl::function_class`
    - `sycl::mutex_class`
    - `sycl::unique_ptr_class`
    - `sycl::shared_ptr_class`
    - `sycl::weak_ptr_class`
    - `sycl::hash_class`
    - `sycl::exception_ptr_class`.
   Replacement: analogs from STL
 - Support for the following attributes have been removed:
    - `intelfpga::num_simd_work_items`
    - `intelfpga::max_work_group_size`
    - `intelfpga::max_global_work_dim`
    - `intelfpga::no_global_work_offset`
    - `intelfpga::ivdep`
    - `intelfpga::ii`
    - `intelfpga::max_concurrency`
    - `intelfpga::loop_coalesce`
    - `intelfpga::disable_loop_pipelining`
    - `intelfpga::max_interleaving`
    - `intelfpga::speculated_iterations`
    - `intelfpga::doublepump`
    - `intelfpga::singlepump`
    - `intelfpga::memory`
    - `intelfpga::register`
    - `intelfpga::bankwidth`
    - `intelfpga::numbanks`
    - `intelfpga::private_copies`
    - `intelfpga::merge`
    - `intelfpga::max_replicates`
    - `intelfpga::simple_dual_port`
    - `intelfpga::bank_bits`
    - `intelfpga::force_pow2_depth`
    - `intelfpga::scheduler_target_fmax_mhz`
   Replacement: the same attributes, but in `intel::` namespace
 - Removed `sycl::ONEAPI` and `sycl::INTEL`
   Replacement: `sycl::ext::oneapi` and `sycl::ext::intel`
 - The patch [8a3bd1c5] should not break SYCL library ABI but can affect
   device-code ABI
 - Removed program class and related API [e7cc7b08ba5d]
 - Removed SYCL 1.2.1-style OpenCL interoperability API [bbafe08fd490]
 - Removed `half` class from global namespace [c9128e68ef39]

## Known issues
 - [new] Having MESA OpenCL implementation which provides no devices on a
   system may cause incorrect device discovery. As a workaround such an OpenCL
   implementation can be disabled by removing `/etc/OpenCL/vendor/mesa.icd`.
 - [new] Compilation may fail on Windows in debug mode if a kernel uses
   `std::array`.  This happens because debug version of `std::array` in
   Microsoft STL C++ headers calls functions that are illegal for the device
   code. As a workaround the following can be done:
    1. Dump compiler pipeline execution strings by passing `-###` option to the
       compiler. The compiler will print the internal execution strings of
       compilation tools. The actual compilation will not happen.
    2. Modify the (usually) first execution string (it should have
       `-fsycl-is-device` option) by adding
       `-D_CONTAINER_DEBUG_LEVEL=0 -D_ITERATOR_DEBUG_LEVEL=0` options to the
       end of the string.  Execute all string one by one.
 - [new] `-fsycl-dead-args-optimization` can't help eliminate offset of
   accessor even though it's created with no offset specified
 - [new] `cuMemPrefetchAsync` has issues on Windows. Hence, using
   `sycl::queue::prefetch` API on Windows might lead to failure [0c33048e4926]
 - SYCL 2020 barriers show worse performance than SYCL 1.2.1 do [18c80fa]
 - When using fallback assert in separate compilation flow it requires explicit
   linking against `lib/libsycl-fallback-cassert.o` or
   `lib/libsycl-fallback-cassert.spv`
 - Performance may be impacted by JIT-ing an extra 'copier' kernel and due
   running the 'copier' kernel and host-task after each kernel which uses
   assert
 - Limit alignment of allocation requests at 64KB which is the only alignment
   supported by Level Zero[7dfaf3bd]
 - On the following scenario on Level Zero backend:
    1. Kernel A, which uses buffer A, is submitted to queue A.
    2. Kernel B, which uses buffer B, is submitted to queue B.
    3. queueA.wait().
    4. queueB.wait().
   DPCPP runtime used to treat unmap/write commands for buffer A/B as host
   dependencies (i.e. they were waited for prior to enqueueing any command
   that's dependent on them). This allowed Level Zero plugin to detect that
   each queue is idle on steps 1/2 and submit the command list right away.
   This is no longer the case since we started passing these dependencies in an
   event waitlist and Level Zero plugin attempts to batch these commands, so
   the execution of kernel B starts only on step 4. The workaround restores the
   old behavior in this case until this is resolved [2023e10d][6c137f87].
 - User-defined functions with the name and signature matching those of any
   OpenCL C built-in function (i.e. an exact match of arguments, return type
   doesn't matter) can lead to Undefined Behavior.
 - A DPC++ system that has FPGAs installed does not support multi-process
   execution. Creating a context opens the device associated with the context
   and places a lock on it for that process. No other process may use that
   device. Some queries about the device through device.get_info<>() also
   open up the device and lock it to that process since the runtime needs
   to query the actual device to obtain that information.
 - The format of the object files produced by the compiler can change between
   versions. The workaround is to rebuild the application.
 - Using `sycl::program`/`sycl::kernel_bundle` API to refer to a kernel defined
   in another translation unit leads to undefined behavior
 - Linkage errors with the following message:
   `error LNK2005: "bool const std::_Is_integral<bool>" (??$_Is_integral@_N@std@@3_NB) already defined`
   can happen when a SYCL application is built using MS Visual Studio 2019
   version below 16.3.0 and user specifies `-std=c++14` or `/std:c++14`.
 - Printing internal defines isn't supported on Windows [50628db]


# September'21 release notes

Release notes for commit range 4fc5ebe..bd68232

## New features
### SYCL Compiler
 - Added support for HIP backend
   [06fdabb5][da138cc][de6e848c][104b45ee][2e08d0e][bfbd5af][c19294f]
### SYCL Library
 - Added [sRGBA support](doc/extensions/supported/sycl_ext_oneapi_srgb.asciidoc)
   [e488327][191efdd]
 - Added a preview feature implementation for the DPC++ experimental
   [matrix extension](https://github.com/intel/llvm/blob/467ef25a309ec882027052f3d4c3df58c11ee2ac/sycl/doc/extensions/Matrix/dpcpp-joint-matrix.asciidoc)
   [7f218531] [a95f46d]
 - Added support for SYCL 2020 exceptions [5c0f748][eef07606][5af8c43d]
 - Added support for [sycl_ext_intel_bf16_conversion extension](doc/extensions/experimental/sycl_ext_intel_bf16_conversion.asciidoc)
   [8075463]
 - Added support for fallback implementation of [assert feature](doc/design/Assert.md)
   [56c9ec4]
 - Added support SYCL 2020 `sycl::logical_and` and `sycl::logical_or` operators
   [6c077a0]
### Documentation
 - Added design document for [optional kernel features](doc/design/OptionalDeviceFeatures.md)
   [88cfe16]
 - Added [SYCL_INTEL_bf16_conversion extension document](doc/extensions/experimental/sycl_ext_intel_bf16_conversion.asciidoc)
   [9f8cc3af]
 - Align [sycl_ext_oneapi_sub_group_mask extension](doc/extensions/supported/sycl_ext_oneapi_sub_group_mask.asciidoc)
   with SYCL 2020 specification [a06bd1fb]
 - Added [documentation](doc/design/SYCLInstrumentationUsingXPTI.md) of XPTI related
   tracing in SYCL [1308fe7b]
 - Align `sycl_ext_oneapi_local_memory` extension
   [document](doc/extensions/supported/sycl_ext_oneapi_local_memory.asciidoc) with SYCL 2020
   specification [6ed6565]

## Improvements
### SYCL Compiler
 - Added default device triple `spir64` when the compiler encounters any
   incoming object/libraries that have been built with the `spir64` target.
   `-fno-sycl-link-spirv` can be used for disabling this behaviour [1342360]
 - Added support for non-uniform `IMul` and `FMul` operation for `ptx-nvidiacl`
   [98a339d]
 - Added splitting modules capability when compiling for NVPTX and AMDGCN
   [c1324e6]
 - Added `-fsycl-footer-path=<path>` command-line option to set path where to
   store integration footer [155acd1]
 - Improved read only accessor handling - added `readonly` attribute to the
   associated pointer to memory [3661685]
 - Improved the output project name generation. If the output value has one of
   .a .o .out .lib .obj .exe extension, the output project directory name will
   omit extension. For any other extension the output project directory will
   keep this extension [d8237a6e]
 - Improved handling of default device with AOCX archive [e3a579f]
 - Added support for NVPTX device `printf` [4af2eb51]
 - Added support for non-const private statics in ESIMD kernels [bc51fe01]
 - Improved diagnostic generation when incorrect accessor format is used
   [a2922141]
 - Allowed passing `-Xsycl-target-backend` and `-Xsycl-target-link` when
   default target is used [d37b832]
 - Disabled kernel function propagation up the call tree to callee when in
   SYCL 2020 mode [2667e3e]
### SYCL Library
 - Improved information passed to XPTI subscribers [2af0599] [66770f0]
 - Added event interoperability to Level Zero plugin [ef33c57]
 - Enabled blitter engine for in-order queues in Level Zero plugin [904967e]
 - Removed deprecation warning for SYCL 1.2.1 barriers [18c80fa]
 - Moved free function queries to experimental namespace
   `sycl::ext::oneapi::experimental` [63ba1ce]
 - Added info query for `device::info::atomic_memory_order_capabilities` and
   `context::info::atomic_memory_order_capabilities` [9b04f41f]
 - Improved performance of generic shuffles [fb08adf]
 - Renamed `ONEAPI/INTEL` namespace to `ext::oneapi/intel` [d703f57] [ea4b8a9]
   [e9d308e]
 - Added Level Zero interoperability which allows to specify ownership of queue
   [4614ee4] [6cf48fa]
 - Added support for `reqd_work_group_size` attribute in CUDA plugin [a8fe4a5]
 - Introduced `SYCL_CACHE_DIR` environment variable which allows to specify a
   directory for persistent cache [4011775]
 - Added version of `parallel_for` accepting `range` and a reduction variable
   [d1556e4]
 - Added verbosity to some errors handling [84ee39a]
 - Added SYCL 2020 `sycl::errc_for` API [02756e3f]
 - Added SYCL 2020 `byte_size` method for `sycl::buffer` and `sycl::vec`
   classes.  `get_size` was deprecated [282d1dec]
 - Added support for USM pointers for `sycl::joint_exclusive_scan` and
   `sycl::joint_inclusive_scan` [2de0f92a]
 - Added copy and move constructors for
   `sycl::ext::intel::experimental::esimd::simd_view` [daae1475]
 - Optimized memory allocation when sub-devices are used in Level Zero plugin
   [6504ba0a]
 - Added constexpr constructors  for `vec` and `marray` classes
   [e7cd86b9][449721b2]
 - Optimized kernel cache [c16705a5]
 - Added caching of device properties in Level Zero plugin [a50f45b8]
 - Optimized Cuda plugin work with small kernels [07189af0]
 - Optimized submission of kernels [441dc3b2][33432df6]
 - Aligned implementation of `sycl_ext_oneapi_local_memory` extension
   [document](doc/extensions/supported/sycl_ext_oneapi_local_memory.asciidoc) with updated
   document [b3db5e5]
 - Improved `sycl::accessor` initialization performance on device [a10199d]
 - Added support `sycl::get_kernel_ids` and cache for `sycl::kernel_id` objects
   [491ec6d]
 - Deprecated `::half` since it should not be available in global
   namespace, `sycl::half` can be used instead [6ff9cf7]
 - Deprecated `sycl::interop_handler`, `sycl::handler::interop_task`,
   `sycl::handler::run_on_host_intel`,  `sycl::kernel::get_work_group_info` and
   `sycl::spec_constant` APIs [5120763]
 - Made `sycl::marray` trivially copyable and device copyable [6e02880]
   [b5b66737]
 - Made Level Zero interoperability API SYCL 2020 compliant for
   `sycl::platform`, `sycl::device` and `sycl::context` [c696415]
 - Deprecated unstable keys of `SYCL_DEVICE_ALLOWLIST` [b27c57c]
 - Added predefined vendor macro `SYCL_IMPLEMENTATION_ONEAPI` and
   `SYCL_IMPLEMENTATION_INTEL` [6d34ebf]
 - Deprecated `sycl::ext::intel::online_compiler`,
   `sycl::ext::intel::experimental::online_compiler` can be used instead
   [7fb56cf]
 - Deprecated `global_device_space` and `global_host_space`  values of
   `sycl::access::address_space` enumeration, `ext_intel_global_device_space`
   `ext_intel_host_device_space` can be used instead [7fb56cf]
 - Deprecated `sycl::handler::barrier` and `sycl::queue::submit_barrier`,
   `sycl::handler::ext_oneapi_barrier` and
   `sycl::queue::ext_oneapi_submit_barrier` can be used instead [7fb56cf]
 - Removed `sycl::handler::codeplay_host_task` API [9a0ea9a]
### Tools
 - Added support for ROCm devices in `get_device_count_by_type` [03155e7]
### Documentation
 - Extended group [sort algorithms extension](doc/extensions/experimental/sycl_ext_oneapi_group_sort.asciidoc)
   with interfaces to scratchpad memory [f57091d]
 - Updated several extension documents to follow SYCL 2020 extension rules
   [7fb56cf]

## Bug fixes
### SYCL Compiler
 - Fixed emission of integration header with type aliases [e3cfa19]
 - Fixed compilation for AMD GPU with `-fsycl-dead-args-optimization` [5ed48b4]
 - Removed faulty implementations for atomic loads and stores for `acquire`,
   `release` and `seq_cst` memory orders libclc for NVPTX [4876443]
 - Fixed having two specialization for the `specialization_id`, one of which was
   invalid [f71a1d5]
 - Fixed context destruction in HIP plugin [6042d3a]
 - Changed `queue::mem_advise` and `handler::mem_advise` to take `int` instead
   of `pi_mem_advice` [af2bf96]
 - Prevented passing of `-fopenmp-simd` to device compilation when used along
   with `-fsycl` [226ed8b]
 - Fixed generation of the integration header when non-base-ascii chars are
   used in the kernel name [91f50477]
 - Fixed a problem which could lead to picking up incorrect kernel at runtime in
   some situations when unnamed lambda feature is used [27c632e]
 - Fixed suggested target triple in the warning message [7cc89fa]
 - Fixed identity for multiplication on CUDA backend [a6447ca4]
 - Fixed a problem with dependency file generation [fd6d948] [1d5b2cb]
 - Fixed builtins address space type for CUDA backend [1e3136e4]
 - Fixed a problem which could lead to incorrect user header to be picked up
   [c23fe4b]
### SYCL Library
 - Added assign operator to specializations of `sycl::ext::oneapi::atomic_ref`
   [c6bc5a6]
 - Fixed the way managed memory is freed in CUDA plugin [e825916]
 - Changed names of  some SYCL internal enumerations to avoid possible
   conflicts with user macro [1419415]
 - Fixed status which was returned for host events by
   `event::get_info<info::event::command_execution_status>()` call [09715f6]
 - Fixed memory ordering used for barriers [73455a1]
 - Fixed several places in CUDA and HIP plugins where `bool` was used instead
   of `uint32_t` [764b6ff]
 - Fixed event pool memory leak in Level Zero plugin [0e95e5a]
 - Removed redundant `memcpy` call for copying struct using `fpga_reg`
   [a5d290d5]
 - Fixed an issue where the native memory object passed to interoperability
   memory object constructor was ignored on devices without host unified memory
   [da196784a]
 - Fixed a bug in `simd::replicate_w` API [d36480da]
 - Fixed group operations for `(u)int8/16` types [6a055ec2]
 - Fixed a problem with non-native specialization constants being undefined if
   they are not explicitly updated to non-default values [3d96e1d]
 - Fixed a crash which could happen when a default constructed event is passed
   to `sycl::handler::depends_on`[2fe7dd3]
 - Fixed `sycl::link` which was returning a kernel bundle in the incorrect state
   [6d98beb]
 - Fixed missing dependency for host tasks in in-order queues [739487c]
 - Fixed a memory leak in Level Zero plugin [a5b221f]
 - Fixed a problem with copying buffers when an offset is specified in CUDA
   backend [0abdfd6]
 - Fixed a problem which was preventing passing asynchronous exceptions produced
   in host tasks [f823d61]
 - Fixed a crash which could happen when submitting tasks to multiple queues
   from multiple threads [4ccfd21]
 - Fixed a possible multiple definitions problem which could happen when using
   `fsycl-device-code-split` option [d21082f]
 - Fixed several problem in configuration file processing [2a35df0]
 - Fixed imbalance in events release and retain in Level Zero plugin [6117a1b]
 - Fixed a problem which could lead to leaks of full paths to source files from
   the build environment [6bbfe42b3]

## API/ABI breakages
 - Removed `intel::reqd_work_group_size` attribute,
   `sycl::reqd_work_group_size` can be used instead [c583a20]

## Known issues
 - [new] SYCL 2020 barriers show worse performance than SYCL 1.2.1 do [18c80fa]
 - [new] When using fallback assert in separate compilation flow requires
   explicit, linking against `lib/libsycl-fallback-cassert.o` or
   `lib/libsycl-fallback-cassert.spv`
 - [new] Performance may be impacted by JIT-ing an extra 'copier' kernel and due
   running the 'copier' kernel and host-task after each kernel which uses
   assert
 - [new] Driver issue. When a two-step AOT build is used and there's at least a
   single call to devicelib function from within kernel, the device binary
   image gets corrupted
 - [new] Limit alignment of allocation requests at 64KB which is the only
   alignment supported by Level Zero[7dfaf3bd]
 - [new] On the following scenario on Level Zero backend:
    1. Kernel A, which uses buffer A, is submitted to queue A.
    2. Kernel B, which uses buffer B, is submitted to queue B.
    3. queueA.wait().
    4. queueB.wait().
   DPCPP runtime used to treat unmap/write commands for buffer A/B as host
   dependencies (i.e. they were waited for prior to enqueueing any command
   that's dependent on them). This allowed Level Zero plugin to detect that
   each queue is idle on steps 1/2 and submit the command list right away.
   This is no longer the case since we started passing these dependencies in an
   event waitlist and Level Zero plugin attempts to batch these commands, so
   the execution of kernel B starts only on step 4. The workaround restores the
   old behavior in this case until this is resolved [2023e10d][6c137f87].
 - User-defined functions with the name and signature matching those of any
   OpenCL C built-in function (i.e. an exact match of arguments, return type
   doesn't matter) can lead to Undefined Behavior.
 - A DPC++ system that has FPGAs installed does not support multi-process
   execution. Creating a context opens the device associated with the context
   and places a lock on it for that process. No other process may use that
   device. Some queries about the device through device.get_info<>() also
   open up the device and lock it to that process since the runtime needs
   to query the actual device to obtain that information.
 - The format of the object files produced by the compiler can change between
   versions. The workaround is to rebuild the application.
 - Using `sycl::program`/`sycl::kernel_bundle` API to refer to a kernel defined
   in another translation unit leads to undefined behavior
 - Linkage errors with the following message:
   `error LNK2005: "bool const std::_Is_integral<bool>" (??$_Is_integral@_N@std@@3_NB) already defined`
   can happen when a SYCL application is built using MS Visual Studio 2019
   version below 16.3.0 and user specifies `-std=c++14` or `/std:c++14`.
 - Printing internal defines isn't supported on Windows [50628db]


# July'21 release notes

Release notes for commit range 6a49170027fb..962909fe9e78

## New features
 - Implemented SYCL 2020 specialization constants [07b27965] [ba3d657]
   [bd8dcf4] [d15b841]
 - Provided SYCL 2020 function objects [24a2ad89]
 - Added support for ITT notification in SYCL Runtime [a7b8daf] [8d3921e3]
 - Implemented SYCL 2020 sub_group algorithms [e8caf6c3]
 - Implemented SYCL 2020 `sycl::handler::host_task` method [75e5a269]
 - Implemented SYCL 2020 `sycl::sub_group` class [19dcac79]
 - Added support for AMD GPU devices [ec612228]
 - Implemented SYCL 2020 `sycl::is_device_copyable` type trait [44c1cbcd]
 - Implemented SYCL 2020 USM features [1df6873d]
 - Implemented support for Device UUID from [Intel's Extensions for Device Information](doc/extensions/supported/sycl_ext_intel_device_info.md) [25aee287]
 - Implemented SYCL 2020 `sycl::atomic_fence` [dcd59547]
 - Implemented `intel::loop_count_max`, `intel::loop_count_max`,
   `intel::loop_count_avg` attributes that allow to specify number of loop
   iterations for FPGA [f74b4ef]
 - Implemented generation of compiler report for kernel arguments [201f902]
 - Implemented SYCL 2020 `[[reqd_sub_group_size]]` attribute [347e41c]
 - Implemented support for `[[intel::named_sub_group_size(primary)]]` attribute
   from [sub-group extension](doc/extensions/deprecated/sycl_ext_oneapi_sub_group.asciidoc#attributes)
   [347e41c]
 - Implemented SYCL 2020 interoperability API [e6733e4]
 - Added [group sorting algorithm](doc/extensions/experimental/sycl_ext_oneapi_group_sort.asciidoc)
   extension specification [edaee9b]
 - Added [initial draft](doc/extensions/supported/sycl_ext_oneapi_backend_level_zero.md)
   for querying of free device memory in LevelZero backend extension [fa428bf]
 - Added [InvokeSIMD](doc/extensions/proposed/sycl_ext_oneapi_invoke_simd.asciidoc) and
   [Uniform](doc/extensions/proposed/sycl_ext_oneapi_uniform.asciidoc) extensions [72e1611]
 - Added [Matrix Programming Extension for DPC++ document](https://github.com/intel/llvm/blob/ce12ec028681aa90133c518126014b0881d9e6bc/sycl/doc/extensions/Matrix/dpcpp-joint-matrix.asciidoc) [ace4c733]
 - Implemented SYCL 2020 `sycl::span` [9356d53]
 - Added [device-if](doc/extensions/proposed/sycl_ext_oneapi_device_if.asciidoc) extension
   [4fb95fc]
 - Added a [programming guide](doc/MultiTileCardWithLevelZero.md) for
   multi-tile and multi-card under Level Zero backend [d581178a]
 - Implemented SYCL 2020 `sycl::bit_cast` [d4b66bd]

## Improvements
### SYCL Compiler
 - Use `opencl-aot` instead of `aoc` when AOT flow for FPGA Emulator is
   triggered [3a99558]
 - Allowed for using an external host compiler [6f0ad1a]
 - Cleaned up preprocessing output when `-fsycl` option is passed [3a18db6]
 - Allowed kernel names in anonymous namespace [e47dbad]
 - Set default value of -sycl-std to 2020 for SYCL enabled compilations
   [680adc0]
 - Added implication of `-fPIC` compilation for wrapped object when using
   `-shared` [1754934]
 - Added a diagnostic for `-fsycl` and `-ffreestanding` as non-supported
   combination [a36c6720]
 - [ESIMD] Renamed `simd::format` to `simd::bit_cast_view` [653dede1]
 - Allowed `[[sycl::work_group_size_hint]]` to accept constant expr args
   [ef8e4019]
 - Deprecated the old-style SYCL attributes according to the SYCL 2020 spec
   [001bbd42]
 - Deprecated `[[intel::reqd_work_group_size]]` attribute spelling, please use
   `[[sycl::reqd_work_group_size]]` instead [8ef7eacc]
 - Enabled native FP atomics by default. Defining the
   `SYCL_USE_NATIVE_FP_ATOMICS` macro explicitly is no longer required - it is
   now automatically defined for hardware targets with "native" support for
   atomic functions. [0bbb68ee]
 - Switched to ignoring `-O0` option for device code when compiling for FPGA
   with hardware [7d94edf4]
 - Allowed for known aliases to be used for `-fsycl-targets`. Passing
   `*-unknown-unknown-sycldevice` components of the SYCL target triple is no
   longer necessary. [9778952a]
 - [ESIMD] Added support for half type in ESIMD intrinsics [d5958ebf]
 - Implemented `sycl::kernel::get_kernel_bundle` method [69a68a6d]
 - Added a diagnostic in case of timing issues for FPGA AOT [c69a3115]
 - Added support for C `memcpy` usages in the device code [76051ccf]
 - [ESIMD] Added support for vectorizing scalar function [3fc66cc]
 - Disabled vectorization and loop transformation passes because loop unrolling
   in "SYCL optimization mode" used default heuristic, which is tuned the code
   for CPU and might not have been profitable for other devices [ff6929e6]
### SYCL Library
 - Added an exception throw if no matched device is found when
   `SYCL_DEVICE_FILTER` is set regardless of `device_selector` used [ef4e6dd]
 - Changed event status update to complete without waiting when run on CUDA
   devices [be7c1cb]
 - Improved performance when executing with dynamic batching on Level Zero
   backend [fa382d6]
 - Introduced pooling for USM and buffer allocations in Level Zero backend
   [4cffedd]
 - Added support for vectors with length of 3 and 16 elements in sub-group load
   and store operations [4e6452d]
 - Added interop types for images for Level Zero and OpenCL backends [a58cfef]
 - Improved plugins discovery - continue discovering even if a plugin fails to
   load [8c07803]
 - Implemented queries for IEEE rounded `sqrt`/`div` in Level Zero backend
   [91b35c4]
 - Added SYCL 2020 `interop_handle::get_backend()` method [041ca27]
 - [ESIMD] Deprecated `block_load`/`block_store` and
   `simd::copy_from`/`simd::copy_to` [5c41ed6]
 - Allowed for `const` and `volatile` pointer in sub-group `load` operation
   [50edee4]
 - Replaced use of `interop<>` with SYCL 2020 `backend_return_t<>` in
   `interop_handle` [d08c21a]
 - [ESIMD] Moved ESIMD APIs to `sycl::ext::intel::experimental::esimd` namespace
   [92da579]
 - Added global offset support for Level Zero backend [9ca2f911]
 - [ESIMD] Changed `simd::replicate` API by adding suffixes into the names to
   reflect the order of template arguments [e45408ad]
 - Introduced `SYCL_REDUCTION_DETERMINISTIC` macro which forces reduction
   algorithms to produce stable results [a3fc51a4]
 - Improved `SYCL_DEVICE_ALLOWLIST` format [9216b49d]
 - Added `SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING` macro to disable range
   rounding [5c4275ac]
 - Disabled range rounding by default when compiling for FPGA [5c4275ac]
 - Deprecated `sycl::buffer::get_count()`, please use `sycl::buffer::size()`
   instead [baf2ed9d]
 - Implemented `sycl::group_barrier` free function [48363902]
 - Added support of [SYCL_INTEL_enqueue_barrier extension](doc/extensions/supported/sycl_ext_oneapi_enqueue_barrier.asciidoc) for CUDA backend [2e978482]
 - Deprecated `has_extension` method of `sycl::device` and `sycl::platform`
   classes, please use `has` method with aspects APIs instead [51c747da]
 - Deprecated `sycl::*_class` types, please use STL classes instead [51c747da]
 - Deprecated `sycl::ndrange` with an offset [51c747da]
 - Deprecated `barrier` and `mem_fence` methods of `sycl::nd_item` class,
   please use `sycl::group_barrier()` and `sycl::atomic_fence()` free functions
   instead [51c747da]
 - Deprecated `sycl::byte`, please use `std::byte` instead [51c747da]
 - Deprecated `sycl::info::device::max_constant_buffer_size` and
   `sycl::info::device::max_constant_args` [51c747da]
 - Deprecated `sycl::ext::intel::fpga_reg` taking non-trivially copyable
   structs [b4c322a8]
 - Added support for `sycl::property::queue::cuda::use_default_stream` queue
   property [08330525]
 - Switched to using atomic version of reductions if `sycl::aspect::atomic64`
   is available for a target [544fb7c8]
 - Added support for `sycl::aspect::fp16` for CUDA backend [db20bab3]
 - Deprecated `sycl::aspect::usm_system_allocator`, please use
   `sycl::aspect::usm_system_allocations` instead [000cc82d]
 - Optimized `sycl::queue::wait` to wait for batch of events rather than
   waiting for each event individually [7fe72dba]
 - Deprecated `sycl::ONEAPI::atomic_fence`, please use `sycl::atomic_fence`
   instead [dcd59547]
 - Added constexpr constructor for `sycl::half` type [5759e2a1]
 - Added support for more than 4Gb device allocations in Level Zero backend
   [fb1808b8]

### Documentation
 - Updated [sub-group algoritms](doc/extensions/SubGroupAlgorithms/SYCL_INTEL_sub_group_algorithms.asciidoc)
   extension to use `marray` instead of `vec` [98715ae]
 - Updated data flow pipes extension to be based on SYCL 2020 [f22f2e0]
 - Updated [ESIMD documentation](doc/extensions/experimental/sycl_ext_intel_esimd/sycl_ext_intel_esimd.md)
   reflecting recent API changes [1e0bd1ed]
 - Updated [devicelib](doc/extensions/supported/C-CXX-StandardLibrary.rst)
   extension document with `scalnbn`, `abs` and `div` (and their variants) as
   supported [febfb5a]
 - Addressed renaming of TBB dll to `tbb12.dll` in the
   [install script](tools/install.bat) [25433ba]

## Bug fixes
### SYCL Compiler
 - Fixed crash which could happen in corner cases when null attribute created
   [cec6469]
 - Fixed crash when lowering `__sycl_alocateLocalMemory` [4960e71]
 - Fixed workflow for multi-file compilation in AOT mode [a0099a5]
 - Fixed problem with unbundling from object for device archives for FPGA
   [25ea6e1]
 - Stopped implying `defaultlib msvcrt` for Linux based driver on Windows
   [d3dc212d]
 - Fixed handling of `[[intel::max_global_work_dim()]]` in case of
   redeclarations [9b615928]
 - Fixed incorrect diagnostics in the presence of OpenMP [cbec0b5f]
 - Fixed an issue with incorrect output project report when using `-o` option
   with FPGA AOT enabled [18ac1723]
 - Removed restriction that was preventing from applying
   `[[intel::use_stall_enable_clusters]]` attribute to ANY function [15da879d]
 - Fixed bugs with recursion in SYCL kernels - diagnostics won't be emitted on
   using recursion in a discarded branch and in constexpr context [9a9a018c]
 - Fixed handling of `intel::use_stall_enable_clusters` attribute [06e4ebc7]
### SYCL Library
 - Fixed build issue when CUDA 11 is used [f7224f1]
 - Fixed caching of sub-devices in Level Zero backend[4c34f93]
 - Fixed requesting of USM memory allocation info on CUDA [691f842]
 - Fixed [`joint_matrix_mad`](https://github.com/intel/llvm/blob/ce12ec028681aa90133c518126014b0881d9e6bc/sycl/doc/extensions/Matrix/dpcpp-joint-matrix.asciidoc)
   behaviour to return `A*B+C` instead of assigning the result to `C` [ea59c2b]
 - Workaround an issue in Level Zero backend when event isn't waited upon its
   completion but is queried for its status in an infinite loop  [bfef316]
 - Fixed persistent cache key comparison (esp. when there is `\0` symbol)
   [3e9ed1d]
 - Fixed a build issue when `sycl::kernel::get_native` is used [eb17836]
 - Fixed collisions of helper functions and SPIR-V operations when building with
   `-O0` or `-O1` [9f2fd98] [c2d6cfa]
 - [OpenCL] Fixed false-positive assertion trigger when allocation alignment is
   expected [3351916ad]
 - Aligned behavior of empty command groups with SYCL 2020 [1cf697bd]
 - Fixed build options handling when they come from different sources
   [67411472]
 - Fixed host task CUDA native memory handle [e9cf124b6]
 - Fixed a memory leak which could happen if a command submission fails
   [67eac4bd]
 - Fixed support for math functions `floor/rndd/rndu/rndz/rnde` in ESIMD mode
   [de694dd8]
 - Fixed memory allocations for multi-device contexts on Level Zero [f83c9356a]
 - Renamed `sycl::property::no_init` property to `sycl::property::no_init` in
   accordance to final SYCL 2020 specification, the old spelling is deprecated
   [ad46b641]
 - Use local size specified in `[[sycl::reqd_work_group_size]]` if no local
   size explicitly passed [0a54bef2]
 - Disabled persistent device code caching by default since it doesn't reliably
   identify driver version change [48f6bc9e]
 - [ESIMD] Fixed a bug in `simd_view::operator--` [ccc97e23]
 - Fixed a memory leak for host USM allocations [c18c3456]
 - Fixed possible crashes that could happen when `sycl::free` is called while
   there are still running kernels [c74f05d6]

## API/ABI breakages
 - None

## Known issues
 - [new] The compiler generates a temporary source file which is used during
   host compilation.  This source file will appear to be a source dependency
   and could break build environments (such as Bazel) which closely keeps track
   of the generated files during a compilation. Build environments such as
   these will need to be configured in the DPC++ space to expect an additional
   intermediate file to be part of the compilation flow.
 - User-defined functions with the name and signature matching those of any
   OpenCL C built-in function (i.e. an exact match of arguments, return type
   doesn't matter) can lead to Undefined Behavior.
 - A DPC++ system that has FPGAs installed does not support multi-process
   execution. Creating a context opens the device associated with the context
   and places a lock on it for that process. No other process may use that
   device. Some queries about the device through device.get_info<>() also
   open up the device and lock it to that process since the runtime needs
   to query the actual device to obtain that information.
 - The format of the object files produced by the compiler can change between
   versions. The workaround is to rebuild the application.
 - Using `sycl::program`/`sycl::kernel_bundle` API to refer to a kernel defined
   in another translation unit leads to undefined behavior
 - Linkage errors with the following message:
   `error LNK2005: "bool const std::_Is_integral<bool>" (??$_Is_integral@_N@std@@3_NB) already defined`
   can happen when a SYCL application is built using MS Visual Studio 2019
   version below 16.3.0 and user specifies `-std=c++14` or `/std:c++14`.
 - Printing internal defines isn't supported on Windows [50628db]

# May'21 release notes

Release notes for commit range 2ffafb95f887..6a49170027fb

## New features
 - [ESIMD] Allowed ESIMD and regular SYCL kernels to coexist in the same
   translation unit and in the same program. The `-fsycl-explicit-simd` option
   is no longer required for compiling ESIMD code and was deprecated. DPCPP RT
   implicitly appends `-vc-codegen` compile option for ESIMD images.
 - [ESIMD] Added indirect read and write methods to ESIMD class [8208427]
 - Provided `sycl::ONEAPI::has_known_identity` type trait to determine if
   reduction interface supports user-defined type at compile-time [0c7bd24]
   [060fd50]
 - Added support for multiple reduction items [c042f9e]
 - Added support for `+=`, `*=`, `|=`, `^=`, `&=` operations for custom type
   reducers [b249099]
 - Added SYCL 2020 `sycl::kernel_bundle` support [5af118a] [dcfb6b1] [ae45333]
   [8335e17]
 - Added `sycl/sycl.hpp` entry header in compliance with SYCL 2020 [5edb228]
   [24d179c]
 - Added `__LIBSYCL_[MAJOR|MINOR|PATCH]_VERSION` macros, see
   [PreprocessorMacros](doc/PreprocessorMacros.md) for more information
   [9f3a74c]
 - Added support for SYCL 2020 reductions with `read_write` access mode to
   reduction variables [733d5e3]
 - Added support for SYCL 2020 reductions with
   `sycl::property::reduction::initialize_to_identity` property [3473c1a]
 - Implemented zero argument version of `sycl::buffer::reinterpret()` for
   SYCL 2020 [c0c3c80]
 - Added an initial AOT implementation of the experimental matrix extension on
   the CPU device to target AMX hardware. Base features are supported [35db973]
 - Added support for
   [sycl_ext_oneapi_local_memory extension](doc/extensions/supported/sycl_ext_oneapi_local_memory.asciidoc)
   [5a66fcb] [9a734f6]
 - Documented [Level Zero backend](doc/extensions/supported/sycl_ext_oneapi_backend_level_zero.md)
   [8994e6d]

## Improvements
### SYCL Compiler
 - Added support for math built-ins: `fmax`, `fmin`, `isinf`, `isfinite`,
   `isnormal`, `fpclassify` [1040b94]
 - The FPGA initiation interval attribute spelling `[[intel::ii]]` is
   deprecated. The new spelling is `[[intel::initiation_interval]]`. In
   addition, `[[intel::initiation_interval]]` may now be used as a function
   attribute, formerly its use was limited to statement attribute [b04e6a0]
 - Added support for function attribute `[[intel::disable_loop_pipelining]]`
   and `[[intel::max_concurrency(n)]]` [7324b3e]
 - Enabled `-fsycl-id-queries-fit-in-int` by default [f27bb01]
 - Added support for stdlib functions: `abs`, `labs`, `llabs`, `div`, `ldiv`,
   `lldiv` [86716c5] [2e9d33c]
 - Enabled range rounding for ESIMD kernels [25b482b] [bb20b7b]
 - Improved diagnostics on invalid kernel names [0c0f4c5]
 - Improved compilation time by combining device code compilation and
   integration header generation into one step [f110dd4]
 - Added support for `sycl::queue::mem_advise` for the CUDA backend [2b56ac9]
### SYCL Library
 - Specialized atomic `fetch_add`, `fetch_min` and `fetch_max` for
   floating-point types [37a9a2a] [59ceaf4]
 - Added support for accessors to array types [7ed4f58]
 - Added sub-group information queries on CUDA [c36fa65]
 - Added support for `sycl::queue::barrier` in Level Zero plugin [7c31f90]
 - Improved runtime memory usage in Level Zero plugin [c9d71d4] [2ce2ca6]
   [46e3c64]
 - Added Level Zero interoperability with specifying of ownership [41221e2]
 - Improved runtime memory usage when using USM [461fa02]
 - Provided facility for user to control execution range rounding [f6ac45f]
 - Ensured correct access mode in `sycl::handler::copy()` method [b489479]
 - Disallowed for atomic accessors in `sycl::handler::copy()` method [14437db]
 - Provided move-assignability of `usm_allocator` class [05a805e]
 -  Improved performance of copying data during native memory object creation
    on devices without host unified memory [ad8c9d1]
 - [ESIMD] Added implicit set up of fence before barrier as required by hardware
   [692228c]
 - Allowed for using of interoperability program constructor with multi-device
   context [c7f7674]
 - Allowed trace of Level Zero calls only with `SYCL_PI_TRACE=-1` [ea73219]
 - Added throw of `feature_not_supported` when when upon attempt to create
   program using `create_program_with_source` with Level Zero or CUDA [ba77e3a]
 - Added support for `inline` `cl` namespace in debugger [8e441d4]
 - Added support for build with GCC 7 [d8fea22]
 - Added in-memory caching of programs built with custom build options
   [86b0e8d] [e152b0d]
 - Improved range rounding heuristics [7efb692]
 - Added `get_backend` methods to SYCL classes [ee7e99f]
 - Added `sycl::sub_group::load` and `sycl::sub_group::store` versions that
   take raw pointers [248f550]
 - Enabled caching of devices in `sycl::device` interoperability constructors
   [d3aeb4a]
 - Added a warning on using SYCL 1.2.1 OpenCL interoperability API when
   compiling in SYCL 2020 mode. It can be suppressed by defining
   `SYCL2020_DISABLE_DEPRECATION_WARNINGS` [a249316]
 - Added support for blitter engine in Level Zero plugin. Some memory
   operations are submitted to a Level Zero copy queue now [11ba5b5]
 - Improved `sycl::INTEL::lsu::load` and `sycl::INTEL::lsu::store` to take
   `sycl::multi_ptr` [697469f]
 - Added a diagnostic on attempt to compile a SYCL application without dynamic
   C++ RT on Windows [d4180f4]
 - Added support for `Queue Order Properties` extension for Level Zero [50005c7]
 - Improved plugin discovery mechanism - if a plugin fails to initialize others
   will be discovered anyway [d513074]
 - Added support for `sycl::info::partition_affinity_domain::numa` in Level
   Zero plugin [2ba8e05]
### Documentation
 - Updated TBB paths in `GetStartedGuide` [a9acb70]
 - Aligned linked allocation document with recent changes [22b9d01]
 - Updated `GetStartedGuide` for building with `libcxx` [d3a74c3]
 - Updated table of contents in `GetStartedGuide` [0f401bf]
 - Filled in address spaces handling section in design documentation [f782c2a]
 - Improved design document for program cache [ed4b4c4]
 - Updated compiler options [description](doc/UsersManual.md) [e56e576]
 - Updated
   [SYCL_INTEL_sub_group](doc/extensions/deprecated/sycl_ext_oneapi_sub_group.asciidoc)
   extension document to use `automatic` instead of `auto` [c4d08f5]

## Bug fixes
### SYCL Compiler
 - Suppressed link time warning on Windows that incorrectly diagnosed
   conflicting section names while linking device binaries [8e6a3ec]
 - Disabled code coverage for device compilations [12a0b11]
 - Fixed an issue when unbundling a fat static archive and targeting non-FPGA
   device [90c79c7]
 - Addressed inconsistencies when performing compilations by using the target
   triple for FPGA (`spir64_fpga-unknown-unknown-sycldevice`) vs using
   `-fintelfpga` [c9a65fc]
 - Fixed generation of the output report folder when performing FPGA AOT
   compilations from a previously generated AOCR archive [eab4791]
 - Addressed issues dealing with improper settings when performing
   preprocessing when offloading is enabled [d03de03]
 - Fixed issue when using `-fsycl-device-only` on Windows when specifying an
   output file with `/o` [d1d6c5d]
 - Fixed inlining functions called from an ESIMD kernel, which broke code
   generation in the Intel GPU vector back-end [65b459d]
 - Fixed JIT crash on ESIMD kernels compiled with `-fsycl-id-queries-fit-in-int`
   [ad86c34]
 - Fixed compiler crash on ESIMD kernels calling external functions with
   `gpu::simd` arguments [dfaaaed]
 - Fixed issue with generating preprocessed output when using
   `-fsycl-device-only` [3d2225a]
### SYCL Library
 - Fixed race-condition happening on application exit [8eb00d7] [c9c1de9]
 - Fixed faulty behaviour that happened when accessing a buffer in different
   contexts using `discard_*` access mode [f75b439]
 - Fixed support for `SYCL_PROGRAM_LINK_OPTIONS` and
   `SYCL_PROGRAM_COMPILE_OPTIONS` environment variables when compiling/linking
   through `sycl::program` class [9d74846]
 - Fixed deadlock in Level Zero plugin when batching enabled [645db17]
 - Fixed possible stack overflow in Level Zero plugin [ec6fbe1]
 - Fixed issues with empty wait list in Level Zero plugin [d8c8e08]
 - Added missing `double3` and `double4` support in geometric function `cross()`
   [b8afff4]
 - Fixed issue when using `std::vector<bool> &` argument for
   `sycl::buffer::set_final_data()` method [084d83a, 2a751bd]
 - Fixed support for `long long` in `sycl::vec::convert()` on Windows [5b49cd3]
 - Aligned local and image accessor with specification by allowing for property
   list in their constructor [88fab25]
 - Fixed support for offset in `parallel_for` for host device [1958715]
 - Added missing constructors for `sycl::buffer` class [bdfad9e]
 - Fixed coordinate conversion for `sampler` class on host device [cd6529f]
 - Fixed support for local accessors in debugger [fdacb75]
 - Fixed dropping of kernel attributes when execution range rounding is used
   [496f9a0] [677a7ea]
 - Added support for interoperability tasks that use `get_mem()` methods with
   Level Zero plugin [149f08d]
 - Fixed sub-device caching in the Level Zero plugin [0b18b49]
 - Fixed `get_native` methods to retain reference counter in case of OpenCL
   backend [ee7e99f]
 - Fixed sporadic failure happening due to illegal destruction of events before
   they have been signaled [2a76b2a]
 - Resolved a pinned host memory specific performance regression on CUDA that
   was introduced with the host unified behavior dependent logic [3be63ab]
 - Fixed illegal accesses that could happen when an application that uses host
   tasks exits without waiting for host tasks completion [552a521]
 - Fixed `sycl::event::get_info` queries that were working incorrectly when
   called on event without an encapsulated native handle [5d5a792]
 - Fixed compilation error with using multidimensional subscript for
   `sycl::accessor` with atomic access mode [0bfd34e]
 -  Fixed a crash that happened when an accessor passed to a reduction was
    destroyed immediately after [b80f13e]
 - Fixed `sycl::device::get_info` with `sycl::info::device::max_mem_alloc_size`
   which was returning incorrect value in case of Level Zero backend [8dbaa53]

## API/ABI breakages
- None

## Known issues
 - GlobalWorkOffset is not supported by Level Zero backend [6f9e9a76]
 - User-defined functions with the same name and signature (exact match of
   arguments, return type doesn't matter) as of an OpenCL C built-in
   function, can lead to Undefined Behavior.
 - A DPC++ system that has FPGAs installed does not support multi-process
   execution. Creating a context opens the device associated with the context
   and places a lock on it for that process. No other process may use that
   device. Some queries about the device through device.get_info<>() also
   open up the device and lock it to that process since the runtime needs
   to query the actual device to obtain that information.
 - The format of the object files produced by the compiler can change between
   versions. The workaround is to rebuild the application.
 - Using `sycl::program`/`sycl::kernel_bundle` API to refer to a kernel defined
   in another translation unit leads to undefined behavior
 - Linkage errors with the following message:
   `error LNK2005: "bool const std::_Is_integral<bool>" (??$_Is_integral@_N@std@@3_NB) already defined`
   can happen when a SYCL application is built using MS Visual Studio 2019
   version below 16.3.0 and user specifies `-std=c++14` or `/std:c++14`.
 - Printing internal defines isn't supported on Windows [50628db]

# January'21 release notes

Release notes for commit range 5eebd1e4bfce..2ffafb95f887

## New features
### SYCL Compiler
 - Added template parameter support for `work_group_size` attributes
   [ca9d816234c6] [289230c0b52c]
 - Introduced `auto` mode for device code split feature. This mode is the
   default one [184d258b902a]
 - Enabled support for multiple AOCX device binaries for FPGA [6ea38f0f1f7a]
### SYCL Library
 - Implemented [`online_compiler`](doc/extensions/experimental/sycl_ext_intel_online_compiler.asciidoc)
   feature [91122526e74d]
### Documentation
 - Added specification for [set kernel cache configuration extension](doc/extensions/IntelGPU/IntelGPUKernelCache.md)
   [db62fe7166ec]

## Improvements
### SYCL Compiler
 - Updated ESIMD support for split barrier, slm load and dp4a [7f5013901874]
 - Disabled swapping dimension values of `WorkGroupAttr` for semantic checks
   [7ef84b6fb60c]
 - Enabled `clang-offload-deps` tool usage for generating device dependencies
   [08a1c00028e5, 237675ba5a16, 84d2eb53e774, c8455167c2fa]
 - Switched to linking only needed symbols from fat static libraries on device
   side [5c3e5385b35d]
 - Made Intel attributes consistent with clang attributes [aeb4de7e4fa5]
### SYCL Library
 - Added filtering of partition properties to exclude non-SYCL ones
   [7f7006410c1d]
 - Fixed memory leaks of global object in the Level-Zero plugins [f58c568]
 - Fixed excessive memory allocation for device handles in the OpenCL plugin
   [8cfcdb20b06e]
 - Added support for
   `syc::ext::oneapi::property::buffer::use_pinned_host_memory` buffery property
   in LevelZero plugin [0b9a7493a2f1]
 - Improved performance of kernels which do not use work-item free functions on
   the host device [b83a1a8028a8]
 - Switched to retaining events in LevelZero event wait list until command has
   completed [c55eb20c2f8c]
 - Improved group scan/reduce algorithms performance for CUDA [bd0edfee2908]
 - Enabled sub-group loads and stores in CUDA [cfce9965c366]
### Documentation
 - Added notice of the `-o` build configuration option in GetStartedGuide
   [665e4062a5ff]
 - Added requirement to reference E2E tests for the pull-request in the commit
   message [965b63c38ca7]

## Bug fixes
### SYCL Compiler
 - Eliminated emission of `llvm.used` global variables to output module(s) in
   the post-link tool [9b07d114f5c2] [b96e647c82b2]
 - Fixed crash on scalar `fptoui` LLVM instruction [d7ca5dafcb3b]
 - Fixed detection of free function calls [6cfc3add16e3]
 - Put constant initializer list data in non-generic address space
   [0f2cf4ddc2fd]
### SYCL Library
 - Fixed memory leak in Level-Zero plugin
   [d98b61a7bd28] [051e1407a195] [45e1b143f0a9]
 - (ESIMD only) Fixed build error "no member named `cl_int2`" [d7f1b53dec62]
 - Disabled execution range rounding for ESIMD kernels [5500262254c3]
 - Fixed argument printing when tracing plugin interface calls [30db15fac6f6]
 - Fixed `as()` function for swizzle vector [70e003acbbfe]
 - Fixed synchronization of sub-group and group functions on CUDA [2b6f2cd7ba0a]

## API/ABI breakages
- None

## Known issues
  - GlobalWorkOffset is not supported by Level Zero backend [6f9e9a76]
  - The code with function pointer is hanging on Level Zero [d384295e]
  - User-defined functions with the same name and signature (exact match of
    arguments, return type doesn't matter) as of an OpenCL C built-in
    function, can lead to Undefined Behavior.
  - A DPC++ system that has FPGAs installed does not support multi-process
    execution. Creating a context opens the device associated with the context
    and places a lock on it for that process. No other process may use that
    device. Some queries about the device through device.get_info<>() also
    open up the device and lock it to that process since the runtime needs
    to query the actual device to obtain that information.
  - On Windows, DPC++ compiler enforces using dynamic C++ runtime for
    application linked with SYCL library by:
     - linking with msvcrt[d].dll when `-fsycl` switch is used;
     - emitting an error on attempts to compile a program with static C++ RT
       using `-fsycl` and `/MT` or `/MTd`.
    That protects you from complicated runtime errors caused by C++ objects
    crossing sycl[d].dll boundary and not always handled properly by different
    versions of C++ RT used on app and sycl[d].dll sides.
  - The format of the object files produced by the compiler can change between
    versions. The workaround is to rebuild the application.
  - Using `cl::sycl::program` API to refer to a kernel defined in another
    translation unit leads to undefined behavior
  - Linkage errors with the following message:
    `error LNK2005: "bool const std::_Is_integral<bool>" (??$_Is_integral@_N@std@@3_NB) already defined`
    can happen when a SYCL application is built using MS Visual Studio 2019
    version below 16.3.0 and user specifies `-std=c++14` or `/std:c++14`.
  - Employing read sampler for image accessor may result in sporadic issues with
    Level Zero plugin/backend [2c50c03]
  - Printing internal defines isn't supported on Windows [50628db]
  - Group algorithms for MUL/AND/OR/XOR cannot be enabled for group scope due to
    SPIR-V limitations, and are not enabled for sub-group scope yet as the
    SPIR-V version isn't automatically raised from 1.1 to 1.3 [96da39e]
  - We cannot run Dead Argument Elimination for ESIMD since the pointers to SPIR
    kernel functions are saved in `!genx.kernels metadata` [cf10351]


# December'20 release notes

Release notes for commit range 5d7e0925..5eebd1e4bfce

## New features
### SYCL Compiler
 - Allow for multiple build options for opencl-aot [5e5703f58449]
### SYCL Library
 - Implement [`SYCL_INTEL_mem_channel_property`](doc/extensions/supported/sycl_ext_intel_mem_channel_property.asciidoc)
   extension [2f1f3167b7c6]
 - Add `marray` class as defined by SYCL 2020 provisional [5eebd1e4bfce]
 - Implement dynamic batch size adjusting when using Level-Zero plugin
   [c70b0477aa8a, cf0d0538d162]
 - Add online compilation API interface [70ac47d23264]
### Documentation
 - Proposal for [new device descriptors extension](doc/extensions/supported/sycl_ext_intel_device_info.md)
   was added [1ad813ba133e]
 - Added [online compilation extension](doc/extensions/experimental/sycl_ext_intel_online_compiler.asciidoc)
   specification [e05a19c8d303]

## Improvements
### SYCL Compiler
 - (ESIMD only) Remove wrapping of buffer objects into images which caused
   problems like incorrect work of scatter/gather of 1- and 2-byte values
   [d2d20d6c4556]
 - Rename FPGA kernel attribute `[[intel::stall_enable]]` to
   `[[intel::use_stall_enable_clusters]]` [dab9debebe70]
 - Add template parameter support for `[[intel::max_global_work_dim]]` and
   `[[intel::no_global_work_offset]]` attributes [bd8fcc7dee34, a5fde5a924ac]
 - Remove partial-link path when dealing with fat static archives [f1aa7f4d8b79]
 - Remove unused device library function definitions from linked program
   [e9423ffdec92]
 - Don't dump IR and dot files by default in the LowerWGScope [9d0e3525ba04]
 - Support LLVM floating-point intrinsics in llvm-spirv and frontend
   [a5065ab85101]
 - Add template parameter support for `[[intel::no_global_work_offset()]]`
   attribute [a5fde5a924ac]
 - Improve group size selection by adjusting `parallel_for` execution range
   global size [74a68b7da4e7]
 - Add clang support for FPGA loop fusion function attributes [23926b0645ad]
 - Reword some compiler diagnostics [50b81c3cd6e9]
### SYCL Library
 - Eliminate performance overhead on devices without host unified memory support
   [a4f092417ef9]
 - Optimize `discard_write` access mode for host accessor [6733c8b0efde]
 - Add support for composite specialization constants
   [c62860fd6b86, d4251e3c55e7, 3ec4594a5a06]
 - Enhance PI tracing with printing output arguments [19f5ad67f30a]
 - Introduce `pi_map_flags` in lieu of `cl_map_flags` [f0e7606a6198]
 - Implement robust error handling in LevelZero plugin [65c719ddfc23]
 - Add new device descriptors as SYCL extensions [51ac08c35294]
 - Remove redundant dependencies for in-order queues [632722165db2]
### Documentation
 - Add information on AOT to GetStartedGuide [71942fbb3655]
 - Add notice on alignemnt checks in ABI policy [4326b9563575]
 - Updated design of specialization contants on using of POD types
   [81963d1ec055]
 - Document linked allocation commands [929a764a5ec4]
 - Improve ESIMD documentation rendering [079597d28f1f]
 - Improved device library documentation [f24e2a9ce464]

## Bug fixes
### SYCL Compiler
 - Do not customize optimizations for non-SPIR targets [cb069fed6712]
 - Fix address space assertion with templates [8905a8cec9a9]
 - Link libm-fp64 device library by default [ac93d6fe3d9d]
 - Add support for specialization constants' typenames declared in namespaces
   [f64f835b4313]
 - Fix loosing OpenMP device binary when program uses both OpenMP and SYCL
   offloading models [eb89f5eaab37]
### SYCL Library
 - Add missing interoperability API to construct SYCL classes with Level-Zero
   handles [10b4e8a6fc19]
 - Fix several builtins implementation for host device
   [8b82c671ab12, 786708914fd4]
 - Fix possible hang upon application finish if streams were used [bd5893ae01b1]
 - Fix failure when employing interoperability host task on queue constructed
   with reused context [9cff6c9b6127]
 - Fix "instantiation after specialization" warnings
   [56b9a1dfb92f, eadce94f8ad0]
 - Support copying of stream by value within a kernel without loss of output
   information [8d37cbacc9b8]
 - Fix handling of big and/or non-uniform work-groups in reduction kernels. The
   solution may change when reduction kernels precompilation/query approach is
   implemented [78e2599bc499]
 - Fix memory leak in event pool in Level Zero plugin [68fc7808a50e]
 - Fixed issue with finalizing context of Level Zero plugin [6cfa921856f5]
 - Fix backend selection for `SYCL_DEVICE_FILTER=*` case [c54da157f5d5]
 - Restore AccessorImplHost layout [a08eeb475679]
### Documentation
 - Updated source checkout instruction for Windows in GetStartedGuide
   [9cde15210d70]

## API/ABI breakages

## Known issues
  - GlobalWorkOffset is not supported by Level Zero backend [6f9e9a76]
  - The code with function pointer is hanging on Level Zero [d384295e]
  - User-defined functions with the same name and signature (exact match of
    arguments, return type doesn't matter) as of an OpenCL C built-in
    function, can lead to Undefined Behavior.
  - A DPC++ system that has FPGAs installed does not support multi-process
    execution. Creating a context opens the device associated with the context
    and places a lock on it for that process. No other process may use that
    device. Some queries about the device through device.get_info<>() also
    open up the device and lock it to that process since the runtime needs
    to query the actual device to obtain that information.
  - On Windows, DPC++ compiler enforces using dynamic C++ runtime for
    application linked with SYCL library by:
     - linking with msvcrt[d].dll when `-fsycl` switch is used;
     - emitting an error on attempts to compile a program with static C++ RT
       using `-fsycl` and `/MT` or `/MTd`.
    That protects you from complicated runtime errors caused by C++ objects
    crossing sycl[d].dll boundary and not always handled properly by different
    versions of C++ RT used on app and sycl[d].dll sides.
  - The format of the object files produced by the compiler can change between
    versions. The workaround is to rebuild the application.
  - Using `cl::sycl::program` API to refer to a kernel defined in another
    translation unit leads to undefined behavior
  - Linkage errors with the following message:
    `error LNK2005: "bool const std::_Is_integral<bool>" (??$_Is_integral@_N@std@@3_NB) already defined`
    can happen when a SYCL application is built using MS Visual Studio 2019
    version below 16.3.0 and user specifies `-std=c++14` or `/std:c++14`.
  - Employing read sampler for image accessor may result in sporadic issues with
    Level Zero plugin/backend [2c50c03]
  - Printing internal defines isn't supported on Windows [50628db]
  - Group algorithms for MUL/AND/OR/XOR cannot be enabled for group scope due to
    SPIR-V limitations, and are not enabled for sub-group scope yet as the
    SPIR-V version isn't automatically raised from 1.1 to 1.3 [96da39e]
  - We cannot run Dead Argument Elimination for ESIMD since the pointers to SPIR
    kernel functions are saved in `!genx.kernels metadata` [cf10351]

# November'20 release notes

Release notes for commit range c9d50752..5d7e0925

## New features
  - Implemented support for new loop attribute(intel::nofusion) for FPGA
    [68ab67ad]
  - Implemented support for new FPGA function attribute stall_enable [8fbf4bbe]
  - Implemented accessor-based gather/scatter and scalar mem access for ESIMD
    feature [0aac708a]
  - Implemented support for dot_product API [6cc97d2a]
  - Implemented ONEAPI::filter_selector that accepts one or more filters for
    device selection [174fd168]
  - Introduced SYCL_DEVICE_FILTER environment variable allowing to filter
    available devices [14e227c4], [ccdf8475]
  - Implemented accessor_properties extension [f7d073d1]
  - Implemented SYCL_INTEL_device_specific_kernel_queries [24ae95b3]
  - Implemented support for group algorithms on CUDA backend [909459ba]
  - Implemented support for sub_group extension in CUDA backend [baed6a5b],
    [551d7067], [f189e413]
  - Implemented support for USM extension in CUDA plugin [da8929e0]
  - Implemented support for cl_intel_create_buffer_with_properties extension [b8a7b012]
  - Implemented support for sycl::info::device::host_unified_memory [08066b24]
  - Added clang support for FPGA kernel attribute scheduler_target_fmax_mhz
    [20013e23]

## Improvements
### SYCL Compiler
  - Enabled USM address space by default for the FPGA hardware [7896819d]
  - Added emitting of a warning when size of kernel arguments exceeds 2kB for all
    devices [e00ab746], [4960fc90]
  - Changed default SYCL standard version to 2020 [67acf814]
  - Added diagnostics when the translator encounters an unknown or unsupported
    LLVM instruction [5a28d4e5]
  - Added diagnostic for attempt to pass a pointer to variable length array as
    kernel argument [538c4c9c]
  - Improved FPGA AOT help output with -fsycl-help [dc8a0593]
  - Made /MD the default option of compiler on Windows and made driver
    generate error if /MT was passed to it. SYCL library is designed in such a way
    that STL objects must cross the sycl.dll boundary, which is guaranteed to
    work safe on Windows only if the runtime in the app using sycl.dll and in
    sycl.dll is the same and is dynamic [d31184e1], [8735bb81], [0092d4da]
  - Enforced C++ for C source files when compiling in SYCL mode [adc2ac72]
  - Added use of template parameter in [[intelfpga::num_simd_work_items()]]
    attribute [678911a8]
  - Added new spellings for SYCL FPGA attributes [5949228d], [b1cf776e9]
  - All definitions used for compiler needs were marked with underscore prefix
    [51d3c205]
  - Disabled diagnostic about use of functions with raw pointer in kernels
    [b4a3f03f]
  - Improved diagnostics for invalid SYCL kernel names [cb5ddb49], [89fd4284]

### SYCL Library
  - Removed deprecated spelling ([[cl::intel_reqd_sub_group_size()]]) of
    IntelReqdSubGroupSize attribute [9dda36fe]
  - Added support for USM shared memory allocator for Level Zero backend
    [db5037ca]
  - Added support for a context with multiple device in Level Zero backend
    [129ee442]
  - Added diagnostics for deprecated environment variables: SYCL_BE and
    SYCL_DEVICE_TYPE [6242160b]
  - Made spec_constant default constructor public and available on host
    [53d909e2]
  - Added constraints instead of static asserts and is_native_function_object()
    for group algorithms [97bec247]
  - Added support for L0 loader validation layer [4c6cda3f]
  - Added multi-device and multi-platform support for SYCL_DEVICE_ALLOWLIST
    [dbf31c3c]
  - Removed two-input sub-group shuffles [ef969c14]
  - Enabled inspecting values wrapped into private_memory<T> by evaluating
    `operator()` from GDB [31c23ddc]
  - Changed buffer allocation in the Level Zero plugin to use host shared memory for integrated GPUs [2ae1bc9e]
  - Implemented `queue::parallel_for()` accepting reduction [ffdadc2e]
  - Improved performance of float atomic_ref [0b7dacf1]
  - Made CUDA backend try to find a better block size using
    cuOccupancyMaxPotentialBlockSize function from the CUDA driver [4fabfd16a]
  - Supported GroupBroadcast with 32-bit id to cover broadcast algorithm with
    the sub_group class [6e3f2440]

### Documentation
  - Added specification for SPV_INTEL_variable_length_array extension [9e4c51c4]
  - Added specification for accessor_properties and buffer_location extensions
    [f90614c5]
  - Moved specification for Unified Shared Memory to Khronos specification
    [a7ffe039]
  - Added documentation for filter_selector [c3f5cfba]
  - Updated C-CXX-StandardLibrary extension specification [0b6f8cd8]
  - Added ESIMD extension introduction document [c36a1411]
  - Added specialization constants extension introduction document [d88ef3b6]
  - Added specialization constant feature design doc [15cac431]
  - Documented kernel-program caching mechanism [5947cde81]
  - Added the SYCL_INTEL_mem_channel_property extension specification [5cf8088c]
  - Provided detailed description for guaranteed sub-group sizes[542c32ae]
  - Documented test-related processes [ff90e06d]
  - Added code examples for all SYCL FPGA loop attributes [6b958205]
  
## Bug fixes
### SYCL Compiler
  - Fixed crash of compiler on invalid kernel type argument. [0c220ca5e]
  - Made clang search for the full executable name as opposed to just the case
    name for the AOT tools (aoc, ocloc, opencl-aot) to avoid directory calls
    [78a86da3], [244e874b]
  - Linked SYCL device libraries by default as not all backends support SPIRV
    online linkage [9dd18ca8]
  - Fixed assertion when /P option is used on windows[a21d7ef4]
  - Fixed crash when array of pointers is passed to kernel[1fc0e4f84]
  - Fixed issues with use of type from std namespace in kernel type names
    [dd7fec83]
  - Fixed debug information missed for work-item built-in translation [9c06d429]
  - Added warnings emission which had been suppressed for SYCL headers [e6eed1a7]
  - Fixed optimization disabling option for gen to use -cl-opt-disable
    [ba4e567fe]
  - Emulated "funnel shift left" which was not supported in the OpenCL
    ExtInst set on SPIRV translator side [97d7eec5]
  - Fixed build issue when TBB and libstdc++ 10.X were used caused by including
    std C++ headers in integration header file [63369132]
  - Fixed processing for partial link step with static archives by passing linker
    specific arguments there [3ab8cc82]
  - Enabled `-reuse-exe` support for Windows [43f2d4ba]
  - Fixed missing dependency file when compiling for `-fintelfpga` and using a named
    dependency output file [df5f1ab67]

### SYCL Library
  - Fixed build log preserving for L0 plugin [638b71b1]
  - Added missing math APIs for devicelib [e438bc814]
  - Enabled async_work_group_copy for scalar and vector bool types [bb78d2cb]
  - Aligned image class constructors with the SYCL specification [049ae996]
  - Removed half type alias causing name conflicts with CUDA headers [c00c1fa3]
  - Fixed explicit copy operation for host device [f20fd4de]
  - Made stream flush operation non-blocking [e7492fb2]
  - Fixed image arguments order when passing to PI routines [70d6f87b]
  - Fixed circular dependency between the device_impl and the platform_impl
    causing handler leak [255f304f]
  - Fixed work-group size selection in reductions [2ae49f57e]
  - Fixed compilation errors when built with --std=c++20 [ecd0adbb]
  - Fixed treating internal allocations of host memory as read only for memory objects created with const pointer, causing double free issue [8b5506255]
  - Fixed a problem in Level Zero plugin with kernels and programs destruction
    while they can be used [b9bf9f5f]
  - Fixed wrong exception raised by ALLOWLIST mechanism [d81081f7]
  - Fixed reporting supported device partitioning in Level Zero [766367be]
  - Aligned get_info<info::device::version>() with the SYCL spec [4644e639]
  - Set default work group size to {1, 1, 1} to fix out-of-memory crashes on
    some configurations [4d76de43]

### Documentation
  - Fixed path to FPGA device selector [ca33f7f7]
  - Renamed LEVEL0 environment variable to LEVEL_ZERO in documents and code
    comments following source code change [2c3908b4]
  - Clarified --system-ocl key in GetStartedGuide.md [e31b94e5]

## API/ABI breakages
  - Implemented accessor_properties extension for accessor class [f7d073d1]

## Known issues
  - GlobalWorkOffset is not supported by Level Zero backend [6f9e9a76]
  - The code with function pointer is hanging on Level Zero [d384295e]
  - If an application uses std::* math function in the kernel code the
    -fsycl-device-lib=libm-fp64 option should be passed to the compiler.
  - User-defined functions with the same name and signature (exact match of
    arguments, return type doesn't matter) as of an OpenCL C built-in
    function, can lead to Undefined Behavior. 
  - A DPC++ system that has FPGAs installed does not support multi-process
    execution. Creating a context opens the device associated with the context
    and places a lock on it for that process. No other process may use that
    device. Some queries about the device through device.get_info<>() also
    open up the device and lock it to that process since the runtime needs
    to query the actual device to obtain that information.
  - On Windows, DPC++ compiler enforces using dynamic C++ runtime for
    application linked with SYCL library by:
     - linking with msvcrt[d].dll when -fsycl switch is used;
     - emitting an error on attempts to compile a program with static C++ RT
       using -fsycl and /MT or /MTd.
    That protects you from complicated runtime errors caused by C++ objects
    crossing sycl[d].dll boundary and not always handled properly by different
    versions of C++ RT used on app and sycl[d].dll sides.
  - The format of the object files produced by the compiler can change between
    versions. The workaround is to rebuild the application.
  - The SYCL library doesn't guarantee stable API/ABI, so applications compiled
    with older version of the SYCL library may not work with new one.
    The workaround is to rebuild the application.
    [ABI policy guide](doc/developer/ABIPolicyGuide.md)
  - Using `cl::sycl::program` API to refer to a kernel defined in another
    translation unit leads to undefined behavior
  - Linkage errors with the following message:
    `error LNK2005: "bool const std::_Is_integral<bool>" (??$_Is_integral@_N@std@@3_NB) already defined`
    can happen when a SYCL application is built using MS Visual Studio 2019
    version below 16.3.0 and user specifies `-std=c++14` or `/std:c++14`.
  - Employing read sampler for image accessor may result in sporadic issues with
    Level Zero plugin/backend [2c50c03]
  - Printing internal defines isn't supported on Windows [50628db]
  - Group algorithms for MUL/AND/OR/XOR cannot be enabled for group scope due to
    SPIR-V limitations, and are not enabled for sub-group scope yet as the
    SPIR-V version isn't automatically raised from 1.1 to 1.3 [96da39e]
  - We cannot run Dead Argument Elimination for ESIMD since the pointers to SPIR
    kernel functions are saved in `!genx.kernels metadata` [cf10351]

# September'20 release notes

Release notes for commit range 5976ff0..1fc0e4f

## New features

## Improvements
### SYCL Compiler
  - Assigned the source location of the kernel caller function to the artificial
    initialization code generated in the kernel body. It enables profiling tools
    to meaningfully attribute the initialization code [6744364]
  - Provided compile-time warning if size of kernel arguments exceeds 2KiB in
    GPU AOT mode [e00ab74]
  - Changed default SYCL standard to SYCL-2020 [67acf81]
  - Removed deprecated `[[cl::intel_reqd_sub_group_size(N)]]` attribute
    [9dda36f]
  - Enable USM address spaces by default for the FPGA hardware [7896819]
  - Assume SYCL device functions are convergent [047e2ec]
  - Added Dead Argument Elimination optimization [b0d98dc] [f53ede9]
  - Simplified the error checking of arrays by only visiting once [c709986]
  - Stop emitting kernel arguments metadata [f658918]
  - Enabled `-f[no-]sycl-early-optimizations` on Windows [e1e3658]
  - Mutable kernel functions are now explicitly forbidden in SYCL 2020
    [1dbc358]
  - Moved hardware targeted extensions to `INTEL` namespace [3084982]
  - Added support for union types as kernel parameters [5adfd79]
  - Renamed `-fsycl-std-optimizations` to `-fsycl-early-optimizations` [077a507]
  - Added support for `-f[no-]sycl-id-queries-fit-in-int`. Enabling this will
    make compiler define `_SYCL_ID_QUERIES_FIT_IN_INT_` macro which will signal
    runtime to emit `__builtin_assume()` for execution range less than `INT_MAX`
    limitation [3e4da3c]
  - Enabled template trail for kernel diagnostics [c767edc]
  - Disabled createIndVarSimplifyPass for SPIR target in SYCL mode [76ffef7]
  - Run Dead Argument Elimination when LLVM optimizations are applied as well
    [cf10351]

### SYCL Library
  - Eliminated circular dependency between `event` and `queue` classes [31843cc]
  - Added `ONEAPI::filter_selector` [174fd168f18]
  - Added CPU-agnostic code path to the host device runtime (validated on
    AArch64 systems) [2f632f8]
  - Added support for `bool2`, `bool3`, `bool4`, `bool8`, `bool16` [4dfd500]
  - Allowed for creating lots of host accessors [b206293]
  - Improved execution graph traversal [f2eaa23]
  - Improved `SYCL_PI_TRACE` [4d468f1]
  - Added implementation for `SYCL_INTEL_free_function_queries` [b6d7792]
  - Allowed for building program for multiple devices within single context
    (esp. for FPGA devices) [2f64227]
  - Cache devices and platforms [d392b51]
  - Reuse devices and platforms in Level Zero PI plugin [43ba606]
  - Added group algorithms for MUL/OR/XOR/AND operations [96da39e]
  - Moved general language extensions to `ONEAPI` namespace [a73369d]
  - Added CMake option `SYCL_DISABLE_STL_ASSERTIONS` to disable assertions
    [ec2ec99]
  - Implemented USM fill operation as defined in SYCL-2020 provisional [4993646]
  - Added runtime support for device code argument elimination [63ac3d3]
  - Imporved implementation of stream class when used in FPGA device code
    [13e8dae]
  - Imporved error reporting in Level Zero plugin [257658c]
  - Improved kernel demangling in graph printing [62192a6]
  - Improved error handling in `parallel_for` [7c73c11]
  - Fixed segfault in interop constructors of context, device, platform classes
    [c4c3494]

### Documentation
  - Added documentation for [`SPV_INTEL_usm_storage_classes`](doc/design/spirv-extensions/SPV_INTEL_usm_storage_classes.asciidoc)
    and [SYCL_INTEL_usm_address_spaces](doc/extensions/supported/sycl_ext_intel_usm_address_spaces.asciidoc) [781fbfc]
  - Fixed SPIR-V format name spelling [6e9bf3b]
  - Added extension [LocalMemory](doc/extensions/supported/sycl_ext_oneapi_local_memory.asciidoc) draft specification [4b5308a]
  - Added extension [free functions queries](doc/extensions/experimental/sycl_ext_oneapi_free_function_queries.asciidoc) draft specification [8953bfd]
  - Removed documentation for implicit attribute `buffer_location` [71a56e7]

## Bug fixes
### SYCL Compiler
  - Fixed crash when array of pointers is a kernel argument [1fc0e4f]
  - Allowed for `-P -fsycl` to be used on Windows when offloading [a21d7ef]
  - Fixed looking for tools (e.g. aoc, ocloc, opencl-aot) with full name on
    Windows (i.e. with `.exe` suffix) [78a86da]
  - Eliminated compiler crash if invalid declaration is used as kernel argument
    [0c220ca]
  - Switch SPIRV debug info to legacy mode to support old OpenCL RTs [500a0b8]
  - Disabled vectorizers in SYCL device code when early optimizations are
    enabled [20921b1]
  - Fixed crash when kernel argument is a multi-dimensional array [36f6ab6]
  - Fixed `cl::sycl::INTELlsu::load()` method to return value instead of
    reference [82e5323]
  - Disabled "early" optimizations for Intel FPGA by default [f8902b8]
  - Fixed regression on unused non-USM pointers inside struct type kernel
    arguments [926eb32]
  - Fixed NULL-pointer dereference in some cases [bdc2b85]
  - Adjusted AUX targets with lang options [43862a3]

### SYCL Library
  - Eliminated circular dependency between command group and stream buffers,
    which caused memory leaking [841e1e7]
  - Added early exit from enqueue process when trying to enqueue blocked
    commands. This eliminated hang in host-task when used along with multiple
    buffers [bc8f0a4]
  - Fixed overflow when casting glbal memory size in Level Zero plugin [82893b2]
  - Fixed waiting for events on Level Zero [e503662]
  - Added missing constructors and propety methods for context, program and
    sampler[30b8acc]
  - Fixed printing types of variables by GDB in some cases [93e1387]
  - Aligned `cl::sycl::handler::require` API with the SYCL specification
    [68c275c]
  - Fixed undefined behaviour in memory management intrinsics [4ff2eee]
  - Fixed race condition when using sampler in parallel [34f0c10]
  - Fixed race condition in `ProgramManager` class, which lead to hang [e6fd911]
  - Fixed thread-safety issue, which took place when using stream class [4688cb3]
  - Unified usm `queue`'s `memcpy`/`memset` methods behavior for corner cases
    [7b7bab6]
  - Enabled USM indirect access for interoperability kernels [ebf5c4e]

## API/ABI breakages
  - Added missing constructors and propety methods for context, program and
    sampler[30b8acc]

## Known issues
  - The format of the object files produced by the compiler can change between
    versions. The workaround is to rebuild the application.
  - The SYCL library doesn't guarantee stable API/ABI, so applications compiled
    with older version of the SYCL library may not work with new one.
    The workaround is to rebuild the application.
    [ABI policy guide](doc/developer/ABIPolicyGuide.md)
  - Using `cl::sycl::program` API to refer to a kernel defined in another
    translation unit leads to undefined behavior
  - Linkage errors with the following message:
    `error LNK2005: "bool const std::_Is_integral<bool>" (??$_Is_integral@_N@std@@3_NB) already defined`
    can happen when a SYCL application is built using MS Visual Studio 2019
    version below 16.3.0 and user specifies `-std=c++14` or `/std:c++14`.
  - Employing read sampler for image accessor may result in sporadic issues with
    Level Zero plugin/backend [2c50c03]
  - Printing internal defines isn't supported on Windows [50628db]
  - Group algorithms for MUL/AND/OR/XOR cannot be enabled for group scope due to
    SPIR-V limitations, and are not enabled for sub-group scope yet as the
    SPIR-V version isn't automatically raised from 1.1 to 1.3 [96da39e]
  - We cannot run Dead Argument Elimination for ESIMD since the pointers to SPIR
    kernel functions are saved in `!genx.kernels metadata` [cf10351]

# August'20 release notes

Release notes for the commit range 75b3dc2..5976ff0

## New features
  - Implemented basic support for the [Explicit SIMD extension](doc/extensions/experimental/sycl_ext_intel_esimd/sycl_ext_intel_esimd.md)
    for low-level GPU performance tuning [84bf234] [32bf607] [a lot of others]
  - Implemented support for the [SYCL_INTEL_usm_address_spaces extension](https://github.com/intel/llvm/pull/1840)
  - Implemented support for the [Use Pinned Host Memory Property extension](doc/extensions/supported/sycl_ext_oneapi_use_pinned_host_memory_property.asciidoc) [e5ea144][aee2d6c][396759d]
  - Implemented aspects feature from the SYCL 2020 provisional Specification
    [89804af]

## Improvements
### SYCL Compiler
  - [CUDA BE] Removed unnecessary memory fence in the `sycl::group::barrier`
    implementation which should improve performance [e2fc1b8]
  - [CUDA BE] Added support for the SYCL builtins from relational, geometric,
    common and math categories [d4e7929] [d9bad0b] [0c9c9c0] [99957c5]
  - Added support for `C array` as a kernel parameter [00e7308]
  - [CUDA BE] Added support for kernel offset [c7bb288]
  - [CUDA BE] Added support for `sycl::half` type [8444189][8f39763]
  - Added support for SYCL kernel inheritance and nested arrays [0b2de9e]
  - Added a diagnostic on attempt to use const static data members that are not
    const-initialized [bde1085]
  - Added support for a set of standard library functions for AOT compilation
    [2bd5dab]
  - Allowed use of function declarators with empty parentheses [a4f2182]
  - The fallback implementation of standard library functions is now linked to
    the device code, only if such functions are used in kernels only [9a8864c]
  - Added support for recursive function calls in a constexpr context [06f667a]
  - Added a diagnostic on attempt to capture `this` as a kernel parameter
    [1b9f026]
  - Added [[intel::reqd_sub_group_size()]] attribute as a replacement for
    [[cl::reqd_sub_group_size()]] which is now deprecated [b2da2c8]
  - Added propagation of attributes from transitive calls to the kernel[5c91609]
  - Changed the driver to pass corresponding device specific options when `-g`
    or `-O0` is passed [31eb425]
  - The `sycl::usm_allocator` has been improved. Now it has equality operators
    and can be used with `std::allocate_shared`. Disallowed usage with 
    device allocations [ce915ef]
  - Added support for lambda functions passed to reductions [115c1a0]
  - Enabled standard optimization pipeline for the device code by default. The
    new compiler flag can be used to disable optimizations at compile time:
    `-fno-sycl-std-optimizations` [5976ff0]

### SYCL Library
  - Added support for braced-init-list or a number as range for
    `sycl::queue::parallel_for` family functions [17299ee]
  - Finished implementation of [parallel_for simplification extension](doc/extensions/ParallelForSimpification) [af792cb]
  - Added 64-bit type support for to `load` and `store` methods of
    `sycl::intel::sub_group` [fe8d852]
  - [CUDA BE] Do not enable event profiling if it's not requested by passing
    `sycl::property::queue::enable_profiling` property [bbe8457]
  - Sub-group support has been aligned with the latest changes to the extension
    document [bea6aa2]
  - [CUDA BE] Optimized waiting for event completion by synchronizing with
    latest event for a queue [d7ee359]
  - Finished implementation of the [Host task with interop capabilities](https://github.com/codeplaysoftware/standards-proposals/blob/master/host_task/host_task.md)
    extension [f088e38]
  - Added builtins for one-element `sycl::vec` for host device [073a36b]
  - [L0 BE] Added support for specialization constants [be4e641]
  - Improved diagnostic on attempt to submit a kernel with local size which
    doesn't math value specified in the `sycl::intel::reqd_work_group_size`
    attribute for the kernel [03ef819]
  - [CUDA BE] Changed active context to be persistent [296fa1a]
  - [CUDA BE] Changed default gpu architecture for device code to `SM_50`
    [800e452]
  - Added a diagnostic on attempt to create a device accessor from zero-sized
    buffer [80b2110]
  - Changed default backend to Level Zero [11ef88c]
  - Improved performance of the SYCL graph cleanup [c099e47]
  - [L0 BE] Added support for `sycl::sampler` [f3b8cdf]
  - Added support for `TriviallyCopyable` types to the
    `sycl::intel::sub_group::shuffle` [d3c7b20]
  - Implemented range simplification for queue Shortcuts [4009b8b]
  - Changed `sycl::accessor::operator[]` to return const reference when access
    mode is `sycl::access::mode::read_only` [03db009]
  - Exceptions thrown in a host task are now will be returned as asynchronous
    exceptions [280b93c]
  - Fixed `sycl::buffer` constructor which takes a contiguous container to
    enable copy back on destruction.
  - Added support for user-defined sub-group reductions [728429a]
  - The `sycl::backend::level0` has been renamed to  `sycl::backend::level_zero`
    [215f591]
  - Extended `sycl::broadcast` to support `TriviallyCopyable` types [df6d715]
  - Implemented `get_native` and `make_*` functions for Level Zero allowing to
    query native handles of SYCL objects and to create SYCL objects by providing
    a native handle: platform, device, queue, program. The feature is described
    in the SYCL 2020 provisional specification [a51c333]
  - Added support for `sycl::intel::atomic_ref` from [sycl_ext_oneapi_extended_atomics extension](doc/extensions/supported/sycl_ext_oneapi_extended_atomics.asciidoc)

### Documentation
  - Added [SYCL_INTEL_accessor_properties](doc/extensions/accessor_properties/SYCL_INTEL_accessor_properties.asciidoc) extension specification [58fc414]
  - The documentation for the CUDA BE has been improved [928b815]
  - The [Queue Shortcuts extension](sycl/doc/extensions/QueueShortcuts/QueueShortcuts.adoc)
    document has been updated [defac3c2]
  - Added [Use Pinned Host Memory Property extension](doc/extensions/supported/sycl_ext_oneapi_use_pinned_host_memory_property.asciidoc) specification [e5ea144]
  - Updated the [sycl_ext_oneapi_extended_atomics extension](doc/extensions/supported/sycl_ext_oneapi_extended_atomics.asciidoc)
    to describe `sycl::intel::atomic_accessor` [4968e7c]
  - The [SYCL_INTEL_sub_group extension](doc/extensions/deprecated/sycl_ext_oneapi_sub_group.asciidoc)
    document has been updated [067536e]
  - Added [FPGA lsu extension](doc/extensions/supported/sycl_ext_intel_fpga_lsu.md)
    document [2c2b5f2]

## Bug fixes
### SYCL Compiler
  - Fixed the diagnostic on `cl::reqd_sub_group_size` attribute mismatches
    [75b3dc2]
  - Fixed the issue with empty input for -foffload-static-lib option [8c8137f]
  - Fixed a problem with template instantiation during integration header
    generation [4ba61d0]
  - Fixed a problem which could happen when using a command lines with large
    numbers of files [87b94d5]
  - Fixed a crash when a kernel object field is an array of structures [b00fb7c]
  - Fixed issue which could prevent using of structures with constant-sized
    arrays as a kernel parameter [a4a7950]
  - Fixed a bug in the pass for lowering hierarchical parallelism code
    (SYCLLowerWGScope). Transformation was generating the code where work items
    hit the barrier in the loop different number of times which is illegal
    [a4a7950]
  - Fixed crash on attempt to use objects of `sycl::experimental::spec_constant`
    in the struct [d5a7f20]

### SYCL Library
  - Fixed problem with waiting on the same events several times which could
    happen when using USM [9bf602c]
  - Fixed a memory leak of `sycl::event` objects happened when using USM
    specific `sycl::queue` methods [a285b9d]
  - Fixed problem which could lead to a crash or deadlock when using
    `sycl::handler::codeplay_host_task` extension [e911de7]
  - Workarounded the problem which happened when an application uses long kernel
    names [b1b8510]
  - Fixed race which could happen when submitting the same kernel from multiple
    threads [95d3ec6]
  - [CUDA BE] Fixed a memory leak related to unreleased events [d0a148a]
  - [CUDA BE] Fixed diagnostic on attempt to fetch profiling info for commands
    which profiling is not enabled for [76bf2ed]
  - [L0 BE] Fixed memory leaks of device objects [eae48f6][6acb812][39e77733]
  - [CUDA BE] Fixed a problem with that several operations were not profiled
    if required [a420e7a]
  - Fixed a possible race which could happen when an application builds an
    object of the `sycl::program` or submits kernels from multiple threads
    [363ad5f]
  - Fixed a memory leak of queue and context handles, which happened when
    backend is not OpenCL [9ddca50]
  - [CUDA BE] Fixed 3 dimensional buffer device to device copy [d917446]
  - Fixed one of the `sycl::queue` constructors which was ignoring
    `sycl::property::queue::enable_profiling` property [7863c0b]
  - Fixed endless-loop in `sycl::intel::reduction` for the data types not having
    fast atomics in case of local size is 1 [e6b6ae7]
  - Fixed a compilation error which happened when using
    `sycl::interop_handle::get_native_mem` method with an object of
    `sycl::accessor` created for host target [280b93c]
  - Fixed a deadlock which could happen when multiple threads try to build a
    program simultaneously
  - Aligned `sycl::handler::set_arg` with the SYCL specification [a6465c9]
  - Fixed an issue which could lead to "No kernel named  was found" exception
    when using `sycl::handler::set_arg` method [a08674e]
  - Fixed `sycl::device::get_info<cl::sycl::info::device::sub_group_sizes>`
    which was return incorrect data [e65841b]

## API/ABI breakages
  - The memory_manager API has changed
  - Layout of internal classes for `sycl::sampler` and `sycl::stream` have been
    changed

## Known issues
  - The format of the object files produced by the compiler can change between
    versions. The workaround is to rebuild the application.
  - The SYCL library doesn't guarantee stable API/ABI, so applications compiled
    with older version of the SYCL library may not work with new one.
    The workaround is to rebuild the application.
    [ABI policy guide](doc/developer/ABIPolicyGuide.md)
  - Using `cl::sycl::program` API to refer to a kernel defined in another
    translation unit leads to undefined behavior
  - Linkage errors with the following message:
    `error LNK2005: "bool const std::_Is_integral<bool>" (??$_Is_integral@_N@std@@3_NB) already defined`
    can happen when a SYCL application is built using MS Visual Studio 2019
    version below 16.3.0
    The workaround is to enable `-std=c++17` for the failing MSVC version.

# June'20 release notes

Release notes for the commit range ba404be..24726df

## New features
  - Added switch to assume that each amount of work-items in each ND-range
    dimension if less that 2G (fits in signed integer), which allows underlying
    BEs to perform additional optimizations [fdcaeae] [08f8656]
  - Added partial support for [Host task with interop capabilities extension](https://github.com/codeplaysoftware/standards-proposals/blob/master/host_task/host_task.md) [ae3fd5c]
  - Added support for [SYCL_INTEL_bitcast](doc/extensions/Bitcast/SYCL_INTEL_bitcast.asciidoc)
    as a `sycl::detail::bit_cast` [e3da4ef]
  - Introduced the Level Zero plugin which enables SYCL working on top of Level0
    API. Interoperability is not supportet yet [d32da99]
  - Implemented [parallel_for simplification extension](doc/extensions/ParallelForSimpification) [13fe9fb]
  - Implemented [SYCL_INTEL_enqueue_barrier extension](doc/extensions/supported/sycl_ext_oneapi_enqueue_barrier.asciidoc) [da6bfd0]
  - Implemented [SYCL_INTEL_accessor_simplification extension](https://github.com/intel/llvm/pull/1498) [1f76efc]
  - Implemented OpenCL interoperability API following [SYCL Generalization proposal](https://github.com/KhronosGroup/SYCL-Shared/blob/master/proposals/sycl_generalization.md) [bae0639]

## Improvements
### SYCL Compiler
  - Now when `-fintelfpga` option is passed, the dependency file is created in
    the temporary files location instead of input source file location [7df381a]
  - Made `-std=c++17` the default for DPC++ [3192ee7]
  - Added support for kernel name types templated using enumerations
    [e7020a1][f9226d2][125b05c][07e8d8f]
  - Added a diagnostic on attempt to use host built-in functions inside device
    code [2a4c1c8]
  - Added diagnostics on attempt to use `sycl::accessor` created for
    unsupported types in the device code [6da42a0]
  - Aligned `sizeof(long double)` between host and device code [87e6240]
  - The pragma spelling for SYCL-specific attributes except
   `cl::reqd_work_group_size` are rejected now [8fe2846]
  - Added template parameter support for `cl::intel_reqd_sub_group_size`
    attribute [0ae9729]
  - Added support for more math builtins for PTX target [9370549]
  - Added support for struct members and pointers in `intelfpga::ivdep`
    attribute [358ec04]
  - Added support for all builtins from integer and shared categories for PTX
    target [f0a4fe2]
  - Improved handling of linker inputs for static lib processing [ed2846f]
  - Dependency files are not generated by default when compiling using
    `-fsycl -fintelfpga` options now [24726df]

### SYCL Library
  - Updated the implementation to align with changes in
    [SubGroup extension](doc/extensions/deprecated/sycl_ext_oneapi_sub_group.asciidoc) [9d4c284]
  - `sycl::ordered_queue` class has been removed [875347a]
  - Added support of rounding modes for floating and integer types in
    `sycl::vec::convert` [096d0a0]
  - Added support for USM vars and placeholder accessors passed to reduction
    version of `sycl::handler::parallel_for` [94cb022][2e73da7]
  - Added support of `sycl::intel::sub_group::load/store` which take
   `sycl::multi_ptr` with `sycl::access::address_space::local_space` [0f5b55b]
  - Added a diagnostic on attempt to recompile an AOT compiled program using
    `sycl::program` API [b031186]
  - Started using custom CUDA context by default as it shows better performance
    results [9d45ead]
  - Prevented NVIDIA OpenCL platform to be selected by a SYCL application
    [7146426]
  - Adjusted the diagnostic message on attempt to use local size which is
    greater than global size to be more informative [894c10d]
  - Added a cache for PI plugins, so subsequent calls for `sycl::device`
    creation should be cheaper [03dd60d]
  - A SYCL program will be aborted now if program linking is requested when
    using Level Zero plugin. This is done because L0 doesn't support program linking
    [d4a5b71]
  - Added a diagnostic on attempt to use `sycl::program::set_spec_constant` when
    the program is already in compiled or linked state [e2e3d3d]
  - Improved `sycl::stream` class implementation on the device side in order to
    reduce local memory consumption [b838f0e]

### Documentation
  - Added [a table](doc/extensions/README.md) with DPC++ extensions status
    [dbbc474]
  - OpenCL CPU runtime installation instructions in
    [GetStartedGuide](doc/GetStartedGuide.md) and the installation script have
    been improved [9aa5029]
  - The [SYCL_INTEL_sub_group extension document](doc/extensions/deprecated/sycl_ext_oneapi_sub_group.asciidoc)
    has been updated [010f112]
  - Render user API classes on a dedicated page [98b6ee4]

## Bug fixes
### SYCL Compiler
  - Fixed device code compile options passing which could lead to
    `CL_INVALID_COMPILER_OPTIONS` error [57bad9e]
  - Fixed a problem with creating a queue for FPGA device as a global inline
    variable [357e9c8]
  - Fixed an issue with that functions which are marked with `SYCL_EXTERNAL` are
    not participate in attribute propogation and conflicting attributes checking
    [0098eab]
  - Fixed an issue which could lead to problems when a kernel name contains a
    CVR qualified type [62e2f3b]
  - Fixed file processing when using `-fsycl-link`, now the generated object
    file can be linked by a non-SYCL enabled compiler/linker [2623abe]

### SYCL Library
  - Fixed an issue with map/unmap events which caused problems with read only
    buffer accessors in CUDA backend [bf1b5b6]
  - Fixed errors happened when using `sycl::handler::copy` with `const void*`,
    `void*` or a `sycl::accessor` for a type with const qualifier [ddc0c9d]
  - Fixed an issue with copying memory to itself during `sycl::buffer` copyback
    [4bf22cc]
  - Fixed a possible deadlock which could happen when simultaneously submitting
    and waiting for kernels from multiple threads on Windows [ebace77]
  - Fixed a problem which caused device with a negative score to be still
    selected [7146426][855d214]
  - Fixed memleak which happened when using `sycl::program::get_kernel`
    [ccefc93]
  - Fixed memory copy being wrongly asynchronous which could cause data races
    on CUDA backend [4f0a3df]
  - Fixed a race which could happen when waiting for the same event from
    multiple threads [5737ad9]
  - Fixed errors which happened when using `half` or `double` types in reduction
    version of `sycl::handler::parallel_for`
  - Fixed `sycl::device::get_info<sycl::info::device::mem_base_addr_align>`
    query which was returning incorrect result for CUDA plugin [a6d03f3]
  - Fixed incorrect behavior of a `sycl::buffer` created with non-writable host
    data(e.g. `const int *`) on CUDA backend [49b6223]
  - A bunch of fixes to reduction version of `sycl::handler::parallel_for`:
    - Enabled `operator*`, `operator+`, `operator|`, `operator&`, `operator^=`
      for corresponding transparent functors used in reduction
    - Fixed the case when reduction object is passed as an R-value
    - Allowed identity-less constructors for reductions with transparent
      functors
    - Replaced some `auto` declarations with Reduction::result_type and added
      intermediate assignments/casts to avoid type ambiguities caused by using
      `sycl::half` type, and which may also be caused by custom/user types as
      well
    - Fixed compile time known identity values for `MIN` and `MAX` reductions

## API/ABI breakages
  - All functions related to `sycl::ordered_queue` have been removed
  - Removed symbols corresponding to
    `sycl::info::kernel_sub_group::max_sub_group_size_for_ndrange` and
    `sycl::info::kernel_sub_group::sub_group_count_for_ndrange` queries

## Known issues
  - [new] If there is an attribute `cl::intel_reqd_sub_group_size` with the
    same value for kernel and function called from the kernel there still can be
    compilation error.
  - The format of the object files produced by the compiler can change between
    versions. The workaround is to rebuild the application.
  - The SYCL library doesn't guarantee stable API/ABI, so applications compiled
    with older version of the SYCL library may not work with new one.
    The workaround is to rebuild the application.
    [ABI policy guide](doc/developer/ABIPolicyGuide.md)
  - Using `cl::sycl::program` API to refer to a kernel defined in another
    translation unit leads to undefined behavior
  - Linkage errors with the following message:
    `error LNK2005: "bool const std::_Is_integral<bool>" (??$_Is_integral@_N@std@@3_NB) already defined`
    can happen when a SYCL application is built using MS Visual Studio 2019
    version below 16.3.0
    The workaround is to enable `-std=c++17` for the failing MSVC version.

# May'20 release notes

Release notes for the commit range ba404be..67d3d9e

## New features
  - Implemented [reduction extension](doc/extensions/deprecated/sycl_ext_oneapi_nd_range_reductions.md)
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
  - The [compiler and runtime design doc](doc/design/CompilerAndRuntimeDesign.md) has
    been updated to describe the CUDA target and reflect changed action graphs
    [91b597b] [212a26c]
  - [ExtendedAtomics documentation](doc/extensions/ExtendedAtomics/README.md)
    has been updated [1084685]
  - Published [sycl_bitcast extension](doc/extensions/Bitcast/SYCL_INTEL_bitcast.asciidoc)
  - Published a [proposal](doc/extensions/proposed/sycl_ext_oneapi_local_static_mem_used.asciidoc)
    which adds ability to query max local size which is used by a specific
    kernel and a specific device.
  - Published [device_specific_kernel_queries](doc/extensions/DeviceSpecificKernelQueries/SYCL_INTEL_device_specific_kernel_queries.asciidoc)
    extension which rephrases work group queries as device-specific kernel
    queries [4c07ff8]
  - Added more information about the [plugin interface (PI)](doc/design/PluginInterface.md)
    [0614e9a]
  - [Contribution guidelines](../CONTRIBUTING.md) were simplified, now sign-off
    line is not required [7886fd8]
  - Added missing constructors and member functions in
    [reduction extension proposal](doc/extensions/deprecated/sycl_ext_oneapi_nd_range_reductions.md)
    [f695479]
  - Published [parallel_for simplification extension](doc/extensions/ParallelForSimpification/SYCL_INTEL_parallel_for_simplification.asciidoc) [856a777]
  - Added memory scope to [ExtendedAtomics extension](doc/extensions/supported/sycl_ext_oneapi_extended_atomics.asciidoc) [f8e11e0]
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
    [ABI policy guide](doc/developer/ABIPolicyGuide.md)
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
  - Added the implementation of [GroupAlgorithms extension](doc/extensions/deprecated/sycl_ext_oneapi_group_algorithms.asciidoc)
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
  - Added [SYCL_INTEL_enqueue_barrier extension document](doc/extensions/supported/sycl_ext_oneapi_enqueue_barrier.asciidoc)
    [6cfd2cb]
  - Added [standard layout relaxation extension](doc/extensions/RelaxStdLayout/SYCL_INTEL_relax_standard_layout.asciidoc)
    [ce53521]
  - Deprecated SubGroupNDRange extension [d9b178f]
  - Added extension for base sub-group class:
    [SubGroup](doc/extensions/deprecated/sycl_ext_oneapi_sub_group.asciidoc) [d9b178f]
  - Added extension for functions operating on sub-groups:
    [SubGroupAlgorithms](doc/extensions/deprecated/sycl_ext_oneapi_group_algorithms.asciidoc)
    [d9b178f]
  - Added extension introducing group masks and ballot functionality:
    [GroupMask](doc/extensions/supported/sycl_ext_oneapi_sub_group_mask.asciidoc)
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
  - Added documentation for [ExtendedAtomics extension](doc/extensions/supported/sycl_ext_oneapi_extended_atomics.asciidoc) [4445462]
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
  - [Design document](doc/extensions/supported/C-CXX-StandardLibrary.rst)
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
  - Added [kernel restrict all extension](doc/extensions/supported/sycl_ext_intel_kernel_args_restrict.asciidoc)
    draft [47c4c71]
  - Added initial draft of [data flow pipes extension](doc/extensions/supported/sycl_ext_intel_dataflow_pipes.asciidoc)
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
  [`cl::sycl::intel::fpga_reg`](doc/extensions/supported/sycl_ext_intel_fpga_reg.md) and
  [`cl::sycl::intel::fpga_selector`](doc/extensions/supported/sycl_ext_intel_fpga_device_selector.md)
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
