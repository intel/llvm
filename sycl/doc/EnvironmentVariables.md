# Environment Variables

This document describes environment variables that are having effect on DPC++
compiler and runtime.

## Controlling DPC++ runtime

| Environment variable | Values | Description |
| -------------------- | ------ | ----------- |
| `SYCL_BE` (deprecated) | `PI_OPENCL`, `PI_LEVEL_ZERO`, `PI_CUDA` | Force SYCL RT to consider only devices of the specified backend during the device selection. We are planning to deprecate `SYCL_BE` environment variable in the future. The specific grace period is not decided yet. Please use the new env var `SYCL_DEVICE_FILTER` instead. |
| `SYCL_DEVICE_TYPE` (deprecated) | CPU, GPU, ACC, HOST | Force SYCL to use the specified device type. If unset, default selection rules are applied. If set to any unlisted value, this control has no effect. If the requested device type is not found, a `cl::sycl::runtime_error` exception is thrown. If a non-default device selector is used, a device must satisfy both the selector and this control to be chosen. This control only has effect on devices created with a selector. We are planning to deprecate `SYCL_DEVICE_TYPE` environment variable in the future. The specific grace period is not decided yet. Please use the new env var `SYCL_DEVICE_FILTER` instead. |
| `SYCL_DEVICE_FILTER` | `backend:device_type:device_num` | See Section [`SYCL_DEVICE_FILTER`](#sycl_device_filter)  below. |
| `SYCL_DEVICE_ALLOWLIST` | See [below](#sycl_device_allowlist) | Filter out devices that do not match the pattern specified. `BackendName` accepts `host`, `opencl`, `level_zero` or `cuda`. `DeviceType` accepts `host`, `cpu`, `gpu` or `acc`. `DeviceVendorId` accepts uint32_t in hex form (`0xXYZW`). `DriverVersion`, `PlatformVersion`, `DeviceName` and `PlatformName` accept regular expression. Special characters, such as parenthesis, must be escaped. DPC++ runtime will select only those devices which satisfy provided values above and regex. More than one device can be specified using the piping symbol "\|".|
| `SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING` | Any(\*) | Disables automatic rounding-up of `parallel_for` invocation ranges. |
| `SYCL_CACHE_DIR` | Path | Path to persistent cache root directory. Default values are `%AppData%\libsycl_cache` for Windows and `$XDG_CACHE_HOME/libsycl_cache` on Linux, if `XDG_CACHE_HOME` is not set then `$HOME/.cache/libsycl_cache`. When none of the environment variables are set SYCL persistent cache is disabled. |
| `SYCL_CACHE_DISABLE_PERSISTENT (deprecated)` | Any(\*) | Has no effect. |
| `SYCL_CACHE_PERSISTENT` | Integer | Controls persistent device compiled code cache. Turns it on if set to '1' and turns it off if set to '0'. When cache is enabled SYCL runtime will try to cache and reuse JIT-compiled binaries. Default is off. |
| `SYCL_CACHE_EVICTION_DISABLE` | Any(\*) | Switches cache eviction off when the variable is set. |
| `SYCL_CACHE_MAX_SIZE` | Positive integer | Cache eviction is triggered once total size of cached images exceeds the value in megabytes (default - 8 192 for 8 GB). Set to 0 to disable size-based cache eviction. |
| `SYCL_CACHE_THRESHOLD` | Positive integer | Cache eviction threshold in days (default value is 7 for 1 week). Set to 0 for disabling time-based cache eviction. |
| `SYCL_CACHE_MIN_DEVICE_IMAGE_SIZE` | Positive integer | Minimum size of device code image in bytes which is reasonable to cache on disk because disk access operation may take more time than do JIT compilation for it. Default value is 0 to cache all images. |
| `SYCL_CACHE_MAX_DEVICE_IMAGE_SIZE` | Positive integer | Maximum size of device image in bytes which is cached. Too big kernels may overload disk too fast. Default value is 1 GB. |
| `SYCL_ENABLE_DEFAULT_CONTEXTS` | '1' or '0' | Enable ('1') or disable ('0') creation of default platform contexts in SYCL runtime. The default context for each platform contains all devices in the platform. Refer to [Platform Default Contexts](extensions/supported/sycl_ext_oneapi_default_context.asciidoc) extension to learn more. Enabled by default on Linux and disabled on Windows. |
| `SYCL_RT_WARNING_LEVEL` | Positive integer | The higher warning level is used the more warnings and performance hints the runtime library may print. Default value is '0', which means no warning/hint messages from the runtime library are allowed. The value '1' enables performance warnings from device runtime/codegen. The values greater than 1 are reserved for future use. |
| `SYCL_USM_HOSTPTR_IMPORT` | Integer | Enable by specifying non-zero value. Buffers created with a host pointer will result in host data promotion to USM, improving data transfer performance. To use this feature, also set SYCL_HOST_UNIFIED_MEMORY=1. |

`(*) Note: Any means this environment variable is effective when set to any non-null value.`

### `SYCL_DEVICE_ALLOWLIST`

A list of devices and their driver version following the pattern:
`BackendName:XXX,DeviceType:YYY,DeviceVendorId:0xXYZW,DriverVersion:{{X.Y.Z.W}}`.
Also may contain `PlatformVersion`, `DeviceName` and `PlatformName`. There is no
fixed order of properties in the pattern.

### `SYCL_DEVICE_FILTER`

This environment variable limits the SYCL RT to use only a subset of the system's devices. Setting this environment variable affects all of the device query functions (`platform::get_devices()` and `platform::get_platforms()`) and all of the device selectors. 

The value of this environment variable is a comma separated list of filters, where each filter is a triple of the form "`backend`:`device_type`:`device_num`" (without the quotes). Each element of the triple is optional, but each filter must have at least one value. Possible values of `backend` are:
- `host`
- `level_zero`
- `opencl`
- `cuda`
- `hip`
- `*`

Possible values of `device_type` are:
- `host`
- `cpu`
- `gpu`
- `acc`
- `*`

`device_num` is an integer that indexes the enumeration of devices from the sycl-ls utility tool, where the first device in that enumeration has index zero in each backend. For example, `SYCL_DEVICE_FILTER=2` will return all devices with index '2' from all different backends. If multiple devices satisfy this device number (e.g., GPU and CPU devices can be assigned device number '2'), then default_selector will choose the device with the highest heuristic point.  When `SYCL_DEVICE_ALLOWLIST` is set, it is applied before enumerating devices and affects `device_num` values.

Assuming a filter has all three elements of the triple, it selects only those devices that come from the given backend, have the specified device type, AND have the given device index. If more than one filter is specified, the RT is restricted to the union of devices selected by all filters. The RT does not include the `host` backend and the `host` device automatically unless one of the filters explicitly specifies the `host` device type. Therefore, `SYCL_DEVICE_FILTER=host` should be set to enforce SYCL to use the `host` device only. 

Note that all device selectors will throw an exception if the filtered list of devices does not include a device that satisfies the selector. For instance, `SYCL_DEVICE_FILTER=cpu,level_zero` will cause `host_selector()` to throw an exception. `SYCL_DEVICE_FILTER` also limits loading only specified plugins into the SYCL RT. In particular, `SYCL_DEVICE_FILTER=level_zero` will cause the `cpu_selector` to throw an exception since SYCL RT will only load the `level_zero` backend which does not support any CPU devices at this time. When multiple devices satisfy the filter (e..g, `SYCL_DEVICE_FILTER=gpu`), only one of them will be selected.

## Controlling DPC++ Level Zero Plugin

| Environment variable | Values | Description |
| -------------------- | ------ | ----------- |
| `SYCL_ENABLE_PCI` | Integer | When set to 1, enables obtaining the GPU PCI address when using the Level Zero backend. The default is 0. |
| `SYCL_PI_LEVEL_ZERO_DISABLE_USM_ALLOCATOR` | Any(\*) | Disable USM allocator in Level Zero plugin (each memory request will go directly to Level Zero runtime) |
| `SYCL_PI_LEVEL_ZERO_TRACK_INDIRECT_ACCESS_MEMORY` | Any(\*) | Enable support of the kernels with indirect access and corresponding deferred release of memory allocations in the Level Zero plugin. |

`(*) Note: Any means this environment variable is effective when set to any non-null value.`

## Tools variables

| Environment variable | Values | Description |
| -------------------- | ------ | ----------- |
| `INTEL_ENABLE_OFFLOAD_ANNOTATIONS` | Any(\*) | Enables ITT Annotations support for SYCL runtime. This variable should only be used by tools, that support ITT Annotations. |
| `XPTI_FRAMEWORK_DISPATCHER`(\*\*) | Path to dispatcher library | Loads XPTI instrumentation dispatcher framework library. See [XPTI Framework documentation][xpti] for more info |
| `XPTI_TRACE_ENABLE`(\*\*) | `1`, `true`, `0`, `false` | Enables XPTI instrumentation. See [XPTI Framework documentation][xpti] for more info |
| `XPTI_SUBSCRIBERS`(\*\*) | Comma separated list of subscriber libraries | Loads XPTI subscribers. See [XPTI Framework documentation][xpti] for more info |

`(*) Note: Any means this environment variable is effective when set to any non-null value.`
`(**) Note: These variables come from XPTI framework`

## Debugging variables for DPC++ Runtime

:warning: **Warning:** <span style="color:red">the environment variables
described below are used for development and debugging of DPC++ compiler
and runtime. Their semantics are subject to change. Do not rely on these
variables in production code.</span>

| Environment variable | Values | Description |
| -------------------- | ------ | ----------- |
| `SYCL_PI_TRACE` | Described [below](#sycl_pi_trace-options)  | Enable specified level of tracing for PI. |
| `SYCL_QUEUE_THREAD_POOL_SIZE` | Positive integer | Number of threads in thread pool of queue. |
| `SYCL_DEVICELIB_NO_FALLBACK` | Any(\*) | Disable loading and linking of device library images |
| `SYCL_PRINT_EXECUTION_GRAPH` | Described [below](#sycl_print_execution_graph-options) | Print execution graph to DOT text file. |
| `SYCL_DISABLE_EXECUTION_GRAPH_CLEANUP` | Any(\*) | Disable cleanup of finished command nodes at host-device synchronization points. |
| `SYCL_DISABLE_POST_ENQUEUE_CLEANUP` | Any(\*) | Disable cleanup of enqueued command nodes during submission. |
| `SYCL_THROW_ON_BLOCK` | Any(\*) | Throw an exception on attempt to wait for a blocked command.  |
| `SYCL_DEVICELIB_INHIBIT_NATIVE` | String of device library extensions (separated by a whitespace) | Do not rely on device native support for devicelib extensions listed in this option. |
| `SYCL_PROGRAM_COMPILE_OPTIONS` | String of valid OpenCL compile options | Override compile options for all programs. |
| `SYCL_PROGRAM_LINK_OPTIONS` | String of valid OpenCL link options | Override link options for all programs. |
| `SYCL_USE_KERNEL_SPV` | Path to the SPIR-V binary | Load device image from the specified file. If runtime is unable to read the file, `cl::sycl::runtime_error` exception is thrown.|
| `SYCL_DUMP_IMAGES` | Any(\*) | Dump device image binaries to file. Control has no effect if `SYCL_USE_KERNEL_SPV` is set. |
| `SYCL_HOST_UNIFIED_MEMORY` | Integer | Enforce host unified memory support or lack of it for the execution graph builder. If set to 0, it is enforced as not supported by all devices. If set to 1, it is enforced as supported by all devices. |
| `SYCL_CACHE_TRACE` | Any(\*) | If the variable is set, messages are sent to std::cerr when caching events or non-blocking failures happen (e.g. unable to access cache item file). |
| `SYCL_PARALLEL_FOR_RANGE_ROUNDING_TRACE` | Any(\*) | Enables tracing of `parallel_for` invocations with rounded-up ranges. |

`(*) Note: Any means this environment variable is effective when set to any non-null value.`

### `SYCL_PRINT_EXECUTION_GRAPH` Options

`SYCL_PRINT_EXECUTION_GRAPH` can accept one or more comma separated values from the table below

| Option | Description |
| ------ | ----------- |
| before_addCG | print graph before addCG method |
| after_addCG | print graph after addCG method |
| before_addCopyBack | print graph before addCopyBack method |
| after_addCopyBack | print graph after addCopyBack method |
| before_addHostAcc | print graph before addHostAccessor method |
| after_addHostAcc | print graph after addHostAccessor method |
| always | print graph before and after each of the above methods |

### `SYCL_PI_TRACE` Options

`SYCL_PI_TRACE` accepts a bit-mask. Supported tracing levels are in the table below

| Option | Description |
| ------ | ----------- |
| 1 | Enable basic tracing, which is tracing of PI plugins/devices discovery |
| 2 | Enable tracing of the PI calls |
| -1 | Enable all levels of tracing |

## Debugging variables for Level Zero Plugin

:warning: **Warning:** <span style="color:red">the environment variables
described below are used for development and debugging of DPC++ compiler
and runtime. Their semantics are subject to change. Do not rely on these
variables in production code.</span>

| Environment variable | Values | Description |
| -------------------- | ------ | ----------- |
| `SYCL_PI_LEVEL_ZERO_MAX_COMMAND_LIST_CACHE` | Positive integer | Maximum number of oneAPI Level Zero Command lists that can be allocated with no reuse before throwing an "out of resources" error. Default is 20000, threshold may be increased based on resource availabilty and workload demand. |
| `SYCL_PI_LEVEL_ZERO_USM_ALLOCATOR` | [EnableBuffers][;[MaxPoolSize][;[host\|device\|shared:][MaxPoolableSize][,[Capacity][,SlabMinSize]]]...] | EnableBuffers enables pooling for SYCL buffers, default 0, set to 1 to enable. MaxPoolSize is the maximum size of the pool, default 0. MemType is host, device or shared. Other parameters are values specified as positive integers with optional K, M or G suffix. MaxPoolableSize is the maximum allocation size that may be pooled, default 0 for host and shared, 32KB for device. Capacity is the number of allocations in each size range freed by the program but retained in the pool for reallocation, default 0. Size ranges follow this pattern: 64, 96, 128, 192, and so on, i.e., powers of 2, with one range in between. SlabMinSize is the minimum allocation size, 64KB for host and device, 2MB for shared. Example: SYCL_PI_LEVEL_ZERO_USM_ALLOCATOR=1;32M;host:1M,4,64K;device:1M,4,64K;shared:0,0,2M|
| `SYCL_PI_LEVEL_ZERO_BATCH_SIZE` | Integer | Sets a preferred number of compute commands to batch into a command list before executing the command list. A value of 0 causes the batch size to be adjusted dynamically. A value greater than 0 specifies fixed size batching, with the batch size set to the specified value. The default is 0. |
| `SYCL_PI_LEVEL_ZERO_COPY_BATCH_SIZE` | Integer | Sets a preferred number of copy commands to batch into a command list before executing the command list. A value of 0 causes the batch size to be adjusted dynamically. A value greater than 0 specifies fixed size batching, with the batch size set to the specified value. The default is 0. |
| `SYCL_PI_LEVEL_ZERO_FILTER_EVENT_WAIT_LIST` | Integer | When set to 0, disables filtering of signaled events from wait lists when using the Level Zero backend. The default is 1. |
| `SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE` | Any(\*) | This environment variable enables users to control use of copy engines for copy operations. If the value is an integer, it will allow the use of copy engines, if available in the device, in Level Zero plugin to transfer SYCL buffer or image data between the host and/or device(s) and to fill SYCL buffer or image data in device or shared memory. The value of this environment variable can also be a pair of the form "lower_index:upper_index" where the indices point to copy engines in a list of all available copy engines. The default is 1. |
| `SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_D2D_COPY` (experimental) | Integer | Allows the use of copy engine, if available in the device, in Level Zero plugin for device to device copy operations. The default is 0. This option is experimental and will be removed once heuristics are added to make a decision about use of copy engine for device to device copy operations. |
| `SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS` | Any(\*) | Enable support of device-scope events whose state is not visible to the host. If enabled mode is SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=1 the Level Zero plugin would create all events having device-scope only and create proxy host-visible events for them when their status is needed (wait/query) on the host. If enabled mode is SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=2 the Level Zero plugin would create all events having device-scope and add proxy host-visible event at the end of each command-list submission. The default is 0, meaning all events are host-visible. |

## Debugging variables for CUDA Plugin

:warning: **Warning:** <span style="color:red">the environment variables
described below are used for development and debugging of DPC++ compiler
and runtime. Their semantics are subject to change. Do not rely on these
variables in production code.</span>

| Environment variable | Values | Description |
| -------------------- | ------ | ----------- |
| `SYCL_PI_CUDA_ENABLE_IMAGE_SUPPORT` (experimental) | Any(\*) | Enable support of images. This option is experimental since the image support is not fully implemented. |

`(*) Note: Any means this environment variable is effective when set to any non-null value.`

[xpti]: https://github.com/intel/llvm/blob/sycl/xptifw/doc/XPTI_Framework.md
