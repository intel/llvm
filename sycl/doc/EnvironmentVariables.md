# Environment Variables

This document describes environment variables that are having effect on DPC++
compiler and runtime.

## Controlling DPC++ runtime

| Environment variable | Values | Description |
| -------------------- | ------ | ----------- |
| `ONEAPI_DEVICE_SELECTOR` | [See below.](#oneapi_device_selector)  | This device selection environment variable can be used to limit the choice of devices available when the SYCL-using application is run.  Useful for limiting devices to a certain type (like GPUs or accelerators) or backends (like Level Zero or OpenCL).  This device selection mechanism is replacing `SYCL_DEVICE_FILTER` .  The `ONEAPI_DEVICE_SELECTOR` syntax is shared with OpenMP and also allows sub-devices to be chosen. [See below.](#oneapi_device_selector) for a full description.   |
| `SYCL_DEVICE_ALLOWLIST` | See [below](#sycl_device_allowlist) | Filter out devices that do not match the pattern specified. `BackendName` accepts `host`, `opencl`, `level_zero`, `native_cpu` or `cuda`. `DeviceType` accepts `host`, `cpu`, `gpu` or `acc`. `DeviceVendorId` accepts uint32_t in hex form (`0xXYZW`). `DriverVersion`, `PlatformVersion`, `DeviceName` and `PlatformName` accept regular expression. Special characters, such as parenthesis, must be escaped. DPC++ runtime will select only those devices which satisfy provided values above and regex. More than one device can be specified using the piping symbol "\|".|
| `SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING` | Any(\*) | Disables automatic rounding-up of `parallel_for` invocation ranges. |
| `SYCL_CACHE_DIR` | Path | Path to persistent cache root directory. Default values are `%AppData%\libsycl_cache` for Windows and `$XDG_CACHE_HOME/libsycl_cache` on Linux, if `XDG_CACHE_HOME` is not set then `$HOME/.cache/libsycl_cache`. When none of the environment variables are set SYCL persistent cache is disabled. |
| `SYCL_CACHE_DISABLE_PERSISTENT (deprecated)` | Any(\*) | Has no effect. |
| `SYCL_CACHE_PERSISTENT` | Integer | Controls persistent device compiled code cache. Turns it on if set to '1' and turns it off if set to '0'. When cache is enabled SYCL runtime will try to cache and reuse JIT-compiled binaries. Default is off. |
| `SYCL_CACHE_IN_MEM` | '1' or '0' | Enable ('1') or disable ('0') in-memory caching of device compiled code. When cache is enabled SYCL runtime will try to cache and reuse JIT-compiled binaries. Default is '1'. |
| `SYCL_CACHE_EVICTION_DISABLE` | Any(\*) | Switches cache eviction off when the variable is set. |
| `SYCL_CACHE_MAX_SIZE` | Positive integer | Cache eviction is triggered once total size of cached images exceeds the value in megabytes (default - 8 192 for 8 GB). Set to 0 to disable size-based cache eviction. |
| `SYCL_CACHE_THRESHOLD` | Positive integer | Cache eviction threshold in days (default value is 7 for 1 week). Set to 0 for disabling time-based cache eviction. |
| `SYCL_CACHE_MIN_DEVICE_IMAGE_SIZE` | Positive integer | Minimum size of device code image in bytes which is reasonable to cache on disk because disk access operation may take more time than do JIT compilation for it. Default value is 0 to cache all images. |
| `SYCL_CACHE_MAX_DEVICE_IMAGE_SIZE` | Positive integer | Maximum size of device image in bytes which is cached. Too big kernels may overload disk too fast. Default value is 1 GB. |
| `SYCL_ENABLE_DEFAULT_CONTEXTS` | '1' or '0' | Enable ('1') or disable ('0') creation of default platform contexts in SYCL runtime. The default context for each platform contains all devices in the platform. Refer to [Platform Default Contexts](extensions/supported/sycl_ext_oneapi_default_context.asciidoc) extension to learn more. Enabled by default on Linux and disabled on Windows. |
| `SYCL_RT_WARNING_LEVEL` | Positive integer | The higher warning level is used the more warnings and performance hints the runtime library may print. Default value is '0', which means no warning/hint messages from the runtime library are allowed. The value '1' enables performance warnings from device runtime/codegen. The values greater than 1 are reserved for future use. |
| `SYCL_USM_HOSTPTR_IMPORT` | Integer | Enable by specifying non-zero value. Buffers created with a host pointer will result in host data promotion to USM, improving data transfer performance. To use this feature, also set SYCL_HOST_UNIFIED_MEMORY=1. |
| `SYCL_EAGER_INIT` | Integer | Enable by specifying non-zero value. Tells the SYCL runtime to do as much as possible initialization at objects construction as opposed to doing lazy initialization on the fly. This may mean doing some redundant work at warmup but ensures fastest possible execution on the following hot and reportable paths. It also instructs PI plugins to do the same. Default is "0".  |
| `SYCL_REDUCTION_PREFERRED_WORKGROUP_SIZE` | See [below](#sycl_reduction_preferred_workgroup_size) | Controls the preferred work-group size of reductions. |
| `SYCL_ENABLE_FUSION_CACHING` | '1' or '0' | Enable ('1') or disable ('0') caching of JIT compilations for kernel fusion. Caching avoids repeatedly running the JIT compilation pipeline if the same sequence of kernels is fused multiple times. Default value is '1'. |

`(*) Note: Any means this environment variable is effective when set to any non-null value.`

### `ONEAPI_DEVICE_SELECTOR`

With no environment variables set to say otherwise, all platforms and devices presently on the machine are available. The default choice will be one of these devices, usually preferring a Level Zero GPU device, if available. The `ONEAPI_DEVICE_SELECTOR` can be used to limit that choice of devices, and to expose GPU sub-devices or sub-sub-devices as individual devices.

The syntax of this environment variable follows this BNF grammar:
```
ONEAPI_DEVICE_SELECTOR = <selector-string>
<selector-string> ::= { <accept-filters> | <discard-filters> | <accept-filters>;<discard-filters> }
<accept-filters> ::= <accept-filter>[;<accept-filter>...]
<discard-filters> ::= <discard-filter>[;<discard-filter>...]
<accept-filter> ::= <term>
<discard-filter> ::= !<term>
<term> ::= <backend>:<devices>
<backend> ::= { * | level_zero | opencl | cuda | hip | native_cpu }  // case insensitive
<devices> ::= <device>[,<device>...]
<device> ::= { * | cpu | gpu | fpga | <num> | <num>.<num> | <num>.* | *.* | <num>.<num>.<num> | <num>.<num>.* | <num>.*.* | *.*.*  }  // case insensitive
```

Each term in the grammar selects a collection of devices from a particular backend. The device names cpu, gpu, and fpga select all devices from that backend with the corresponding type. A backend's device can also be selected by its numeric index (zero-based) or by using `*` which selects all devices in the backend.

The dot syntax (e.g. `<num>.<num>`) causes one or more GPU sub-devices to be exposed to the application as SYCL root devices. For example, `1.0` exposes the first sub-device of the second device as a SYCL root device. The syntax `<num>.*` exposes all sub-devices of the give device as SYCL root devices. The syntax `*.*` exposes all sub-devices of all GPU devices as SYCL root devices.

In general, a term with one or more asterisks ( `*` ) matches all backends, devices, or sub-devices with the given pattern. However, a warning is generated if the term does not match anything. For example, `*:gpu` matches all GPU devices in all backends (ignoring backends with no GPU devices), but it generates a warning if there are no GPU devices in any backend. Likewise, `level_zero:*.*` matches all sub-devices of partitionable GPUs in the Level Zero backend, but it generates a warning if there are no Level Zero GPU devices that are partitionable into sub-devices.

The device indices are zero-based and are unique only within a backend. Therefore, `level_zero:0` is a different device from `cuda:0`. To see the indices of all available devices, run the `sycl-ls` tool. Note that different backends sometimes expose the same hardware as different "devices". For example, the level_zero and opencl backends both expose the Intel GPU devices.


Additionally, if a sub-device is chosen (via numeric index or wildcard), then an additional layer of partitioning can be specified. In other words, a sub-sub-device can be selected. Like sub-devices, this is done with a period ( `.` ) and a sub-sub-device specifier which is a wildcard symbol ( `*` ) or a numeric index.  Example `ONEAPI_DEVICE_SELECTOR=level_zero:0.*.*` would partition device 0 into sub-devices and then partition each of those into sub-sub-devices. The range of grandchild sub-sub-devices would be the final devices available to the app, neither device 0, nor its child partitions would be in that list.

Lastly, a filter in the grammar can be thought of as a term in conjunction with an action that is taken on all devices that are selected by the term. The action can be an accept action or a discard action. Based on the action, a filter can be an accept filter or a discard filter.
The string `<term>` represents an accept filter and the string `!<term>` represents a discard filter. The underlying term is the same but they perform different actions on the matching devices list.
For example, `!opencl:*` discards all devices of the opencl backend from the list of available devices. The discarding filters, if there are any, must all appear at the end of the selector string.
When one or more filters accept a device and one or more filters discard the device, the latter have priority and the device is ultimately not made available to the user. This allows the user to provide selector strings such as `*:gpu;!cuda:*` that accepts all gpu devices except those with a CUDA backend.
Furthermore, if the value of this environment variable only has discarding filters, an accepting filter that matches all devices, but not sub-devices and sub-sub-devices, will be implicitly included in the
environment variable to allow the user to specify only the list of devices that must not be made available. Therefore, `!*:cpu` will accept all devices except those that are of the cpu type and `opencl:*;!*:cpu`
will accept all devices of the opencl backend except those that are of the opencl backend and of the cpu type. It is legal to have a rejection filter even if it specifies devices have already been omitted by previous filters in the selection string. Doing so has no effect; the rejected devices are still omitted.

The following examples further illustrate the usage of this environment variable:

| Example | Result |
-----------|---------
| `ONEAPI_DEVICE_SELECTOR=opencl:*` | Only the OpenCL devices are available |
| `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` | Only GPU devices on the Level Zero platform are available.|
| `ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"` | GPU devices from both Level Zero and OpenCL are available. Note that escaping (like quotation marks) will likely be needed when using semi-colon separated entries. |
| `ONEAPI_DEVICE_SELECTOR=opencl:gpu,cpu` | Only CPU and GPU devices on the OpenCL platform are available.|
| `ONEAPI_DEVICE_SELECTOR=opencl:0` | Only the device with index 0 on the OpenCL backend is available. |
| `ONEAPI_DEVICE_SELECTOR=hip:0,2` | Only devices with indices of 0 and 2 from the HIP backend are available. |
| `ONEAPI_DEVICE_SELECTOR=opencl:0.*` | All the sub-devices from the OpenCL device with index 0 are exposed as SYCL root devices. No other devices are available. |
| `ONEAPI_DEVICE_SELECTOR=opencl:0.2` | The third sub-device (2 in zero-based counting) of the OpenCL device with index 0 will be the sole device available.  |
| `ONEAPI_DEVICE_SELECTOR=level_zero:*,*.*` | Exposes Level Zero devices to the application in two different ways. Each device (aka "card") is exposed as a SYCL root device and each sub-device is also exposed as a SYCL root device.|
| `ONEAPI_DEVICE_SELECTOR="opencl:*;!opencl:0"` | All OpenCL devices except for the device with index 0 are available. |
| `ONEAPI_DEVICE_SELECTOR="!*:cpu"` | All devices except for CPU devices are available. |

Notes:
- The backend argument is always required. An error will be thrown if it is absent.
- Additionally, the backend MUST be followed by colon ( `:` ) and at least one device specifier of some sort, else an error is thrown.
- The sub-device and sub-sub-device syntax attempt to partition the root device
  according to the rules defined by
  `info::partition_property::partition_by_affinity_domain` and
  `info::partition_affinity_domain::next_partitionable`.
  (See the SYCL 2020 specification for a precise definition.)
  The root device is determined by the underlying backend.
- When using the Level Zero backend, see also the documentation of the
  [`ZE_FLAT_DEVICE_HIERARCHY`][ze-env] environment variable because it affects
  how this backend exposes root devices to SYCL.
  For Intel GPUs, the sub-device and sub-sub-device syntax can be used to
  expose tiles or CCSs to the SYCL application as SYCL root devices, however
  the exact mapping is determined by the `ZE_FLAT_DEVICE_HIERARCHY` environment
  variable.
- The semi-colon character ( `;` ) and the exclamation mark character ( `!`  ) are treated specially by many shells, so you may need to enclose the string in quotes if the selection string contains these characters.

[ze-env]: https://spec.oneapi.io/level-zero/latest/core/PROG.html#environment-variables


### `SYCL_DEVICE_ALLOWLIST`

A list of devices and their driver version following the pattern:
`BackendName:XXX,DeviceType:YYY,DeviceVendorId:0xXYZW,DriverVersion:{{X.Y.Z.W}}`.
Also may contain `PlatformVersion`, `DeviceName` and `PlatformName`. There is no
fixed order of properties in the pattern.

## `SYCL_REDUCTION_PREFERRED_WORKGROUP_SIZE`

This environment variable controls the preferred work-group size for reductions on specified device types. Setting this will affect all reductions without an explicitly specified work-group size on devices of types in the value of the environment variable.

The value of this environment variable is a comma separated list of one or more configurations, where each configuration is a pair of the form "`device_type`:`size`" (without the quotes). Possible values of `device_type` are:
- `cpu`
- `gpu`
- `acc`
- `*`

`size` is a positive integer larger than 0.

For a configuration `device_type`:`size` the `device_type` element specifies the type of device the configuration applies to, that is `cpu` is for CPU devices, `gpu` is for GPU devices, and `acc` is for accelerator devices. If `device_type` is `*` the configuration applies to all applicable device types. `size` denotes the preferred work-group size to be used for devices of types specified by `device_type`.

If `info::device::max_work_group_size` on a device on which a reduction is being enqueued is less than the value specified by a configuration in this environment variable, the value of `info::device::max_work_group_size` on that device is used instead.

A `sycl::exception` with `sycl::errc::invalid` is thrown during submission of a reduction kernel in the following cases:
- If the specified device type in any configuration is not one of the valid values.
- If the specified preferred work-group size in any configuration is not a valid integer.
- If the specified preferred work-group size in any configuration is not an integer value larger than 0.
- If any configuration does not have the `:` delimiter.

If this environment variable is not set, the preferred work-group size for reductions is implementation defined.

Note that conflicting configuration tuples in the same list will favor the last entry. For example, a list `cpu:32,gpu:32,cpu:16` will set the preferred work-group size of reductions to 32 for GPUs and 16 for CPUs. This also applies to `*`, for example `cpu:32,*:16` sets the preferred work-group size of reductions on all devices to 16, while `*:16,cpu:32` sets the preferred work-group size of reductions to 32 on CPUs and to 16 on all other devices.

## Range Rounding Environment Variables

For a description of parallel for range rounding in DPC++ see
[Parallel For Range Rounding][range-rounding].

| Environment variable | Values | Description |
| -------------------- | ------ | ----------- |
| `SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING` | Any(\*) | Disables automatic rounding-up of `parallel_for` invocation ranges. |
| `SYCL_PARALLEL_FOR_RANGE_ROUNDING_TRACE`   | Any(\*) | Enables tracing of `parallel_for` invocations with rounded-up ranges. |
| `SYCL_PARALLEL_FOR_RANGE_ROUNDING_PARAMS`  | `MinFactorX:GoodFactor:MinRangeX` | `MinFactorX`: The minimum range that the rounded range should be a multiple of (Default 16)  |
|  |  | `GoodFactor`: The preferred range that the rounded range be a multiple of (Default 32)  |
|  |  | `MinRangeX`: The minimum X dimension of the range such that range rounding is activated (Default 1024) |


## Controlling DPC++ Level Zero Plugin

| Environment variable | Values | Description |
| -------------------- | ------ | ----------- |
| `SYCL_ENABLE_PCI` (Deprecated) | Integer | When set to 1, enables obtaining the GPU PCI address when using the Level Zero backend. The default is 1. This option is kept for compatibility reasons and is immediately deprecated. |
| `SYCL_PI_LEVEL_ZERO_DISABLE_USM_ALLOCATOR` | Any(\*) | Disable USM allocator in Level Zero plugin (each memory request will go directly to Level Zero runtime) |
| `SYCL_PI_LEVEL_ZERO_TRACK_INDIRECT_ACCESS_MEMORY` | Any(\*) | Enable support of the kernels with indirect access and corresponding deferred release of memory allocations in the Level Zero plugin. |

`(*) Note: Any means this environment variable is effective when set to any non-null value.`

## Controlling DPC++ CUDA Plugin

| Environment variable | Values | Description |
| -------------------- | ------ | ----------- |
| `SYCL_PI_CUDA_MAX_LOCAL_MEM_SIZE` | Integer | Specifies the maximum size of a local memory allocation in bytes. If the value exceeds the device's capabilities then a `sycl::runtime_error` is thrown. In order for the full error message to be printed, `SYCL_RT_WARNING_LEVEL=2` must be set. The default value for `SYCL_PI_CUDA_MAX_LOCAL_MEM_SIZE` is determined by the hardware. |

## Controlling DPC++ HIP Plugin

| Environment variable | Values | Description |
| -------------------- | ------ | ----------- |
| `SYCL_PI_HIP_MAX_LOCAL_MEM_SIZE` | Integer | Specifies the maximum size of a local memory allocation in bytes. If the value exceeds the device's capabilities then a `sycl::runtime_error` is thrown. In order for the full error message to be printed, `SYCL_RT_WARNING_LEVEL=2` must be set. The default value for `SYCL_PI_HIP_MAX_LOCAL_MEM_SIZE` is determined by the hardware. |

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
| `SYCL_PREFER_UR` | Integer | If non-0 then run through Unified Runtime if desired backend is supported there. Default is 0.  |
| `SYCL_PI_TRACE` | Described [below](#sycl_pi_trace-options)  | Enable specified level of tracing for PI. |
| `SYCL_QUEUE_THREAD_POOL_SIZE` | Positive integer | Number of threads in thread pool of queue. |
| `SYCL_DEVICELIB_NO_FALLBACK` | Any(\*) | Disable loading and linking of device library images |
| `SYCL_PRINT_EXECUTION_GRAPH` | Described [below](#sycl_print_execution_graph-options) | Print execution graph to DOT text file. |
| `SYCL_DISABLE_EXECUTION_GRAPH_CLEANUP` | Any(\*) | Disable regular cleanup of enqueued (or finished, in case of host tasks) non-leaf command nodes. If disabled, command nodes will be cleaned up only during the destruction of the last remaining memory object used by them. |
| `SYCL_DISABLE_POST_ENQUEUE_CLEANUP` (deprecated) | Any(\*) | Use `SYCL_DISABLE_EXECUTION_GRAPH_CLEANUP` instead. |
| `SYCL_DEVICELIB_INHIBIT_NATIVE` | String of device library extensions (separated by a whitespace) | Do not rely on device native support for devicelib extensions listed in this option. |
| `SYCL_PROGRAM_COMPILE_OPTIONS` | String of valid compile options | Override compile options for all programs. |
| `SYCL_PROGRAM_LINK_OPTIONS` | String of valid link options | Override link options for all programs. |
| `SYCL_PROGRAM_APPEND_COMPILE_OPTIONS` | String of valid compile options | Append to the end of compile options for all programs. |
| `SYCL_PROGRAM_APPEND_LINK_OPTIONS` | String of valid link options | Append to the end of link options for all programs. |
| `SYCL_USE_KERNEL_SPV` | Path to the SPIR-V binary | Load device image from the specified file. If runtime is unable to read the file, `sycl::runtime_error` exception is thrown. The image is assumed to have been created using the `-fno-sycl-dead-args-optimization` option. |
| `SYCL_DUMP_IMAGES` | Any(\*) | Dump device image binaries to file. Control has no effect if `SYCL_USE_KERNEL_SPV` is set. |
| `SYCL_HOST_UNIFIED_MEMORY` | Integer | Enforce host unified memory support or lack of it for the execution graph builder. If set to 0, it is enforced as not supported by all devices. If set to 1, it is enforced as supported by all devices. |
| `SYCL_CACHE_TRACE` | Any(\*) | If the variable is set, messages are sent to std::cerr when caching events or non-blocking failures happen (e.g. unable to access cache item file). |
| `SYCL_PARALLEL_FOR_RANGE_ROUNDING_TRACE` | Any(\*) | Enables tracing of `parallel_for` invocations with rounded-up ranges. |
| `SYCL_PI_SUPPRESS_ERROR_MESSAGE` | Any(\*) | Suppress printing of error message, only used for CI in order not to interrupt errors generated by underlying toolchains; note that the variable only modifies the printing of the error message (error value, name, description and location), the handling of error return code and aborting/throwing behaviour remains unchanged. |

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
| `SYCL_PI_LEVEL_ZERO_SINGLE_THREAD_MODE` | Integer | A single-threaded app has an opportunity to enable this mode to avoid overhead from mutex locking in the Level Zero plugin. A value greater than 0 enables single thread mode. A value of 0 disables single thread mode. The default is 0. |
| `SYCL_PI_LEVEL_ZERO_USM_ALLOCATOR` | [EnableBuffers][;[MaxPoolSize][;[host\|device\|shared:][MaxPoolableSize][,[Capacity][,SlabMinSize]]]...] | EnableBuffers enables pooling for SYCL buffers, default 0, set to 1 to enable. MaxPoolSize is the maximum size of the pool, default 0. MemType is host, device or shared. Other parameters are values specified as positive integers with optional K, M or G suffix. MaxPoolableSize is the maximum allocation size that may be pooled, default 0 for host and shared, 32KB for device. Capacity is the number of allocations in each size range freed by the program but retained in the pool for reallocation, default 0. Size ranges follow this pattern: 64, 96, 128, 192, and so on, i.e., powers of 2, with one range in between. SlabMinSize is the minimum allocation size, 64KB for host and device, 2MB for shared. Example: SYCL_PI_LEVEL_ZERO_USM_ALLOCATOR=1;32M;host:1M,4,64K;device:1M,4,64K;shared:0,0,2M|
| `SYCL_PI_LEVEL_ZERO_BATCH_SIZE` | Integer | Sets a preferred number of compute commands to batch into a command list before executing the command list. A value of 0 causes the batch size to be adjusted dynamically. A value greater than 0 specifies fixed size batching, with the batch size set to the specified value. The default is 0. |
| `SYCL_PI_LEVEL_ZERO_COPY_BATCH_SIZE` | Integer | Sets a preferred number of copy commands to batch into a command list before executing the command list. A value of 0 causes the batch size to be adjusted dynamically. A value greater than 0 specifies fixed size batching, with the batch size set to the specified value. The default is 0. |
| `SYCL_PI_LEVEL_ZERO_FILTER_EVENT_WAIT_LIST` | Integer | When set to 0, disables filtering of signaled events from wait lists when using the Level Zero backend. The default is 0. |
| `SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE` | Any(\*) | This environment variable enables users to control use of copy engines for copy operations. If the value is an integer, it will allow the use of copy engines, if available in the device, in Level Zero plugin to transfer SYCL buffer or image data between the host and/or device(s) and to fill SYCL buffer or image data in device or shared memory. The value of this environment variable can also be a pair of the form "lower_index:upper_index" where the indices point to copy engines in a list of all available copy engines. The default is 0:0 when immediate command lists are being used on the device and 1 otherwise. (Also see description of SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS). |
| `SYCL_PI_LEVEL_ZERO_USE_COMPUTE_ENGINE` | Integer | It can be set to an integer (>=0) in which case all compute commands will be submitted to the command-queue with the given index in the compute command group. If it is instead set to a negative value then all available compute engines may be used. The default value is "0" |
| `SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_D2D_COPY` (experimental) | Integer | Allows the use of copy engine, if available in the device, in Level Zero plugin for device to device copy operations. The default is 0. This option is experimental and will be removed once heuristics are added to make a decision about use of copy engine for device to device copy operations. |
| `SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS` | Any(\*) | Enable support of device-scope events whose state is not visible to the host. If enabled mode is SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=1 the Level Zero plugin would create all events having device-scope only and create proxy host-visible events for them when their status is needed (wait/query) on the host. If enabled mode is SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=2 the Level Zero plugin would create all events having device-scope and add proxy host-visible event at the end of each command-list submission. The default is 0, meaning all events have host visibility. SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS is ignored when using immediate command lists (SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS = 1) and all events use default scope of 0. |
| `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS` | Integer | When set to a positive value enables use of Level Zero immediate commandlists, which means there is no batching and all commands are immediately submitted for execution. When set to 1, unique immediate commandlists are created for each SYCL queue. When set to 2, unique immediate commandlists are created per host thread per SYCL queue. Default is 1 on Intel® Data Center GPU Max Series running Linux and 0 elsewhere. |
| `SYCL_PI_LEVEL_ZERO_USE_MULTIPLE_COMMANDLIST_BARRIERS` | Integer | When set to a positive value enables use of multiple Level Zero commandlists when submitting barriers. Default is 1. |
| `SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_FILL` | Integer | When set to a positive value enables use of a copy engine for memory fill operations. Default is 0. |
| `SYCL_PI_LEVEL_ZERO_SINGLE_ROOT_DEVICE_BUFFER_MIGRATION` | Integer | When set to "0" tells to use single root-device allocation for all devices in a context where all devices have same root. Otherwise performs regular buffer migration. Default is 1. |
| `SYCL_PI_LEVEL_ZERO_REUSE_DISCARDED_EVENTS` | Integer |  When set to a positive value enables the mode when discarded Level Zero events are reset and reused in scope of the same in-order queue based on the dependency chain between commands. Default is 1. |
| `SYCL_PI_LEVEL_ZERO_EXPOSE_CSLICE_IN_AFFINITY_PARTITIONING` (Deprecated) | Integer | When set to non-zero value exposes compute slices as sub-sub-devices in `sycl::info::partition_property::partition_by_affinity_domain` partitioning scheme. Default is zero meaning that they are only exposed when partitioning by `sycl::info::partition_property::ext_intel_partition_by_cslice`. This option is introduced for compatibility reasons and is immediately deprecated. New code must not rely on this behavior. Also note that even if sub-sub-device was created using `partition_by_affinity_domain` it would still be reported as created via partitioning by compute slices. |
| `SYCL_PI_LEVEL_ZERO_COMMANDLISTS_CLEANUP_THRESHOLD` | Integer | If non-negative then the threshold is set to this value. If negative, the threshold is set to INT_MAX. Whenever the number of command lists in a queue exceeds this threshold, an attempt is made to cleanup completed command lists for their subsequent reuse. The default is 20. |
| `SYCL_PI_LEVEL_ZERO_IMMEDIATE_COMMANDLISTS_EVENT_CLEANUP_THRESHOLD` | Integer | If non-negative then the threshold is set to this value. If negative, the threshold is set to INT_MAX. Whenever the number of events associated with an immediate command list exceeds this threshold, a check is made for signaled events and these events are recycled. Setting this threshold low causes events to be checked more often, which could result in unneeded events being recycled sooner. However, more frequent event status checks may cost time. The default is 1000. |
| `SYCL_PI_LEVEL_ZERO_USM_RESIDENT` | Integer | Bit-mask controls if/where to make USM allocations resident at the time of allocation. Input value is of the form 0xHSD, where 4-bits of D control device allocations, 4-bits of S control shared allocations, and 4-bits of H control host allocations. Each 4-bit component is holding one of the following values: "0" - then no special residency is forced, "1" - then allocation is made resident at the device of allocation, or "2" - then allocation is made resident on all devices in the context of allocation that have P2P access to the device of allocation. Default is 0x002, i.e. force full residency for device allocations only. |
| `SYCL_PI_LEVEL_ZERO_USE_NATIVE_USM_MEMCPY2D` | Integer | When set to a positive value enables the use of Level Zero USM 2D memory copy operations. Default is 0. |

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
[range-rounding]: https://github.com/intel/llvm/blob/sycl/doc/design/ParallelForRangeRounding.md
