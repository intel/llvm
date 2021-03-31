# Environment Variables

This document describes environment variables that are having effect on DPC++
compiler and runtime.

## Controlling DPC++ RT

**Warning:** the environment variables described in this document are used for
development and debugging of DPC++ compiler and runtime. Their semantics are
subject to change. Do not rely on these variables in production code.

| Environment variable | Values | Description |
| -------------------- | ------ | ----------- |
| SYCL_PI_TRACE | Described [below](#sycl_pi_trace-options)  | Enable specified level of tracing for PI. |
| SYCL_BE (deprecated) | PI_OPENCL, PI_LEVEL_ZERO, PI_CUDA | Force SYCL RT to consider only devices of the specified backend during the device selection. We are planning to deprecate SYCL_BE environment variable in the future. The specific grace period is not decided yet. Please use the new env var SYCL_DEVICE_FILTER instead. |
| SYCL_DEVICE_TYPE (deprecated) | One of: CPU, GPU, ACC, HOST | Force SYCL to use the specified device type. If unset, default selection rules are applied. If set to any unlisted value, this control has no effect. If the requested device type is not found, a `cl::sycl::runtime_error` exception is thrown. If a non-default device selector is used, a device must satisfy both the selector and this control to be chosen. This control only has effect on devices created with a selector. We are planning to deprecate SYCL_DEVICE_TYPE environment variable in the future. The specific grace period is not decided yet. Please use the new env var SYCL_DEVICE_FILTER instead. |
| SYCL_DEVICE_FILTER | backend:device_type:device_num | See Section [SYCL_DEVIC_EFILTER](#sycl_device_filter)  below. |
| SYCL_PROGRAM_COMPILE_OPTIONS | String of valid OpenCL compile options | Override compile options for all programs. |
| SYCL_PROGRAM_LINK_OPTIONS | String of valid OpenCL link options | Override link options for all programs. |
| SYCL_USE_KERNEL_SPV | Path to the SPIR-V binary | Load device image from the specified file. If runtime is unable to read the file, `cl::sycl::runtime_error` exception is thrown.|
| SYCL_DUMP_IMAGES | Any(\*) | Dump device image binaries to file. Control has no effect if SYCL_USE_KERNEL_SPV is set. |
| SYCL_PRINT_EXECUTION_GRAPH | Described [below](#sycl_print_execution_graph-options) | Print execution graph to DOT text file. |
| SYCL_DISABLE_EXECUTION_GRAPH_CLEANUP | Any(\*) | Disable cleanup of finished command nodes at host-device synchronization points. |
| SYCL_THROW_ON_BLOCK | Any(\*) | Throw an exception on attempt to wait for a blocked command.  |
| SYCL_DEVICELIB_INHIBIT_NATIVE | String of device library extensions (separated by a whitespace) | Do not rely on device native support for devicelib extensions listed in this option. |
| SYCL_DEVICE_ALLOWLIST | A list of devices and their driver version following the pattern: DeviceName:{{XXX}},DriverVersion:{{X.Y.Z.W}}. Also may contain PlatformName and PlatformVersion | Filter out devices that do not match the pattern specified. Regular expression can be passed and the DPC++ runtime will select only those devices which satisfy the regex. Special characters, such as parenthesis, must be escaped. More than one device can be specified using the piping symbol "\|".|
| SYCL_QUEUE_THREAD_POOL_SIZE | Positive integer | Number of threads in thread pool of queue. |
| SYCL_DEVICELIB_NO_FALLBACK | Any(\*) | Disable loading and linking of device library images |
| SYCL_PI_LEVEL_ZERO_MAX_COMMAND_LIST_CACHE | Positive integer | Maximum number of oneAPI Level Zero Command lists that can be allocated with no reuse before throwing an "out of resources" error. Default is 20000, threshold may be increased based on resource availabilty and workload demand. |
| SYCL_PI_LEVEL_ZERO_DISABLE_USM_ALLOCATOR | Any(\*) | Disable USM allocator in Level Zero plugin (each memory request will go directly to Level Zero runtime) |
| SYCL_PI_LEVEL_ZERO_BATCH_SIZE | Integer | Sets a preferred number of commands to batch into a command list before executing the command list. A value of 0 causes the batch size to be adjusted dynamically. A value greater than 0 specifies fixed size batching, with the batch size set to the specified value. The default is 0. |
| SYCL_PI_LEVEL_ZERO_FILTER_EVENT_WAIT_LIST | Integer | When set to 0, disables filtering of signaled events from wait lists when using the Level Zero backend. The default is 1. |
| SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE | Integer | Allows the use of copy engine, if available in the device, in Level Zero plugin to transfer SYCL buffer or image data between the host and/or device(s) and to fill SYCL buffer or image data in device or shared memory. The default is 1. |
| SYCL_PARALLEL_FOR_RANGE_ROUNDING_TRACE | Any(\*) | Enables tracing of parallel_for invocations with rounded-up ranges. |
| SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING | Any(\*) | Disables automatic rounding-up of parallel_for invocation ranges. |
| SYCL_ENABLE_PCI | Integer | When set to 1, enables obtaining the GPU PCI address when using the Level Zero backend. The default is 0. |
| SYCL_HOST_UNIFIED_MEMORY | Integer | Enforce host unified memory support or lack of it for the execution graph builder. If set to 0, it is enforced as not supported by all devices. If set to 1, it is enforced as supported by all devices. |

`(*) Note: Any means this environment variable is effective when set to any non-null value.`

### SYCL_DEVICE_FILTER

Limits the SYCL RT to use only a subset of the system's devices. Setting this environment variable affects all of the device query functions (platform::get_devices() and platform::get_platforms()) and all of the device selectors. The value of this environment variable is a comma separated list of filters, where each filter is a triple of the form "backend:device_type:device_num" (without the quotes). Each element of the triple is optional, but each filter must have at least one value. Possible values of "backend" are "host", "level_zero", "opencl", "cuda", or "\*". Possible values of "device_type" are "host", "cpu", "gpu", "acc", or "\*". Device_num is an integer that indexes the enumeration of devices from the sycl-ls utility tool, where the first device in that enumeration has index zero in each backend. For example, SYCL_DEVICE_FILTER=2 will return all devices with index '2' from all different backends. If multiple devices satisfy this device number (e.g., GPU and CPU devices can be assigned device number '2'), then default_selector will choose the device with the highest heuristic point. Assuming a filter has all three elements of the triple, it selects only those devices that come from the given backend, have the specified device type, AND have the given device index. If more than one filter is specified, the RT is restricted to the union of devices selected by all filters. The RT does not includes the "host" backend and the host device automatically unless one of the filter explicitly specify the "host" device type. Therefore, SYCL_DEVICE_FILTER=host should be set to enforce SYCL to use the host device only. Note that all device selectors will throw an exception if the filtered list of devices does not include a device that satisfies the selector. For instance, SYCL_DEVICE_FILTER=cpu,level_zero will cause host_selector() to throw an exception. SYCL_DEVICE_FILTER also imits loading only specified plugins into the SYCL RT. In particular, SYCL_DEVICE_FILTER=level_zero will cause the cpu_selector to throw an exception since SYCL RT will only load the level_zero backend which does not support any CPU devices at this time. When multiple devices satisfy the filter (e..g, SYCL_DEVICE_FILTER=gpu), only one of them will be selected.

### SYCL_PRINT_EXECUTION_GRAPH Options

SYCL_PRINT_EXECUTION_GRAPH can accept one or more comma separated values from the table below

| Option | Description |
| ------ | ----------- |
| before_addCG | print graph before addCG method |
| after_addCG | print graph after addCG method |
| before_addCopyBack | print graph before addCopyBack method |
| after_addCopyBack | print graph after addCopyBack method |
| before_addHostAcc | print graph before addHostAccessor method |
| after_addHostAcc | print graph after addHostAccessor method |
| always | print graph before and after each of the above methods |

### SYCL_PI_TRACE Options

SYCL_PI_TRACE accepts a bit-mask. Supported tracing levels are in the table below

| Option | Description |
| ------ | ----------- |
| 1 | Enable basic tracing, which is tracing of PI plugins/devices discovery |
| 2 | Enable tracing of the PI calls |
| -1 | Enable all levels of tracing |
