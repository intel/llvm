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
| SYCL_BE | PI_OPENCL, PI_CUDA | Force SYCL RT to consider only devices of the specified backend during the device selection. |
| SYCL_DEVICE_TYPE | One of: CPU, GPU, ACC, HOST | Force SYCL to use the specified device type. If unset, default selection rules are applied. If set to any unlisted value, this control has no effect. If the requested device type is not found, a `cl::sycl::runtime_error` exception is thrown. If a non-default device selector is used, a device must satisfy both the selector and this control to be chosen. This control only has effect on devices created with a selector. |
| SYCL_PROGRAM_COMPILE_OPTIONS | String of valid OpenCL compile options | Override compile options for all programs. |
| SYCL_PROGRAM_LINK_OPTIONS | String of valid OpenCL link options | Override link options for all programs. |
| SYCL_USE_KERNEL_SPV | Path to the SPIR-V binary | Load device image from the specified file. If runtime is unable to read the file, `cl::sycl::runtime_error` exception is thrown.|
| SYCL_DUMP_IMAGES | Any(\*) | Dump device image binaries to file. Control has no effect if SYCL_USE_KERNEL_SPV is set. |
| SYCL_PRINT_EXECUTION_GRAPH | Described [below](#sycl_print_execution_graph-options) | Print execution graph to DOT text file. |
| SYCL_DISABLE_EXECUTION_GRAPH_CLEANUP | Any(\*) | Disable cleanup of finished command nodes at host-device synchronization points. |
| SYCL_THROW_ON_BLOCK | Any(\*) | Throw an exception on attempt to wait for a blocked command.  |
| SYCL_DEVICELIB_INHIBIT_NATIVE | String of device library extensions (separated by a whitespace) | Do not rely on device native support for devicelib extensions listed in this option. |
| SYCL_DEVICE_ALLOWLIST | A list of devices and their minimum driver version following the pattern: DeviceName:{{XXX}},DriverVersion:{{X.Y.Z.W}}. Also may contain PlatformName and PlatformVersion | Filter out devices that do not match the pattern specified. Regular expression can be passed and the DPC++ runtime will select only those devices which satisfy the regex. |
| SYCL_QUEUE_THREAD_POOL_SIZE | Positive integer | Number of threads in thread pool of queue. |
`(*) Note: Any means this environment variable is effective when set to any non-null value.`

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
