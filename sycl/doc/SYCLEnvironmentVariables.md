# Overview

This file describes environment variables that are having effect on SYCL compiler and run-time.

# Controlling SYCL RT

**Warning:** the environment variables described in this document are used for
development and debugging of SYCL runtime and compiler. Their semantics are
subject to change. Do not rely on these variables in production code.

| Environment variable | Values | Description |
| -------------------- | ------ | ----------- |
| SYCL_PI_TRACE | Any(*) | Force tracing of PI calls to stderr. |
| SYCL_BE | PI_OPENCL, PI_OTHER | When SYCL RT is built with PI this controls which plugin to use. Default value is PI_OPENCL. |
| SYCL_DEVICE_TYPE | One of: CPU, GPU, ACC, HOST | Force SYCL to use the specified device type. If unset, default selection rules are applied. If set to any unlisted value, this control has no effect. If the requested device type is not found, a `cl::sycl::runtime_error` exception is thrown. If a non-default device selector is used, a device must satisfy both the selector and this control to be chosen. This control only has effect on devices created with a selector. |
| SYCL_PROGRAM_COMPILE_OPTIONS | String of valid OpenCL compile options | Override compile options for all programs. |
| SYCL_PROGRAM_LINK_OPTIONS | String of valid OpenCL link options | Override link options for all programs. |
| SYCL_USE_KERNEL_SPV | Path to the SPIR-V binary | Load device image from the specified file. If runtime is unable to read the file, `cl::sycl::runtime_error` exception is thrown.|
| SYCL_DUMP_IMAGES | Any(*) | Dump device image binaries to file. Control has no effect if SYCL_USE_KERNEL_SPV is set. |
| SYCL_PRINT_EXECUTION_GRAPH | Described [below](#sycl_print_execution_graph-options) | Print execution graph to DOT text file. |
| SYCL_THROW_ON_BLOCK | Any(*) | Throw an exception on attempt to wait for a blocked command.  |
| SYCL_DEVICELIB_INHIBIT_NATIVE | String of device library extensions (separated by a whitespace) | Do not rely on device native support for devicelib extensions listed in this option. |

`(*) Note: Any means this environment variable is effective when set to any non-null value.`

## SYCL_PRINT_EXECUTION_GRAPH Options

SYCL_PRINT_EXECUTION_GRAPH can accept one or more comma separated values from table below

| Option | Description |
| ------ | ----------- |
| before_addCG | print graph before addCG method |
| after_addCG | print graph after addCG method |
| before_addCopyBack | print graph before addCopyBack method |
| after_addCopyBack | print graph after addCopyBack method |
| before_addHostAcc | print graph before addHostAccessor method |
| after_addHostAcc | print graph after addHostAccessor method |
| always | print graph before and after each of the above methods |

