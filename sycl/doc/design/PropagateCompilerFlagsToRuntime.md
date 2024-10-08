# Propagation of optimization levels used by front-end compiler to backend

In order to ease the process of debugging, there is a user requirement to
compile different modules with different levels of optimization. This document
proposes a compiler flow that will enable propagation of compiler options
specified from front-end to the runtimes and eventually to the backend.
Currently, only `O0`/`O1`/`O2`/`O3` options are handled.
Please note that this document only describes support for JIT path. AOT path
support will be added later.

## Background

When building an application with several source and object files, it is
possible to specify the optimization parameters individually for each source
file/object file (for each invocation of the DPCPP compiler). The SYCL runtime
should pass the original optimization options (e.g. `-O0` or `-O2`) used when
building an object file to the device backend compiler. This will improve the
debugging experience by selectively disabling/enabling optimizations for each
source file, and therefore achieving better debuggability and better performance
as needed.

The current behavior is that the optimization level option is captured at link
time and converted into its backend-specific equivalent. This option is
propagated to the backend. For example, If `-O0` option is specified during
link-time when using the OpenCL backend, the SYCL runtime will pass
`-cl-opt-disable` option to the backend device compiler for all modules
essentially disabling optimizations globally. Otherwise, if the `-O0`
option is not specified for linker, it will not pass `-cl-opt-disable` option at
all, therefore making the kernels mostly undebuggable, regardless of the
original front-end compiler options. Link-time capturing of optimization option
is the essence of the current implementation and this leads to loss of 
information about the compile-time options. Proposed design aims to rectify this
behavior.

Here is an example that demonstrates this pain point:

```
clang++ -c test_host.cpp -o test_host.o
clang++ -c -fsycl test_device_1.cpp -o test_device_1.o
clang++ -c -fsycl -g -O0 test_device_2.cpp -o test_device_2.o
clang++ -fsycl -g test_host.o test_device_1.o test_device_2.o -o test
```

In this scenario, the fat binary is 'test' and there are no compilation flags
sent across to the backend compiler. Though the user wanted to have full
debuggability with test_device_2.cpp module, some of the debuggability is lost.

Another scenario is shown below:

```
clang++ -c -g -O0 -fsycl test.cpp -o test.o
clang++ -g -fsycl test.o -o test
```

In this scenario, the fat binary is 'test' and there are no compilation flags
sent across to the backend compiler. Though the user wanted to have full
debuggability with test.cpp module, some of the debuggability is lost. The user 
was not able to set a breakpoint inside device code.

## Requirements

In order to support module-level debuggability, the user will compile different
module files with different levels of optimization. These optimization levels
must be preserved and made use of during the backend compilation. The following
is a key requirement for this feature.
- If the user specifies `-Ox` as a front-end compile option for a particular
module, this option must be converted to appropriate backend option and then
propagated fo use during backend JIT compilation.

The following table specifies the appropriate backend options for level-zero and
OpenCL backends.

| Front-end option | L0 backend option | OpenCL backend option |
| ---------------- | ----------------- | --------------------- |
|      -O0         |  -ze-opt-disable  |   -cl-opt-disable     |
|      -O1         |  -ze-opt-level=2  |   /* no option */     |
|      -O2         |  -ze-opt-level=2  |   /* no option */     |
|      -O3         |  -ze-opt-level=2  |   /* no option */     |


## Proposed design

This chapter discusses changes required in various stages of the compilation
pipeline.


### Changes to the clang front-end

For each function in SYCL device code, we add a new function attribute that is
named `sycl-optlevel`. Value of this attribute is set to the optimization level
which was used to compile the overlying module.

### Changes to the sycl-post-link tool

During device code split performed in the `sycl-post-link` tool, optimization
level attribute `sycl-optlevel` is treated as an optional feature,
i.e. device code split algorithm ensures that no kernels with different values
of sycl-optlevel are bundled into the same device image. See also optional
kernel features [design document](https://github.com/intel/llvm/blob/sycl/sycl/doc/design/OptionalDeviceFeatures.md#changes-to-the-post-link-tool).
The `sycl-post-link` tool also adds a new property into the 
`SYCL/misc properties` property set for each device code module. This entry will
be used to store the optimization level. Name of this property is `optLevel` and
the value is stored as a 32-bit integer. If there is a module where the user did
not specify an optimization module, there is no new entry in the property set.

### Changes to the SYCL runtime

In the SYCL runtime, the device image properties can be accessed to extract the
associated optimization level. Once the optimization level is available, it is
converted to its equivalent frontend option string
(`-O0`, `-O1`, `-O2`, or `-O3`). This frontend option string is passed into a
query that is made to the plugin to identify the correct backend option. This
backend option is added to the existing list of compiler options and is sent to
the backend.

### Changes to the plugin

A new plugin API has been added. It takes the frontend option string as input in
string format and returns `pi_result`. A string format is used for sending the
frontend option so that this API can be used for querying other frontend
options as well. The signature of this API is as follows:

```C++
pi_result piPluginGetBackendOption(pi_platform platform,
                                   const char *frontend_option,
                                   const char **backend_option);
```

In the level-zero and OpenCL plugins, the table provided in the 'Requirements'
section is used as a guide to identify the appropriate backend option.
The option is returned in `backend_option`. For other plugins (HIP, cuda),
empty string is returned. This API returns `PI_SUCCESS` for
valid inputs (frontend_option != ""). For invalid inputs, it returns
`PI_ERROR_INVALID_VALUE`.
