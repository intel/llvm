# Propagation of optimization levels used by front-end compiler to backend compiler

In order to ease the process of debugging, there is a user requirement to compile different modules with different levels of optimization. This document proposes a compiler flow that will enable propagation of compiler options specified for front-end to the runtimes and eventually to the backend. Currently, only `O0`/`O1`/`O2`/`O3` options are handled.

**NOTE**: This is not a final version. The document is still in progress.

## Background

When building an application with several source and object files, it is possible to specify the optimization parameters individually for each source file/object file (for each invocation of the DPCPP compiler). The SYCL runtime should pass the original optimization options (e.g. `-O0` or `-O2`) used when building an object file to the device backend compiler. This will improve the debugging experience by selectively disabling/enabling optimizations for each source file, and therefore achieving better debuggability and better performance as needed.

The current behavior is that the device backend optimization options can be propagated to the backend by setting the environment variable `SYCL_PROGRAM_COMPILE_OPTIONS`. For example, If `-O0` option is specified when using the OpenCL backend, the SYCL runtime will pass `-cl-opt-disable` option to the backend device compiler for {*}all modules{*} essentially disabling optimizations globally. Otherwise, if the `-O0` option is not specified for linker, it will not pass `-cl-opt-disable` option at all, therefore making the kernels mostly undebuggable, regardless of the original front-end compiler options.

Here is an example that demonstrates this pain point:

```
clang++ -c test_host.cpp -o test_host.o
clang++ -c -fsycl test_device_1.cpp -o test_device_1.o
clang++ -c -fsycl -O0 test_device_2.cpp -o test_device_2.o
clang++ -fsycl -o test test_host.o test_device_1.o test_device_2.o
```

In this scenario, the fat binary is 'test' and there are no compilation flags sent across to the backend compiler. Though the user wanted to have full debuggability with test_device_2.c module, some of the debuggability is lost.

Another scenario is shown below:

```
clang++ -c -O0 -fsycl -g test.cpp -o test.o
clang++ -fsycl test.o -o test
```

In this scenario, the fat binary is 'test' and there are no compilation flags sent across to the backend compiler. Though the user wanted to have full debuggability with test.cpp module, some of the debuggability is lost. The user was not able to set a breakpoint inside device code.

## Requirements

In order to support module-level debuggability, the user will compile different module files with different levels of optimization. These optimization levels must be preserved and made use of during the backend compilation. Following are the requirements for this feature.
- If the user specifies `-Ox` as a front-end compile option for a particular module, this option must be preserved during backend JIT compilation.
- If the user specifies `-Ox` option using the environment variable, this option will override any front-end compile option and the new option will be preserved during JIT compilation.
- If the user specifies `-O0` option, SYCL runtime needs to pass the appropriate backend option to JIT compilation stages.

The following table specifies the appropriate backend options for level-zero and OpenCL backends.

| Front-end option | L0 backend option | OpenCL backend option |
| ---------------- | ----------------- | --------------------- |
|      -O0         |  -ze-opt-disable  |   -cl-opt-disable     |
|      -O1         |  -ze-opt-level=1  |   /* no option */     |
|      -O2         |  -ze-opt-level=1  |   /* no option */     |
|      -O3         |  -ze-opt-level=2  |   /* no option */     |


## Proposed design

This chapter discusses changes required in various stages of the compilation pipeline.


### Changes to the clang front-end

For each SYCL kernel, we add a new function attribute that is named `sycl-optlevel`. Value of this attribute is set to the optimization level which was used to compile the overlying module.

### Changes to the sycl-post-link tool

During `sycl-post-link` stage, a set of optional kernel features are combined to form a hash value and this hash value is used as a key to split a module into multiple sub-modules. Current list of optional kernel features include: (1) SYCL aspects (2) `large-grf` mode (3) `reqd-work-group-size`. In this design, we add the optimization level associated with the kernel into this list. This helps us to split the kernels based on their optimization level.
The `sycl-post-link` tool also adds a new property into the `SYCL/misc properties` property set for each device code module. This entry will be used to store the optimization level. Name of this property is 'optLevel' and the value is stored as a 32-bit integer. If there is a module where user did not specify an optimization module, there is no new entry in the property set.

### Changes to the SYCL runtime

In the SYCL runtime, the device image properties can be accessed to extract the associated optimization level. Once the optimization level is available, a query is made to identify the correct backend option. The table provided in the 'Requirements' section is used as a guide to identify the appropriate backend option. This backend option is added to the existing list of compiler options and is sent to the backend.