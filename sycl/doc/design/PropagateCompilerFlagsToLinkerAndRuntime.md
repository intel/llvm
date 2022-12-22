# Propagation of optimization levels used by front-end compiler to linker and backend compiler

In order to ease the process of debugging, there is a user requirement to compile different modules with different levels of optimization. This document proposes a compiler flow that will enable propagation of compiler options specified for front-end to the linker and runtimes and eventually to the backend. Currently, only O0/O1/O2/O3 options are handled.

**NOTE**: This is not a final version. The document is still in progress.

## Background

When building an application with several source and object files, it should be possible to specify the optimization parameters individually for each source file/object file (for each invocation of the DPCPP compiler). The linker should pass the original optimization options (e.g. -O0 or -O2) used when building an object file to the device backend compiler (IGC compiler). This will improve the debugging experience by selectively disabling/enabling optimizations for each source file, and therefore achieving better debuggability and better performance as needed.

The current behavior, is that the device backend optimization options are determined by the linker optimization options. If the -O0 option is specified for linker, the linker will pass -cl-opt-disable option to IGC for {*}all kernels{*}, essentially disabling optimizations globally. Otherwise, if the -O0 option is not specified for linker, it will not pass -cl-opt-disable option at all, therefore making the kernels mostly undebuggable, regardless of the original front-end compiler options.

Here is an example that demonstrates this pain point:

```
icx -c -fsycl test1.c -o test1
icx -c -O0 -fsycl test2.c -o test2
icx -fsycl -o test test1.o test2.o
```

In this scenario, the fat binary is 'test' and there are no compilation flags sent across to the backend compiler. Though the user wanted to have full debuggability with test2.c module, some of the debuggablity is lost.

Another scenario is shown below:

```
icpx -c -O0 -fsycl -g test.cpp -o test.o
icpx -fsycl test.o
```

In this scenario, the fat binary is 'test' and there are no compilation flags sent across to the backend compiler. Though the user wanted to have full debuggability with test.cpp module, some of the debuggablity is lost. The user was not able to set a breakpoint inside device code.

## Requirements

In order to support module-level debuggability, the user will compile different module files with different levels of optimization. These optimization levels must be preserved and made use of during every stage of compilation. Following are the requirements for this feature.
- If the user specifies '-Ox' as a front-end compile option for a particular module, this option must be preserved during compilation, linking, AOT compilation as well as JIT compilation.
- If the user specifies '-Ox' option as a front-end linker option, this option will override any front-end compile options and the linker option will be preserved during AOT and JIT compilation.
- If the user specifies '-O0' option, we need to pass '-cl-opt-disable' to AOT and JIT compilation stages.

## Use case

Following is a possible use case:

```
A list of modules:
test1.cpp
test2.cpp
test3.cpp
```

```
Following are the compilation steps:
# compiling
icpx -c -O0 -fsycl test1.cpp -o test1.o
icpx -c -O3 -fsycl test2.cpp -o test2.o
icpx -c -fsycl test3.cpp -o test3.o
# linking
icpx -o test -fsycl test1.o test2.o test3.o
# JIT compilation (For GPU backends, this calls igc-standalone compiler in the background)
./test
```

Since we have three modules with three different compiler options, we will need to end up with three device binaries, each with their own compiler option specified.

## Proposed design

Following are changes required in various stages of the compilation pipeline:
- Front-end code generation: For each SYCL kernel, identify the compilation option. Add an appropriate attribute to that kernel. Name of that attribute is 'sycl-device-compile-optlevel'.
- During the llvm-link stage, all modules are linked into a single module. This is an existing behavior.
- During sycl-post-link stage, we first split the kernels into multiple modules based on their optimization level. For each split module, an entry corresponding to its optimization level is made in its .props file.
- During ocloc call generation, the .props file will be parsed and appropriate option will be added to the list of compiler options.
- In SYCL runtime, logic will be added to program manager to parse the .props file, extract the optimization level, and add '-cl-opt-disable' if the optimization level is 0. Otherwise, we do nothing.
