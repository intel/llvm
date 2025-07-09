# Fuzzing plan for intel/llvm

Fuzzing (or fuzz testing) is an automated testing approach where inputs are
randomly generated. It often leads to passing unexpected and invalid inputs
which in turn uncovers various corner cases that weren't considered during
development or regular testing.

The main product which is being developed at intel/llvm repo is a SYCL
implementation. At high-level, it consists of two components: a compiler and a
runtime, and therefore this document will be divided into two major sections
covering those components. Those components are essentially the only entry
points through which a user can interact with our product.

## SYCL Runtime

SYCL runtime is a library which implements SYCL API, but besides that API it
also has multiple configuration options which can be tweaked through environment
variables and config file.

### Fuzzing environment variables

Every environment variable in the [documentation][sycl-rt-env-variables] should
should be fuzzed.

The most interesting of the environment variables are ones which expect data in
a certain format, like `ONEAPI_DEVICE_SELECTOR` or `SYCL_CACHE_TRESHOLD`.

[sycl-rt-env-variables]: https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md

### Fuzzing sycl config file

Instead of tweaking SYCL Runtime behavior through environment variables, the
same can be done by providing a config file. We don't seem to have any
documentation on it except the source code for the functionality which is
located in `sycl/source/detail/config.cpp` file.

There is a prototype for the sycl config file fuzzer available at
https://github.com/intel/llvm/pull/16308

### Fuzzing API entry points

TODO: think more about this section, i.e. whether or not we want to fuzz SYCL
APIs. Not every of them accepts "raw" data, but instead expects some SYCL
objects returned from previous API calls. However, there are still plenty of
APIs which accept raw pointers and other fundamental data types. Note: to
properly fuzz them structure-aware fuzzing may be needed.

## SYCL Compiler

SYCL compiler is based on the [upstream LLVM compiler project][llvm-project]
and it is an enormously huge codebase. Some of LLVM components have been re-used
without any modifications to them at all. Some of LLVM components were slightly
tweaked or significantly modified and there are components which are completely
new and only exist in our implementation.

For every re-used component we should be able to benefit from existing fuzz
testing written for those. Upstream documentation has them documented
[here][llvm-fuzzers].

[llvm-fuzzers]: https://llvm.org/docs/FuzzingLLVM.html
[llvm-project]: https://github.com/llvm/llvm-project

However, even though we could re-use existing fuzzers, we can't just rely on
someone else running them on the upstream codebase, because those runs won't
cover any customizations we made (including new components like optimization
passes which we added only in our downstream).

There are also some unique components which may require special fuzzers.
Sections below will go through components that we have and describe in more
details like what should we fuzz and if we already have an existing fuzzer for
that.

There is also the [intel/yarpgen](https://github.com/intel/yarpgen) project that
can be used to fuzz SYCL compilers. It generate random programs (of certain
structure) to detect weaknesses and bugs in optimization passes.

### Command line options

There are plenty of SYCL-specific command line options and there are multiple of
those which are not mere flag, but expect a user-provided value in a certain
format.

Those options should be fuzzed as well to ensure proper error handling of
various weird inputs.

### SYCL-specific passes

We have developed a number of passes to implement different SYCL features. They
all can be found in the `llvm/lib/SYCLLowerIR` folder. We don't need a dedicated
fuzzer for every pass, but we can instead re-use existing LLVM fuzzer intended
for compiler passes to cover those.

### SYCL-specific tools

As of now, we still use legacy offloading flow which involves multiple custom
tools and some custom data format to communicate information between compiler
phases.

Even though strictly speaking, we should probably fuzz those as well, we are
going to replace that with so-called new offloading model which significantly
simplifies the flow by reducing amount of tools we have and therefore amount of
custom data formats used to communicate between those.

#### SPIRV-LLVM-Translator

Going forward, this tool may be replaced by a SPIRV Backend, but ultimately a
step of translating LLVM IR into SPIR-V format will stay in place.

SPIR-V is a way stricter format and it moves at a slower pace than LLVM does and
it is important that we have fuzzing for this phase as well.
