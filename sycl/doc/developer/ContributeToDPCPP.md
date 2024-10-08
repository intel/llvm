# Contributing to DPC++

## General guidelines

Read [CONTRIBUTING.md](https://github.com/intel/llvm/blob/sycl/CONTRIBUTING.md) first.

## Maintaining stable ABI/API

All changes made to the DPC++ compiler and runtime library should generally
preserve existing ABI/API and contributors should avoid making incompatible
changes. One of the exceptions is experimental APIs, clearly marked so by
namespace or related specification.
If you wish to propose a new experimental DPC++ extension then read
[README-process.md](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/README-process.md).

Another exceptional case is the transition from SYCL 1.2.1 to SYCL 2020
standard.

Deprecation of older APIs is happening in the following order:

- Newer API implemented and covered with tests
- Deprecation warning is added to older API
- Wait some time allowing users to migrate their codebases (exact period depends
  on the change made)
- Old API is removed in headers
- Old API is removed in library

See [ABI Policy Guide](ABIPolicyGuide.md) for more information.

## Project build and local testing

See [Get Started Guide instructions](../GetStartedGuide.md)

## Commit message

For any DPC++-related commit, the `[SYCL]` tag should be present in the
commit message title. To a reasonable extent, additional tags can be used
to signify the component changed, e.g.: `[PI]`, `[CUDA]`, `[Doc]`.

## Using \<iostream\> 

According to 
[LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html#include-iostream-is-forbidden),
the use `#include <iostream>` is forbidden in library files. Instead, the
sycl/detail/iostream_proxy.hpp header offers the functionality of <iostream>
without its static constructor.
This header should be used in place of <iostream> in DPC++ headers and runtime
library files.

## Tests development

Every product change should be accompanied with corresponding test modification
(adding new test(s), extending, removing or modifying existing test(s)).

There are 3 types of tests which are used for DPC++ toolchain validation:
* DPC++ device-independent tests
* DPC++ end-to-end (E2E) tests
* SYCL Conformance Test Suite (CTS)

### DPC++ device-independent tests

DPC++ device-independent tests are hosted in this repository. They can be run by
[check-llvm](https://github.com/intel/llvm/blob/sycl/llvm/test), [check-clang](https://github.com/intel/llvm/blob/sycl/clang/test),
[check-llvm-spirv](https://github.com/intel/llvm/blob/sycl/llvm-spirv/test) and [check-sycl](https://github.com/intel/llvm/blob/sycl/sycl/test) targets.
These tests are expected not to have hardware (e.g. GPU, FPGA, etc.) or
external software (e.g. OpenCL, Level Zero, CUDA runtimes) dependencies. All
other tests should land at DPC++ end-to-end or SYCL CTS tests.

Generally, any functional change to any of the DPC++ toolchain components
should be accompanied by one or more tests of this type when possible. They
allow verifying individual components and tend to be more lightweight than
end-to-end or SYCL-CTS tests.

#### General guidelines

- Use `sycl::` namespace instead of `cl::sycl::`

- Add a helpful comment describing what the test does at the beginning and
  other comments throughout the test as necessary.

- All identifiers used in `llvm/sycl` headers files must contain at
  least one lowercase letter due to avoid conflicts with user-defined macros.

- Try to follow descriptive naming convention for variables, functions as
  much as possible. Please refer to
  [LLVM naming convention](https://llvm.org/docs/CodingStandards.html#name-types-functions-variables-and-enumerators-properly)

#### DPC++ clang FE tests

- Include sycl mock headers as system headers.
  Example: `-internal-isystem %S/Inputs`

  ```C++
  `#include "sycl.hpp"`
  ```

- Use SYCL functions for invoking kernels from the mock header
  `(single_task, parallel_for, parallel_for_work_group)`
  Example:

  ```C++
  `#include "Inputs/sycl.hpp"`
  sycl::queue q;
  q.submit([&](sycl::handler &h) {
  h.single_task( { //code });
  });
  ```

#### DPC++ headers and runtime tests

- [check-sycl](https://github.com/intel/llvm/blob/sycl/sycl/test) target contains 2 types of tests: LIT tests and
  unit tests. LIT tests make compile-time checks of DPC++ headers, e.g. device
  code IR verification, `static_assert` tests. Unit tests check DPC++ runtime
  behavior and do not perform any device code compilation, instead relying on
  redefining plugin API with [PiMock](https://github.com/intel/llvm/blob/sycl/sycl/unittests/helpers/PiMock.hpp) when
  necessary.

When adding new test to `check-sycl`, please consider the following:

- if you only need to check that compilation succeeds, please use
  [`-fsyntax-only`](https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-fsyntax-only)
  compiler flag for such tests: it instructs the compiler to launch reduced
  set of commands and produce no output (no need to add `-o %t.out`).

- if you are only interested in checking device or host compilation, please use
  corresponding flags to reduce the scope of test and therefore speed it up.
  To launch only device compilation, use `-fsycl-device-only` compiler flag; to
  launch only host compilation, use `%fsycl-host-only` substitution.

- tests which want to check generated device code (either in LLVM IR or SPIR-V
  form) should be placed under [check_device_code](https://github.com/intel/llvm/blob/sycl/sycl/test/check_device_code)
  folder.

- if compiler invocation in your LIT test produces an output file, please make
  sure to redirect it into a temporary file using `-o` option and
  [`%t`](https://llvm.org/docs/CommandGuide/lit.html#substitutions)
  substitution. This is needed to avoid possible race conditions when two LIT
  tests attempt to write into the same file.

- if you need to check some runtime behavior please add unit test instead of
  LIT test. Unit tests are built with regular C++ compiler which is used to
  build the project and therefore they are not affected by `clang++` being slow
  when the project is built in Debug mode. As another side effect of using
  standard C++ compiler, device side compilation is skipped entirely, making
  them quicker to compile. And finally, unit tests are written with
  [googletest](https://google.github.io/googletest/primer.html) framework,
  which allows to use plenty of useful assertions and other helpers.

### DPC++ end-to-end (E2E) tests

These tests are located in [/sycl/test-e2e](https://github.com/intel/llvm/blob/sycl/sycl/test-e2e) directory and are not
configured to be run by default. See
[End-to-End tests documentation](https://github.com/intel/llvm/blob/sycl/sycl/test-e2e/README.md)
for instructions on how to run them.

A test which requires full stack including backend runtimes (e.g. OpenCL,
Level Zero or CUDA) should be added to DPC++ E2E tests.

### SYCL Conformance Test Suite (CTS)

These tests are hosted at
[Khronos SYCL conformance tests](https://github.com/KhronosGroup/SYCL-CTS).
These tests verify SYCL specification conformance. All implementation details
are out of scope for the tests.
See DPC++ compiler invocation definitions at
[FindIntel_SYCL](https://github.com/KhronosGroup/SYCL-CTS/blob/SYCL-1.2.1/master/cmake/FindIntel_SYCL.cmake))

## Unified Runtime Updates

To integrate changes from the [Unified Runtime][ur] project into DPC++ there
two main options which depend on the scope of those changes and the current
state of DPC++.

1. Synchronized update:
  * When: If the Unified Runtime change touches the API/ABI, more than one
    adapter, or common code such as the loader.
  * How: Update the `UNIFIED_RUNTIME_TAG` to point at the desired commit or tag
    name in the Unified Runtime repository and ensure that any tag for specific
    adapters are set to use `${UNIFIED_RUNTIME_TAG}`.

2. Decoupled update:
  * When: If only a single Unified Runtime adatper has changed.
  * How: Update the tag used in the `fetch_adapter_source()` call for a
    specific Unified Runtime adapter, e.g. Level Zero, OpenCL, CUDA, HIP, or
    Native CPU.

In general, a synchronized update should be the default. However, when there
are a lot of changes in flight in parallel always synchronizing the tag can be
troublesome. This is when a decoupled update can help sustain the merge
velocity of Unified Runtime changes.

The [intel/unified-runtime-reviewers][ur-reviewers-team] team is responsible
for ensuring that the Unified Runtime tag is updated correctly and will only
provide code owner approval to pull requests once the following criteria are
met:

* Tags are pointing to a valid commit or tag on Unified Runtime main branch.
* Changes to additional code owned files are in a good state.
* GitHub Actions checks are passing.

[ur]: https://github.com/oneapi-src/unified-runtime
[ur-reviewers-team]: https://github.com/orgs/intel/teams/unified-runtime-reviewers
