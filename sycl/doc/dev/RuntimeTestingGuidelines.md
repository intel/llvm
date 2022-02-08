# SYCL Runtime and Device Headers Testing Guidelines

## SYCL Host Runtime Testing

Most changes to the SYCL Host Runtime are device-agnostic, thus can and must be
tested in isolation from the rest of the SYCL toolchain. Below is the
information on some techniques, that allow one to isolate these parts of the
runtime and simulate different behavior of low-level runtimes.

### Plugin Interface Mock

PI Mock layer provides capabilities to override plugins' behavior. There're two
ways to enable PI Mock in your unit test:

1. Include `helpers/sycl_test.hpp` header and use `SYCL_TEST` macro instead of
   `TEST` macro from Google Test framework. All other Google Test macros remain
   unchanged.
2. Include `helpers/sycl_test.hpp` header and derive from SYCLUnitTest class
   template. Then use the derived class with Google Test `TEST_F` macro. Make
   sure to call base class `SetUp()` and `TearDown()` methods in child's
   overrides.

By default, PI Mock will add one plugin for each backend and set up some
sensible overrides, enough to emulate a basic SYCL application. Each plugin will
contain one platform with three devices in each platform: a CPU, a GPU, and an
accelerator. You can use SYCL device selectors to pick a particular device.
Other plugin routines will be replaced with dummy implementation, that
terminates application immediately.

Plugin Interface routines can be overriden for all backends with `redefine` free
funcion or for a particular backend with `redefineOne` free function. Both
functions accept a function pointer or a lambda. These overrides can be used
to check, that a particular function has been called, or to emulate behavior
of a low-level runtime.

> If a pull request changes the way runtime plugin calls PI functions, it must
> also contain a unit test with PI mock to verify these changes.

> If a pull request adds new API, that may throw an exception, it must also
> contain a unit test with exception path covered. In case exception depends
> on low-level runtime call result, use PI Mock to simulate faulty call.

### Device Image Mock

SYCL Runtime discoveres user-defined kernels by looking up kernel names in
registered device images. These images contain all information about SPIR-V or
binary device code, including kernel names and some metadata passed from the
compiler frontend.

`helpers/PIImage.hpp` header provides routines to insert a fake device image
into Program Manager, and emulate presence of particular kernels.

In the default compiler flow, Clang generates a so-called integration header,
which contains information about available kernels. When compiling unit tests,
the device compiler is not used, and developers must provide specializations
for `KernelInfo` class template. See `kernel_desc.hpp` for more info.

### Other helpers

#### Mock Scheduler

For the purpose of testing SYCL Scheduler without submitting kernels to the
queue, there're some mock classes, that use Google Mock framework.
See `syc/unittests/scheduler/SchedulerTestUtils.hpp` for more info, and refer
to other scheduler tests as an example.

#### Testing environment variables

To facilitate testing environment variables and config controls, there's a
`ScopedEnvVar` helper class, which is a RAII-style wrapper around `setenv` and
`getenv`. See `helpers/ScopedEnvVar.hpp` for more info.

### Changing ABI

Whenever a new data type is added in the pull request as part of SYCL library
ABI (i.e. the data structure crosses library boundaries), such pull requests
must also provide a test, that would cover the layout of that data type.

New APIs are also covered by ABI testing. The testing tool will suggest steps
if a new API is added to SYCL runtime library.

Refer to [ABI Policy Guide](https://intel.github.io/llvm-docs/ABIPolicyGuide.html)
for more info.

## SYCL Device Headers

SYCL headers often use "magical" compiler builtins, that are later lowered to
device-specific instruction sequences. Whenever Pull Request adds or changes
uses of those builtins, it must also provide a LIT test, that would check
the correctness of compiler frontend output in the form of LLVM IR (i.e.
check calls to particular intrinsics exist, function mangling is correct, etc).

Find examples of such tests in `sycl/test/check_device_code` directory.

Refer to [FileCheck](https://llvm.org/docs/CommandGuide/FileCheck.html) and
[LIT](https://llvm.org/docs/CommandGuide/lit.html) tools documentation for
more info.

### Changing ABI

Whenever a new data structure is added as part of ABI (either between device and
host, or between two device images), a test for data layout must be added,
same as for host runtime. If the data structure can be used both on host and
device sides of user code, both tests must be present.

Refer to [ABI Policy Guide](https://intel.github.io/llvm-docs/ABIPolicyGuide.html)
for more info.

## End-to-end tests

Complex changes (especially those, that span across multiple compiler
components) may require end-to-end tests. Such tests should be simple and
concise SYCL applications, showing a good example of using the new feature.

> End-to-end tests **must not** address or expose SYCL runtime internals.

All end-to-end tests must go to
[intel/llvm-test-suite](https://github.com/intel/llvm-test-suite/tree/intel/SYCL)
repository.

Whenever an end-to-end test is required, Pull Request description must contain
a link to the corresponding PR in the llvm-test-suite repository.

