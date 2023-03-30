# Test plan for dynamic linking of device code

This is a test plan for device code dynamic linking feature described in
https://github.com/intel/llvm/blob/sycl/sycl/doc/SharedLibraries.md document.

## 1. Testing scope

### 1.1. Device & BE coverage

All of the tests described below are performed on Intel CPU and GPU devices.
For GPU device both OpenCL and L0 backends are used.

**NOTE**: The feature isn't supported on FPGA device.

### 1.2. AOT & JIT compilation

All the tests are compiled using AOT, JIT and mixed (AOT with JIT) compilation.
Mixed compilation is the case when application is built using AOT and used
shared library uses JIT or vice versa.

### 1.3. Device code split modes

All the tests are compiled using different combination of device code split
mode.
There are four possible device code split modes that can be requested via
`fsycl-device-code-split` compiler's option:
- `per_kernel` - device code module is created for each SYCL kernel
- `per_source` - device code module is created for each source
- `off` - no device code split
- `auto` - use heuristic to select the best way of splitting device code

For each test the main application and all shared libraries are compiled with
all the variations of device code split modes. This results in four binaries for
each application or library.
Each tests covers all the combinations of device code split modes used for
the application and libraries, i.e. all binaries of the application are checked
with all binaries for shared libraries.
For one application and one shared library examples of the combinations will be:
- Application compiled with `per_kernel` and library compiled with `per_kernel`
- Application compiled with `per_kernel` and library compiled with `per_source`
- Application compiled with `per_kernel` and library compiled with `off`
- Application compiled with `per_kernel` and library compiled with `auto`

- Application compiled with `per_source` and library compiled with `per_kernel`
- Application compiled with `per_source` and library compiled with `per_source`
- Application compiled with `per_source` and library compiled with `off`
- Application compiled with `per_source` and library compiled with `auto`

and etc. Total 16 combinations.

## 2. Tests

All the tests consist of at least one main application and at least one shared
library.
For all the tests simple calculation is performed by the application's kernel(s)
that used `SYCL_EXTERNAL` from the shared library. Example : addition of
vectors.
The result of the calculation must be checked on host.

### 2.1. Basic test with shared library

- Several kernels and `SYCL_EXTERNAL` functions are defined in both library and
  application.
- At least one `SYCL_EXTERNAL` function defined in shared library is used from
  the application's kernel.
- At least one `SYCL_EXTERNAL` function defined in shared library is used in
  kernel inside the library.
- Defined kernels are submitted and ran.

### 2.2. Test with dependent shared libraries

The test consists of the main application and **two** shared libraries libA and
libB and performs the following:
- Application defines kernel(s) with dependency on libA.
- libA defines `SYCL_EXTERNAL` function(s) with dependency on libB.
- libB defines `SYCL_EXTERNAL` function(s) without external dependencies.
- Defined kernels are submitted and ran.

### 2.3. Tests with function pointers

#### 2.3.1. Variation 1

- Application defines a function with `indirectly_callable` attribute.
- Shared library defines host API that can receive function pointer as an
  argument and runs a kernel that calls through passed function pointer.
- Address of `indirectly_callable` function is taken and passed to shared
  library's host API.

#### 2.3.2. Variation 2

- Shared library defines a function with `indirectly_callable` attribute.
- Shared library defines host API that can return a function pointer to
  a function with `indirectly_callable` attribute.
- Application used host API of shared library and calls through received
  function pointer from a kernel.

### 2.4. Test with shared library without device code

The test aimed to check that support for device code inside shared library
doesn't break support for host-only shared library.

- Shared library defines host-only non-`SYCL_EXTERNAL` function.
- The function from shared library is used by host part of the application.
- Application defines and runs a kernel with a simple calculation.

### 2.5. Tests with run-time loaded shared library

Tests aimed to check that support for dynamic linking of device code doesn't
break run-time loading of shared library.

#### 2.5.1. Variation 1

The test makes sure that run-time loading of host-only shared libraries is
not broken.

- Shared library defines several host-only external functions
- Shared library doesn't contain any device code
- There is no explicit dependencies between application and library
- Shared library is loaded via dlopen/LoadLibrary functionality
- Functions from the shared library are used by host part of the application
  to perform a simple calculation
- Result of the simple calculation is checked

#### 2.5.2. Variation 2

The test makes sure that run-time loading of shared libraries with embedded
device code is not broken.

- Shared library defines several host-only external functions, several
  `SYCL_EXTERNAL` functions and several kernels
- There is no explicit dependencies between application and library on
  device and host
- Shared library is loaded via dlopen/LoadLibrary functionality
- Host functions from the shared library are used by host part of the
  application to perform a simple calculation
- Result of the simple calculation is checked

### 2.6. Test with two versions of shared library

The test is compiled with version 1 of the shared library and ran with
shared library of version 2.
Code can be distributed in the same way as in basic test, but `SYCL_EXTERNAL`
functions defined in shared library of version 2, should have a different
definition comparing to version 1, so it is possible to check that correct
version of the shared library were used.

### 2.7. Test with kernel bundle

The tests aim to check that operations with kernel bundles work correctly in
co-existence with dynamic linking feature.

#### 2.7.1. Variation 1

- Shared library defines at least one `SYCL_EXTERNAL` function
- Application defines a kernel with dependency on `SYCL_EXTERNAL` function(s)
  from the shared library
- Application gets a `kernel_bundle` object that contains kernel with
  dependency (can be done via getting `kernel_id` object and using it as a
  parameter for `get_kernel_bundle` free function overload).
- Application builds the `kernel_bundle` object and uses it to run the kernel
  with dependency

#### 2.7.2. Variation 2

- Shared library defines at least one `SYCL_EXTERNAL` function
- Shared library defines several kernels, at least one kernel from shared
  library uses `SYCL_EXTERNAL` function
- Application defines a kernel with dependency on `SYCL_EXTERNAL` function(s)
  from the shared library
- Shared library provides a host function that returns `kernel_bundle` object
  containing kernels from the shared library
- Application gets a `kernel_bundle` object that contains kernel with
  dependency and links it using link API with `kernel_bundle` that contains
  kernels from shared library
- Resulting `kernel_bundle` object is used to run the kernel with dependency
  on shared library

### 2.8. Test with several applications

Same as [Basic test with shared library](#Basic-test-with-shared-library) but
two or more applications using same shared library are ran at the same time.
