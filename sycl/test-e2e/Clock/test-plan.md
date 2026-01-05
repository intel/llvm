:sectnums:
:xrefstyle: short

# Test plan for sycl_ext_oneapi_clock

This is a test plan for the APIs described in
https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/proposed/sycl_ext_oneapi_clock.asciidoc[sycl_ext_oneapi_clock].

## Testing scope

### Device coverage

All of the tests described below are performed only on the default device that
is selected on the CTS command line.

## Tests

### New aspects

The extension introduces the following aspects:

* `ext_oneapi_clock_sub_group`,
* `ext_oneapi_clock_work_group`,
* `ext_oneapi_clock_device`

Check if these aspects are defined by the implementation.

### New function

The extension introduces the following function:

* `uint64_t clock<clock_scope scope>()`

Check if the function:

* can be successfully called from a kernel,
* accepts all possible values of `clock_scope` enum,
* returns different values when called multiple times and these values are
increasing (keep in mind an overflow).

When possible, check if the implementation throws an exception with the
`errc::kernel_not_supported` error code when the kernel is submitted to a queue,
but a device doesn't have a corresponding aspect.
