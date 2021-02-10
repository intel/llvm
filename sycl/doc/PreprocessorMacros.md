# Preprocessor Macros 

This file describes macros that have effect on SYCL compiler and run-time.

### RESTRICT_WRITE_ACCESS_TO_CONSTANT_PTR

The spec assumes that the SYCL implementation does address space deduction.
However, for our implementation, the deduction is performed in the middle end,
where it's hard to provide user friendly diagnositcs.
Due to these problems writing to raw pointers obtained from `constant_ptr` is
not diagnosed now.
The user can enable diagnostics upon writing to such pointers via enabling the
`RESTRICT_WRITE_ACCESS_TO_CONSTANT_PTR` macro.
This allows `constant_ptr` to use constant pointers as underlying
pointer types. Thus, conversions from `constant_ptr` to raw pointers will return
constant pointers and writing to const pointers will be diagnosed by the
front-end.
This behavior is not following the SYCL spec since `constant_ptr` conversions to
the underlying pointer types return pointers without any additional qualifiers
so it's disabled by default.

### SYCL2020_DISABLE_DEPRECATION_WARNINGS

Disables warning coming from usage of SYCL 1.2.1 APIs, that are deprecated in
SYCL 2020.

### SYCL_DISABLE_DEPRECATION_WARNINGS

Disables all deprecation warnings in SYCL runtime headers, including deprecation
of OpenCL interop APIs.
