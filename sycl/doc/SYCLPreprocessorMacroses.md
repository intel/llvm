# Overview

This file describes macroses that are having effect on SYCL compiler and run-time.

### RESTRICT_WRITE_ACCESS_TO_CONSTANT_PTR

Spec assumes that SYCL implementation does address space deduction, but we made
an implementation doing address space deduction in the middle end, where it's
hard to provide user friendly diagnositcs.
Due to these problems writing to raw pointers obtained from `constant_ptr` is not
diagnosed now.
User can enable diagnostics of writing to such pointers via enabling of
`RESTRICT_WRITE_ACCESS_TO_CONSTANT_PTR` preprocessor macro.
Enabling this macro allows `constant_ptr` to use constant pointers as underlying
pointer types. So, conversions from `constant_ptr` to raw pointers will return
constant pointers and writing to const pointers will be diagnosed by the
front-end.
This behavior is not following SYCL spec since `constant_ptr` conversions to the
underlying pointer types return pointers without any additional qualifiers so
it's disabled by default.
