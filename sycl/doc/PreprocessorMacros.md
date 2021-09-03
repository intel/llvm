# Preprocessor Macros 

This file describes macros that have effect on SYCL compiler and run-time.

- **RESTRICT_WRITE_ACCESS_TO_CONSTANT_PTR**

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

- **DISABLE_SYCL_INSTRUMENTATION_METADATA**

  This macro is used to disable passing of code location information to public
  methods.

- **SYCL2020_DISABLE_DEPRECATION_WARNINGS**

  Disables warnings coming from usage of SYCL 1.2.1 APIs, that are deprecated in
  SYCL 2020.

- **SYCL_DISABLE_DEPRECATION_WARNINGS**

  Disables all deprecation warnings in SYCL runtime headers, including SYCL 1.2.1 deprecations.

- **SYCL_DISABLE_FALLBACK_ASSERT**

  Defining this macro eliminates some overhead that is associated with
  submitting kernels that call `assert()`. When this macro is defined, the logic
  for detecting assertion failures in kernels is disabled, so a failed assert
  will not cause a message to be printed and will not cause the program to
  abort. However, this macro only affects kernels that are submitted to devices
  that do **not** have native support for `assert()` because devices with native
  support do not impose any extra overhead. One can check to see if a device has
  native support for `assert()` via `aspect::ext_oneapi_native_assert`.

## Version macros

- `__LIBSYCL_MAJOR_VERSION` is set to SYCL runtime library major version.
- `__LIBSYCL_MINOR_VERSION` is set to SYCL runtime library minor version.
- `__LIBSYCL_PATCH_VERSION` is set to SYCL runtime library patch version.
