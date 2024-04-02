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

- **SYCL2020_DISABLE_DEPRECATION_WARNINGS**

  Disables warnings coming from usage of SYCL 1.2.1 APIs, that are deprecated in
  SYCL 2020.

- **SYCL_DISABLE_DEPRECATION_WARNINGS**

  Disables all deprecation warnings in SYCL runtime headers, including SYCL
  1.2.1 deprecations.

- **SYCL_DISABLE_IMAGE_ASPECT_WARNING**

  Disables warning diagnostic issued when calling `device::has(aspect::image)`
  and `platform::has(aspect::image)`.

- **SYCL_FALLBACK_ASSERT**

  Defining as non-zero enables the fallback assert feature even on devices
  without native support. Be aware that this will add some overhead that is
  associated with submitting kernels that call `assert()`. When this macro is
  defined as 0 or is not defined, the logic for detecting assertion failures in kernels is
  disabled, so a failed assert will not cause a message to be printed and will
  not cause the program to abort. Some devices have native support for
  assertions. The logic for detecting assertion failures is always enabled on
  these devices regardless of whether this macro is defined because that logic
  does not add any extra overhead. One can check to see if a device has native
  support for `assert()` via `aspect::ext_oneapi_native_assert`.
  This macro is undefined by default.

- **SYCL2020_CONFORMANT_APIS (deprecated)**
  This macro is used to comply with the SYCL 2020 specification, as some of the current 
  implementations may be widespread and not conform to it.
  Defining this macro currently has no effect on the API.
  This preprocessor macro has been deprecated in favor of the
  `-fpreview-breaking-changes` compiler option.

## Version macros

- `__LIBSYCL_MAJOR_VERSION` is set to SYCL runtime library major version.
- `__LIBSYCL_MINOR_VERSION` is set to SYCL runtime library minor version.
- `__LIBSYCL_PATCH_VERSION` is set to SYCL runtime library patch version.
