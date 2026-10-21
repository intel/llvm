// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify -DHAS_BUILTIN %s
// RUN: %clang_cc1 -fsycl-is-host -fsyntax-only -verify -DHAS_BUILTIN %s
// RUN: %clang_cc1 -fsyntax-only -verify %s

// The SYCL headers enable SYCL_EXT_ONEAPI_KERNEL_FUNCTION by testing
// __has_builtin(__builtin_sycl_launch_kernel). The builtin must be available on
// both the SYCL device and host sides, and only when compiling SYCL. The header
// logic itself is tested by
// sycl/test/extensions/free_function_kernels/kernel_function_macro.cpp.

// expected-no-diagnostics

#if defined(HAS_BUILTIN) != __has_builtin(__builtin_sycl_launch_kernel)
#error "__builtin_sycl_launch_kernel availability is wrong"
#endif
