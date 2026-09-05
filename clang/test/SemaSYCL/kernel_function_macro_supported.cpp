// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-device -fsyntax-only %s

// Under the SYCL device compiler the __builtin_sycl_launch_kernel builtin is
// available, so the SYCL_EXT_ONEAPI_KERNEL_FUNCTION macro is enabled and its
// capability macro SYCL_EXT_ONEAPI_KERNEL_FUNCTION_SUPPORTED is defined. This
// mirrors the __has_builtin gate the extension header uses so applications can
// detect the feature portably.

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

#if !__has_builtin(__builtin_sycl_launch_kernel)
#error "__builtin_sycl_launch_kernel should be available under -fsycl"
#endif

#if __has_builtin(__builtin_sycl_launch_kernel)
#define SYCL_EXT_ONEAPI_KERNEL_FUNCTION_SUPPORTED 1
#endif

#ifndef SYCL_EXT_ONEAPI_KERNEL_FUNCTION_SUPPORTED
#error "SYCL_EXT_ONEAPI_KERNEL_FUNCTION_SUPPORTED must be defined when the "    \
       "builtin is available"
#endif

// expected-no-diagnostics
