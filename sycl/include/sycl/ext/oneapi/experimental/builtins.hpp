//==------ builtins.hpp - Non-standard SYCL built-in functions -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/builtins.hpp>
#include <CL/sycl/detail/builtins.hpp>
#include <CL/sycl/detail/generic_type_lists.hpp>
#include <CL/sycl/detail/generic_type_traits.hpp>
#include <CL/sycl/detail/type_traits.hpp>

#include <CL/__spirv/spirv_ops.hpp>
#include <sycl/ext/oneapi/experimental/bfloat16.hpp>

// TODO Decide whether to mark functions with this attribute.
#define __NOEXC /*noexcept*/

#ifdef __SYCL_DEVICE_ONLY__
#define __SYCL_CONSTANT_AS __attribute__((opencl_constant))
#else
#define __SYCL_CONSTANT_AS
#endif

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace oneapi {
namespace experimental {

// Provides functionality to print data from kernels in a C way:
// - On non-host devices this function is directly mapped to printf from
//   OpenCL C
// - On host device, this function should be equivalent to standard printf
//   function from C/C++.
//
// Please refer to corresponding section in OpenCL C specification to find
// information about format string and its differences from standard C rules.
//
// This function is placed under 'experimental' namespace on purpose, because it
// has too much caveats you need to be aware of before using it. Please find
// them below and read carefully before using it:
//
// - According to the OpenCL spec, the format string must be
// resolvable at compile time i.e. cannot be dynamically created by the
// executing program.
//
// - According to the OpenCL spec, the format string must reside in constant
// address space. The constant address space declarations might get "tricky",
// see test/built-ins/printf.cpp for examples.
// In simple cases (compile-time known string contents, direct declaration of
// the format literal inside the printf call, etc.), the compiler should handle
// the automatic address space conversion.
// FIXME: Once the extension to generic address space is fully supported, the
// constant AS version may need to be deprecated.
//
// - The format string is interpreted according to the OpenCL C spec, where all
// data types has fixed size, opposed to C++ types which doesn't guarantee
// the exact width of particular data types (except, may be, char). This might
// lead to unexpected result, for example: %ld in OpenCL C means that printed
// argument has 'long' type which is 64-bit wide by the OpenCL C spec. However,
// by C++ spec long is just at least 32-bit wide, so, you need to ensure (by
// performing a cast, for example) that if you use %ld specifier, you pass
// 64-bit argument to the cl::sycl::experimental::printf
//
// - OpenCL spec defines several additional features, like, for example, 'v'
// modifier which allows to print OpenCL vectors: note that these features are
// not available on host device and therefore their usage should be either
// guarded using __SYCL_DEVICE_ONLY__ preprocessor macro or avoided in favor
// of more portable solutions if needed
//
template <typename FormatT, typename... Args>
int printf(const FormatT *__format, Args... args) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
  return __spirv_ocl_printf(__format, args...);
#else
  return ::printf(__format, args...);
#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
}

namespace native {

// genfloatfh tanh (genfloatfh x)
template <typename T>
inline __SYCL_ALWAYS_INLINE
    sycl::detail::enable_if_t<sycl::detail::is_genfloatf<T>::value ||
                                  sycl::detail::is_genfloath<T>::value,
                              T>
    tanh(T x) __NOEXC {
#if defined(__NVPTX__)
  using _ocl_T = cl::sycl::detail::ConvertToOpenCLType_t<T>;
  _ocl_T arg1 = cl::sycl::detail::convertDataToType<T, _ocl_T>(x);
  return cl::sycl::detail::convertDataToType<_ocl_T, T>(
      __clc_native_tanh(arg1));
#else
  return __sycl_std::__invoke_tanh<T>(x);
#endif
}

// genfloath exp2 (genfloath x)
template <typename T>
inline __SYCL_ALWAYS_INLINE
    sycl::detail::enable_if_t<sycl::detail::is_genfloath<T>::value, T>
    exp2(T x) __NOEXC {
#if defined(__NVPTX__)
  using _ocl_T = cl::sycl::detail::ConvertToOpenCLType_t<T>;
  _ocl_T arg1 = cl::sycl::detail::convertDataToType<T, _ocl_T>(x);
  return cl::sycl::detail::convertDataToType<_ocl_T, T>(
      __clc_native_exp2(arg1));
#else
  return __sycl_std::__invoke_exp2<T>(x);
#endif
}

} // namespace native

namespace detail {

template <typename T> struct is_bf16_storage_type {
  static constexpr int value = false;
};

template <> struct is_bf16_storage_type<uint16_t> {
  static constexpr int value = true;
};

template <> struct is_bf16_storage_type<uint32_t> {
  static constexpr int value = true;
};

template <int N> struct is_bf16_storage_type<vec<uint16_t, N>> {
  static constexpr int value = true;
};

template <int N> struct is_bf16_storage_type<vec<uint32_t, N>> {
  static constexpr int value = true;
};

} // namespace detail

template <typename T>
std::enable_if_t<experimental::detail::is_bf16_storage_type<T>::value, T>
fabs(T x) {
#ifdef __SYCL_DEVICE_ONLY__
  return __clc_fabs(x);
#else
  (void)x;
  throw runtime_error("bf16 is not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename T>
std::enable_if_t<
    std::is_same<T, cl::sycl::ext::oneapi::experimental::bfloat16>::value, T>
fabs(T x) {
#ifdef __SYCL_DEVICE_ONLY__
  return cl::sycl::ext::oneapi::experimental::bfloat16::from_bits(
      __clc_fabs(x.raw()));
#else
  (void)x;
  throw runtime_error("bf16 is not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename T>
std::enable_if_t<experimental::detail::is_bf16_storage_type<T>::value, T>
fmin(T x, T y) {
#ifdef __SYCL_DEVICE_ONLY__
  return __clc_fmin(x, y);
#else
  (void)x;
  (void)y;
  throw runtime_error("bf16 is not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename T>
std::enable_if_t<
    std::is_same<T, cl::sycl::ext::oneapi::experimental::bfloat16>::value, T>
fmin(T x, T y) {
#ifdef __SYCL_DEVICE_ONLY__
  return cl::sycl::ext::oneapi::experimental::bfloat16::from_bits(
      __clc_fmin(x.raw(), y.raw()));
#else
  (void)x;
  (void)y;
  throw runtime_error("bf16 is not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename T>
std::enable_if_t<experimental::detail::is_bf16_storage_type<T>::value, T>
fmax(T x, T y) {
#ifdef __SYCL_DEVICE_ONLY__
  return __clc_fmax(x, y);
#else
  (void)x;
  (void)y;
  throw runtime_error("bf16 is not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename T>
std::enable_if_t<
    std::is_same<T, cl::sycl::ext::oneapi::experimental::bfloat16>::value, T>
fmax(T x, T y) {
#ifdef __SYCL_DEVICE_ONLY__
  return cl::sycl::ext::oneapi::experimental::bfloat16::from_bits(
      __clc_fmax(x.raw(), y.raw()));
#else
  (void)x;
  (void)y;
  throw runtime_error("bf16 is not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename T>
std::enable_if_t<experimental::detail::is_bf16_storage_type<T>::value, T>
fma(T x, T y, T z) {
#ifdef __SYCL_DEVICE_ONLY__
  return __clc_fma(x, y, z);
#else
  (void)x;
  (void)y;
  (void)z;
  throw runtime_error("bf16 is not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename T>
std::enable_if_t<
    std::is_same<T, cl::sycl::ext::oneapi::experimental::bfloat16>::value, T>
fma(T x, T y, T z) {
#ifdef __SYCL_DEVICE_ONLY__
  return cl::sycl::ext::oneapi::experimental::bfloat16::from_bits(
      __clc_fma(x.raw(), y.raw(), z.raw()));
#else
  (void)x;
  (void)y;
  (void)z;
  throw runtime_error("bf16 is not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

} // namespace experimental
} // namespace oneapi
} // namespace ext

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

#undef __SYCL_CONSTANT_AS
