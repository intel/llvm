//==------ builtins.hpp - Non-standard SYCL built-in functions -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/builtins.hpp>
#include <sycl/detail/builtins.hpp>
#include <sycl/detail/generic_type_lists.hpp>
#include <sycl/detail/generic_type_traits.hpp>
#include <sycl/detail/type_traits.hpp>

#include <CL/__spirv/spirv_ops.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>

// TODO Decide whether to mark functions with this attribute.
#define __NOEXC /*noexcept*/

#ifdef __SYCL_DEVICE_ONLY__
#define __SYCL_CONSTANT_AS __attribute__((opencl_constant))
#else
#define __SYCL_CONSTANT_AS
#endif

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext::oneapi::experimental {

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
// 64-bit argument to the sycl::experimental::printf
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
// sycl::native::tanh is only implemented on nvptx backend so far. For other
// backends we revert to the sycl::tanh impl.
template <typename T>
inline __SYCL_ALWAYS_INLINE
    std::enable_if_t<sycl::detail::is_svgenfloatf<T>::value ||
                         sycl::detail::is_svgenfloath<T>::value,
                     T>
    tanh(T x) __NOEXC {
#if defined(__NVPTX__)
  using _ocl_T = sycl::detail::ConvertToOpenCLType_t<T>;
  _ocl_T arg1 = sycl::detail::convertDataToType<T, _ocl_T>(x);
  return sycl::detail::convertDataToType<_ocl_T, T>(__clc_native_tanh(arg1));
#else
  return __sycl_std::__invoke_tanh<T>(x);
#endif
}

// The marray math function implementations use vectorizations of
// size two as a simple general optimization. A more complex implementation
// using larger vectorizations for large marray sizes is possible; however more
// testing is required in order to ascertain the performance implications for
// all backends.
// sycl::native::tanh is only implemented on nvptx backend so far. For other
// backends we revert to the sycl::tanh impl.
template <typename T, size_t N>
inline __SYCL_ALWAYS_INLINE
    std::enable_if_t<std::is_same_v<T, half> || std::is_same_v<T, float>,
                     sycl::marray<T, N>>
    tanh(sycl::marray<T, N> x) __NOEXC {
  sycl::marray<T, N> res;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
#define FUNC_VEC native::tanh
#define FUNC FUNC_VEC
#else
#define FUNC_VEC __sycl_std::__invoke_tanh<sycl::vec<T, 2>>
#define FUNC __sycl_std::__invoke_tanh<T>
#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)

  for (size_t i = 0; i < N / 2; i++) {
    auto partial_res = FUNC_VEC(sycl::detail::to_vec2(x, i * 2));
    std::memcpy(&res[i * 2], &partial_res, sizeof(vec<T, 2>));
  }
  if (N % 2) {
    res[N - 1] = FUNC(x[N - 1]);
  }
#undef FUNC_VEC
#undef FUNC
  return res;
}

// genfloath exp2 (genfloath x)
// sycl::native::exp2 (using half) is only implemented on nvptx backend so far.
// For other backends we revert to the sycl::exp2 impl.
template <typename T>
inline __SYCL_ALWAYS_INLINE
    std::enable_if_t<sycl::detail::is_svgenfloath<T>::value, T>
    exp2(T x) __NOEXC {
#if defined(__NVPTX__)
  using _ocl_T = sycl::detail::ConvertToOpenCLType_t<T>;
  _ocl_T arg1 = sycl::detail::convertDataToType<T, _ocl_T>(x);
  return sycl::detail::convertDataToType<_ocl_T, T>(__clc_native_exp2(arg1));
#else
  return __sycl_std::__invoke_exp2<T>(x);
#endif
}

// sycl::native::exp2 (using half) is only implemented on nvptx backend so far.
// For other backends we revert to the sycl::exp2 impl.
template <size_t N>
inline __SYCL_ALWAYS_INLINE sycl::marray<half, N>
exp2(sycl::marray<half, N> x) __NOEXC {
  sycl::marray<half, N> res;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
#define FUNC_VEC native::exp2
#define FUNC FUNC_VEC
#else
#define FUNC_VEC __sycl_std::__invoke_exp2<sycl::vec<half, 2>>
#define FUNC __sycl_std::__invoke_exp2<half>
#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)

  for (size_t i = 0; i < N / 2; i++) {
    auto partial_res = FUNC_VEC(sycl::detail::to_vec2(x, i * 2));
    std::memcpy(&res[i * 2], &partial_res, sizeof(vec<half, 2>));
  }
  if (N % 2) {
    res[N - 1] = FUNC(x[N - 1]);
  }
#undef FUNC_VEC
#undef FUNC
  return res;
}

} // namespace native

} // namespace ext::oneapi::experimental
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

#undef __SYCL_CONSTANT_AS
