//==------ builtins.hpp - Non-standard SYCL built-in functions -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/aliases.hpp>                    // for half
#include <sycl/builtins.hpp>                   // for to_vec2
#include <sycl/builtins_utils_vec.hpp>         // for to_vec, to_marray...
#include <sycl/detail/defines_elementary.hpp>  // for __SYCL_ALWAYS_INLINE
#include <sycl/detail/generic_type_traits.hpp> // for is_svgenfloath, is_sv...
#include <sycl/detail/memcpy.hpp>              // detail::memcpy
#include <sycl/marray.hpp>                     // for marray
#include <sycl/types.hpp>                      // for vec

#include <cstring>     // for size_t
#include <stdio.h>     // for printf
#include <type_traits> // for enable_if_t

// TODO Decide whether to mark functions with this attribute.
#define __NOEXC /*noexcept*/

#ifdef __SYCL_DEVICE_ONLY__
#define __SYCL_CONSTANT_AS __attribute__((opencl_constant))
#else
#define __SYCL_CONSTANT_AS
#endif

namespace sycl {
inline namespace _V1 {
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
#if defined(__SYCL_DEVICE_ONLY__)
#if (defined(__SPIR__) || defined(__SPIRV__))
  return __spirv_ocl_printf(__format, args...);
#else
  return __builtin_printf(__format, args...);
#endif
#else
  return ::printf(__format, args...);
#endif // defined(__SYCL_DEVICE_ONLY__) && (defined(__SPIR__) ||
       // defined(__SPIRV__))
}

namespace native {

// genfloatfh tanh (genfloatfh x)
// sycl::native::tanh is only implemented on nvptx backend so far. For other
// backends we revert to the sycl::tanh impl.
template <typename T>
inline __SYCL_ALWAYS_INLINE std::enable_if_t<
    sycl::detail::is_svgenfloatf_v<T> || sycl::detail::is_svgenfloath_v<T>, T>
tanh(T x) __NOEXC {
#if defined(__NVPTX__)
  return sycl::detail::convertFromOpenCLTypeFor<T>(
      __clc_native_tanh(sycl::detail::convertToOpenCLType(x)));
#else
  return sycl::tanh(x);
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

  for (size_t i = 0; i < N / 2; i++) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    auto partial_res = native::tanh(sycl::detail::to_vec2(x, i * 2));
#else
    auto partial_res = sycl::tanh(sycl::detail::to_vec2(x, i * 2));
#endif
    sycl::detail::memcpy(&res[i * 2], &partial_res, sizeof(vec<T, 2>));
  }
  if (N % 2) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    res[N - 1] = native::tanh(x[N - 1]);
#else
    res[N - 1] = sycl::tanh(x[N - 1]);
#endif
  }

  return res;
}

// genfloath exp2 (genfloath x)
// sycl::native::exp2 (using half) is only implemented on nvptx backend so far.
// For other backends we revert to the sycl::exp2 impl.
template <typename T>
inline __SYCL_ALWAYS_INLINE
    std::enable_if_t<sycl::detail::is_svgenfloath_v<T>, T>
    exp2(T x) __NOEXC {
#if defined(__NVPTX__)
  return sycl::detail::convertFromOpenCLTypeFor<T>(
      __clc_native_exp2(sycl::detail::convertToOpenCLType(x)));
#else
  return sycl::exp2(x);
#endif
}

// sycl::native::exp2 (using half) is only implemented on nvptx backend so far.
// For other backends we revert to the sycl::exp2 impl.
template <size_t N>
inline __SYCL_ALWAYS_INLINE sycl::marray<half, N>
exp2(sycl::marray<half, N> x) __NOEXC {
  sycl::marray<half, N> res;

  for (size_t i = 0; i < N / 2; i++) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    auto partial_res = native::exp2(sycl::detail::to_vec2(x, i * 2));
#else
    auto partial_res = sycl::exp2(sycl::detail::to_vec2(x, i * 2));
#endif
    sycl::detail::memcpy(&res[i * 2], &partial_res, sizeof(vec<half, 2>));
  }
  if (N % 2) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    res[N - 1] = native::exp2(x[N - 1]);
#else
    res[N - 1] = sycl::exp2(x[N - 1]);
#endif
  }
  return res;
}

} // namespace native

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl

#undef __SYCL_CONSTANT_AS
