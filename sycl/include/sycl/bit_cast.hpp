//==---------------- bit_cast.hpp - SYCL bit_cast --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <type_traits> // for is_trivially_copyable, enable_if_t

// std::bit_cast is first choice, __builtin_bit_cast second.
// memcpy fallback of last resort, not constexpr :-(

// MSVC 2019 Update 9 or later (aka Visual Studio 2019 v 16.8.1)
#if defined(_MSC_VER) && _MSC_VER >= 1928
#define __SYCL_HAS_BUILTIN_BIT_CAST 1
#elif defined(__has_builtin)
#define __SYCL_HAS_BUILTIN_BIT_CAST __has_builtin(__builtin_bit_cast)
#else
#define __SYCL_HAS_BUILTIN_BIT_CAST 0
#endif

#if __cplusplus >= 202002L
#include <version> // defines __cpp_lib_bit_cast
#endif

#if __cpp_lib_bit_cast
// first choice std::bit_cast
#include <bit>
#define __SYCL_BITCAST_IS_CONSTEXPR 1
#elif __SYCL_HAS_BUILTIN_BIT_CAST
// second choice __builtin_bit_cast
#define __SYCL_BITCAST_IS_CONSTEXPR 1
#else
// fallback memcpy
#include <sycl/detail/memcpy.hpp>
#endif

namespace sycl {
inline namespace _V1 {

template <typename To, typename From>
#if defined(__SYCL_BITCAST_IS_CONSTEXPR)
constexpr
#endif
    std::enable_if_t<sizeof(To) == sizeof(From) &&
                         std::is_trivially_copyable<From>::value &&
                         std::is_trivially_copyable<To>::value,
                     To>
    bit_cast(const From &from) noexcept {
#if __cpp_lib_bit_cast
  // first choice std::bit_cast
  return std::bit_cast<To>(from);
#elif __SYCL_HAS_BUILTIN_BIT_CAST
  // second choice __builtin_bit_cast
  return __builtin_bit_cast(To, from);
#else
  // fallback memcpy
  static_assert(std::is_trivially_default_constructible<To>::value,
                "To must be trivially default constructible");
  To to;
  sycl::detail::memcpy(&to, &from, sizeof(To));
  return to;
#endif
}

} // namespace _V1
} // namespace sycl
