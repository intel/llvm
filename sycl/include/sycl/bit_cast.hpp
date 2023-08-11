//==---------------- bit_cast.hpp - SYCL bit_cast --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <type_traits> // for is_trivially_copyable, enable_if_t

#if __cplusplus >= 202002L
#include <version> // defines __cpp_lib_bit_cast
#endif

#if __cpp_lib_bit_cast
#include <bit>
#elif !defined(__has_builtin) || !__has_builtin(__builtin_bit_cast)
#include <sycl/detail/memcpy.hpp>
#endif

namespace sycl {
inline namespace _V1 {

template <typename To, typename From>
#if __cpp_lib_bit_cast ||                                                      \
    (defined(__has_builtin) && __has_builtin(__builtin_bit_cast))
constexpr
#endif
    std::enable_if_t<sizeof(To) == sizeof(From) &&
                         std::is_trivially_copyable<From>::value &&
                         std::is_trivially_copyable<To>::value,
                     To>
    bit_cast(const From &from) noexcept {
#if __cpp_lib_bit_cast
  return std::bit_cast<To>(from);
#else // __cpp_lib_bit_cast

#if defined(__has_builtin) && __has_builtin(__builtin_bit_cast)
  return __builtin_bit_cast(To, from);
#else // __has_builtin(__builtin_bit_cast)
  static_assert(std::is_trivially_default_constructible<To>::value,
                "To must be trivially default constructible");
  To to;
  sycl::detail::memcpy(&to, &from, sizeof(To));
  return to;
#endif // __has_builtin(__builtin_bit_cast)

#endif // __cpp_lib_bit_cast
}
} // namespace _V1
} // namespace sycl
