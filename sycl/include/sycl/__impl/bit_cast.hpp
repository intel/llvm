//==---------------- bit_cast.hpp - SYCL bit_cast --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <type_traits>

#if __cpp_lib_bit_cast
#include <bit>
#endif

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

// forward decl
namespace detail {
inline void memcpy(void *Dst, const void *Src, std::size_t Size);
}

// sycl::bit_cast ( no longer sycl::detail::bit_cast )
template <typename To, typename From>
#if __cpp_lib_bit_cast || __has_builtin(__builtin_bit_cast)
constexpr
#endif
    To
    bit_cast(const From &from) noexcept {
  static_assert(sizeof(To) == sizeof(From),
                "Sizes of To and From must be equal");
  static_assert(std::is_trivially_copyable<From>::value,
                "From must be trivially copyable");
  static_assert(std::is_trivially_copyable<To>::value,
                "To must be trivially copyable");
#if __cpp_lib_bit_cast
  return std::bit_cast<To>(from);
#else // __cpp_lib_bit_cast

#if __has_builtin(__builtin_bit_cast)
  return __builtin_bit_cast(To, from);
#else  // __has_builtin(__builtin_bit_cast)
  static_assert(std::is_trivially_default_constructible<To>::value,
                "To must be trivially default constructible");
  To to;
  sycl::detail::memcpy(&to, &from, sizeof(To));
  return to;
#endif // __has_builtin(__builtin_bit_cast)

#endif // __cpp_lib_bit_cast
}

namespace detail {
template <typename To, typename From>
#if __cpp_lib_bit_cast || __has_builtin(__builtin_bit_cast)
constexpr
#endif
    To
    bit_cast(const From &from) noexcept {
  return sycl::bit_cast<To>(from);
}
} // namespace detail

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
