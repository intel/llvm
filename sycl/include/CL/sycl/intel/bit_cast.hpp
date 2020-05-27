//==----------- bit_cast.hpp --- SYCL bit_cast -----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/defines.hpp>
#include <CL/sycl/detail/helpers.hpp>

#include <numeric>
#include <type_traits>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

template <typename To, typename From>
constexpr To bit_cast(const From &from) noexcept {
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
  To to;
  sycl::detail::memcpy(&to, &from, sizeof(To));
  return to;
#endif // __has_builtin(__builtin_bit_cast)

#endif // __cpp_lib_bit_cast
}

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)