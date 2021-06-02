//==------------- sycl_util.hpp - DPC++ Explicit SIMD API  -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Utility functions related to interaction with generic SYCL and used for
// implementing Explicit SIMD APIs.
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/accessor.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace intel {
namespace experimental {
namespace esimd {
namespace detail {

// Checks that given type is a SYCL accessor type. Sets its static field
// \c value accordingly. Also, if the check is succesful, sets \c mode and
// \c target static fields to the accessor type's access mode and access target
// respectively. Otherwise they are set to -1.
template <typename T> struct is_sycl_accessor : public std::false_type {
  static constexpr sycl::access::mode mode =
      static_cast<sycl::access::mode>(-1);
  static constexpr sycl::access::target target =
      static_cast<sycl::access::target>(-1);
};

template <typename DataT, int Dimensions, sycl::access::mode AccessMode,
          sycl::access::target AccessTarget,
          sycl::access::placeholder IsPlaceholder, typename PropertyListT>
struct is_sycl_accessor<sycl::accessor<
    DataT, Dimensions, AccessMode, AccessTarget, IsPlaceholder, PropertyListT>>
    : public std::true_type {
  static constexpr sycl::access::mode mode = AccessMode;
  static constexpr sycl::access::target target = AccessTarget;
};

using accessor_mode_cap_val_t = bool;

// Denotes an accessor's capability - whether it can read or write.
struct accessor_mode_cap {
  static inline constexpr accessor_mode_cap_val_t can_read = false;
  static inline constexpr accessor_mode_cap_val_t can_write = true;
};

template <sycl::access::mode Mode, accessor_mode_cap_val_t Cap>
constexpr bool accessor_mode_has_capability() {
  static_assert(Cap == accessor_mode_cap::can_read ||
                    Cap == accessor_mode_cap::can_write,
                "unsupported capability");

  if constexpr (Mode == sycl::access::mode::atomic ||
                Mode == sycl::access::mode::read_write ||
                Mode == sycl::access::mode::discard_read_write)
    return true; // atomic and *read_write accessors can read/write

  return (Cap == accessor_mode_cap::can_read) ==
         (Mode == sycl::access::mode::read);
}

// Checks that given type is a SYCL accessor type with given capability and
// target.
template <typename T, accessor_mode_cap_val_t Capability,
          sycl::access::target AccessTarget>
struct is_sycl_accessor_with
    : public std::conditional_t<
          accessor_mode_has_capability<is_sycl_accessor<T>::mode,
                                       Capability>() &&
              (is_sycl_accessor<T>::target == AccessTarget),
          std::true_type, std::false_type> {};

template <typename T, accessor_mode_cap_val_t Capability,
          sycl::access::target AccessTarget, typename RetT>
using EnableIfAccessor = sycl::detail::enable_if_t<
    detail::is_sycl_accessor_with<T, Capability, AccessTarget>::value, RetT>;

} // namespace detail
} // namespace esimd
} // namespace experimental
} // namespace intel
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
