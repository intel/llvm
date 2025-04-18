//==----- work_group_static.hpp --- SYCL group local memory extension -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <sycl/detail/defines_elementary.hpp> // for __SYCL_ALWAYS_INLINE
#include <sycl/exception.hpp>                 // for exception

#include <type_traits> // for enable_if_t, is_trivially_destructible_v ...

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi {
namespace experimental {

#ifdef __SYCL_DEVICE_ONLY__
#define __SYCL_WG_SCOPE [[__sycl_detail__::wg_scope]]
#else
#define __SYCL_WG_SCOPE
#endif

/// @brief Allocate data in device local memory.
/// Any work_group_static object will be place in device local memory and hold
/// an object of type T. work_group_static object are implicitly treated as
/// static.
/// @tparam T must be a trivially constructible and destructible type
template <typename T> class __SYCL_WG_SCOPE work_group_static final {
public:
  static_assert(
      std::is_trivially_destructible_v<T> &&
          std::is_trivially_constructible_v<T>,
      "Can only be used with trivially constructible and destructible types");
  static_assert(!std::is_const_v<T> && !std::is_volatile_v<T>,
                "Can only be used with non const and non volatile types");
  __SYCL_ALWAYS_INLINE work_group_static() = default;
  work_group_static(const work_group_static &) = delete;
  work_group_static &operator=(const work_group_static &) = delete;

  operator T &() noexcept { return data; }

  template <class TArg = T, typename = std::enable_if_t<!std::is_array_v<TArg>>>
  work_group_static &operator=(const T &value) noexcept {
    data = value;
    return *this;
  }

  T *operator&() noexcept { return &data; }

private:
  T data;
};

#undef __SYCL_WG_SCOPE

} // namespace experimental
} // namespace ext::oneapi
} // namespace _V1
} // namespace sycl
