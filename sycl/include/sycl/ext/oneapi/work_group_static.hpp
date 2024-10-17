//==----- work_group_static.hpp --- SYCL group local memory extension -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <sycl/access/access.hpp>             // for address_space, decorated
#include <sycl/detail/defines_elementary.hpp> // for __SYCL_ALWAYS_INLINE
#include <sycl/exception.hpp>                 // for exception
#include <sycl/ext/intel/usm_pointers.hpp>    // for multi_ptr
#include <sycl/ext/oneapi/properties/properties.hpp> // for properties

#include <type_traits> // for enable_if_t

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
  __SYCL_ALWAYS_INLINE work_group_static() = default;
  work_group_static(const work_group_static &) = delete;
  work_group_static &operator=(const work_group_static &) = delete;

  operator T &() noexcept { return data; }

  template <class TArg = T>
  typename std::enable_if<!std::is_array_v<TArg>, work_group_static &>::type
  operator=(const T &value) noexcept {
    data = value;
    return *this;
  }

  T *operator&() noexcept { return &data; }

private:
  T data;
};

__SYCL_ALWAYS_INLINE
inline void* get_dynamic_work_group_memory() {
#ifdef __SYCL_DEVICE_ONLY__
  return __sycl_dynamicLocalMemoryPlaceholder();
#else
  throw sycl::exception(
      sycl::errc::feature_not_supported,
      "sycl_ext_oneapi_work_group_static extension is not supported on host");
#endif
}

// Property
struct work_group_static_size_key
    : ::sycl::ext::oneapi::experimental::detail::compile_time_property_key<
          ::sycl::ext::oneapi::experimental::detail::WorkGroupStaticMem>,
      property_value<work_group_static_size_key> {
  using value_t = property_value<work_group_static_size_key>;
};

struct work_group_static_size
    : ::sycl::ext::oneapi::experimental::detail::run_time_property_key<
          ::sycl::ext::oneapi::experimental::detail::WorkGroupStaticMem>,
      work_group_static_size_key {
  using value_t = work_group_static_size_key::value_t;
  // Runtime property part
  constexpr work_group_static_size(size_t bytes) : size(bytes) {}

  size_t size;
};

template <> struct is_property_key<work_group_static_size> : std::true_type {};

template <typename T>
struct is_property_key_of<work_group_static_size, T> : std::true_type {};
template <>
struct is_property_value<work_group_static_size>
    : is_property_key<work_group_static_size> {};

namespace detail {
template <> struct PropertyMetaInfo<work_group_static_size> {
  static constexpr const char *name = "sycl-work-group-static";
  static constexpr int value = 1;
};

} // namespace detail

inline bool operator==(const work_group_static_size &lhs,
                       const work_group_static_size &rhs) {
  return lhs.size == rhs.size;
}
inline bool operator!=(const work_group_static_size &lhs,
                       const work_group_static_size &rhs) {
  return !(lhs == rhs);
}

#undef __SYCL_WG_SCOPE

} // namespace experimental
} // namespace ext::oneapi
} // namespace _V1
} // namespace sycl
