//==--- work_group_scratch_memory.hpp - SYCL group local memory extension --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <sycl/detail/defines_elementary.hpp> // for __SYCL_ALWAYS_INLINE
#include <sycl/exception.hpp>                 // for exception
#include <sycl/ext/oneapi/properties/properties.hpp> // for properties

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi {
namespace experimental {

__SYCL_ALWAYS_INLINE
inline void* get_work_group_scratch_memory() {
#ifdef __SYCL_DEVICE_ONLY__
  return __sycl_dynamicLocalMemoryPlaceholder();
#else
  throw sycl::exception(
      sycl::errc::feature_not_supported,
      "sycl_ext_oneapi_work_scratch_memory extension is not supported on host");
#endif
}

// Property
struct work_group_scratch_size_key
    : ::sycl::ext::oneapi::experimental::detail::compile_time_property_key<
          ::sycl::ext::oneapi::experimental::detail::WorkGroupStaticMem>,
      property_value<work_group_scratch_size_key> {
  using value_t = property_value<work_group_scratch_size_key>;
};

struct work_group_scratch_size
    : ::sycl::ext::oneapi::experimental::detail::run_time_property_key<
          ::sycl::ext::oneapi::experimental::detail::WorkGroupStaticMem>,
      work_group_scratch_size_key {
  using value_t = work_group_scratch_size_key::value_t;
  // Runtime property part
  constexpr work_group_scratch_size(size_t bytes) : size(bytes) {}

  size_t size;
};

template <> struct is_property_key<work_group_scratch_size> : std::true_type {};

template <>
struct is_property_value<work_group_scratch_size>
    : is_property_key<work_group_scratch_size> {};

namespace detail {
template <> struct PropertyMetaInfo<work_group_scratch_size> {
  static constexpr const char *name = "sycl-work-group-static";
  static constexpr int value = 1;
};

} // namespace detail

inline bool operator==(const work_group_scratch_size &lhs,
                       const work_group_scratch_size &rhs) {
  return lhs.size == rhs.size;
}
inline bool operator!=(const work_group_scratch_size &lhs,
                       const work_group_scratch_size &rhs) {
  return !(lhs == rhs);
}

} // namespace experimental
} // namespace ext::oneapi
} // namespace _V1
} // namespace sycl
