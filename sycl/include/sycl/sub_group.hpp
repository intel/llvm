//==----------- sub_group.hpp --- SYCL sub-group ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/__spirv/spirv_ops_subgroup.hpp>
#include <sycl/detail/address_space_cast.hpp>
#include <sycl/detail/defines_elementary.hpp> // for __SYCL_DEPRECATED
#include <sycl/id.hpp>                         // for id
#include <sycl/memory_enums.hpp>               // for memory_scope
#include <sycl/nd_item.hpp>
#include <sycl/range.hpp> // for range

#include <stdint.h> // for uint32_t

#ifndef __SYCL_DEVICE_ONLY__
#include <sycl/exception.hpp> // for exception, make_error...
#endif

namespace sycl {
inline namespace _V1 {

struct sub_group;
namespace ext::oneapi::this_work_item {
inline sycl::sub_group get_sub_group();
} // namespace ext::oneapi::this_work_item

struct sub_group {

  using id_type = id<1>;
  using range_type = range<1>;
  using linear_id_type = uint32_t;
  static constexpr int dimensions = 1;
  static constexpr sycl::memory_scope fence_scope =
      sycl::memory_scope::sub_group;

  /* --- common interface members --- */

  id_type get_local_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_BuiltInSubgroupLocalInvocationId();
#else
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  linear_id_type get_local_linear_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return static_cast<linear_id_type>(get_local_id()[0]);
#else
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  range_type get_local_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_BuiltInSubgroupSize();
#else
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  range_type get_max_local_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_BuiltInSubgroupMaxSize();
#else
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  id_type get_group_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_BuiltInSubgroupId();
#else
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  linear_id_type get_group_linear_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return static_cast<linear_id_type>(get_group_id()[0]);
#else
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  range_type get_group_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_BuiltInNumSubgroups();
#else
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  /* --- synchronization functions --- */
  __SYCL_DEPRECATED(
      "Sub-group barrier with no arguments is deprecated."
      "Use sycl::group_barrier with the sub-group as the argument instead.")
  void barrier() const {
#ifdef __SYCL_DEVICE_ONLY__
    __spirv_ControlBarrier(
        __spv::Scope::Subgroup, __spv::Scope::Subgroup,
        __spv::MemorySemanticsMask::AcquireRelease |
            __spv::MemorySemanticsMask::SubgroupMemory |
            __spv::MemorySemanticsMask::WorkgroupMemory |
            __spv::MemorySemanticsMask::CrossWorkgroupMemory);
#else
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  __SYCL_DEPRECATED(
      "Sub-group barrier accepting fence_space is deprecated."
      "Use sycl::group_barrier with the sub-group as the argument instead.")
  void barrier(access::fence_space accessSpace) const {
#ifdef __SYCL_DEVICE_ONLY__
    int32_t flags = sycl::detail::getSPIRVMemorySemanticsMask(accessSpace);
    __spirv_ControlBarrier(__spv::Scope::Subgroup, __spv::Scope::Subgroup,
                           flags);
#else
    (void)accessSpace;
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  linear_id_type get_group_linear_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return static_cast<linear_id_type>(get_group_range()[0]);
#else
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  linear_id_type get_local_linear_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return static_cast<linear_id_type>(get_local_range()[0]);
#else
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  bool leader() const {
#ifdef __SYCL_DEVICE_ONLY__
    return get_local_linear_id() == 0;
#else
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  // Common member functions for by-value semantics
  friend bool operator==(const sub_group &lhs, const sub_group &rhs) {
#ifdef __SYCL_DEVICE_ONLY__
    return lhs.get_group_id() == rhs.get_group_id();
#else
    (void)lhs;
    (void)rhs;
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  friend bool operator!=(const sub_group &lhs, const sub_group &rhs) {
#ifdef __SYCL_DEVICE_ONLY__
    return !(lhs == rhs);
#else
    (void)lhs;
    (void)rhs;
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

protected:
  template <int dimensions> friend class sycl::nd_item;
  friend sub_group ext::oneapi::this_work_item::get_sub_group();
  sub_group() = default;
};

template <int Dimensions> sub_group nd_item<Dimensions>::get_sub_group() const {
  return sub_group();
}

} // namespace _V1
} // namespace sycl
