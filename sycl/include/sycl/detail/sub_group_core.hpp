//==----- detail/sub_group_core.hpp --- SYCL sub_group core declarations --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/__spirv/spirv_vars.hpp>
#include <sycl/access/access_base.hpp>
#include <sycl/detail/defines_elementary.hpp>
#include <sycl/detail/fwd/multi_ptr.hpp>
#include <sycl/id.hpp>
#include <sycl/memory_enums.hpp>
#include <sycl/range.hpp>

#include <stdint.h>
#include <type_traits>

#ifndef __SYCL_DEVICE_ONLY__
#include <sycl/exception.hpp>
#endif

namespace sycl {
inline namespace _V1 {
template <int Dimensions> class nd_item;
template <typename DataT, int NumElements> class vec;

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

  template <typename CVT, typename T = std::remove_cv_t<CVT>>
  __SYCL_DEPRECATED("Use sycl::ext::oneapi::experimental::group_load instead.")
  T load(CVT *cv_src) const;

  template <typename CVT, access::address_space Space,
            access::decorated IsDecorated, typename T = std::remove_cv_t<CVT>>
  __SYCL_DEPRECATED("Use sycl::ext::oneapi::experimental::group_load instead.")
  T load(multi_ptr<CVT, Space, IsDecorated> cv_src) const;

  template <int N, typename CVT, access::address_space Space,
            access::decorated IsDecorated, typename T = std::remove_cv_t<CVT>>
  __SYCL_DEPRECATED("Use sycl::ext::oneapi::experimental::group_load instead.")
  vec<T, N> load(multi_ptr<CVT, Space, IsDecorated> cv_src) const;

  template <typename T>
  __SYCL_DEPRECATED("Use sycl::ext::oneapi::experimental::group_store instead.")
  void store(T *dst, const remove_decoration_t<T> &x) const;

  template <typename T, access::address_space Space,
            access::decorated DecorateAddress>
  __SYCL_DEPRECATED("Use sycl::ext::oneapi::experimental::group_store instead.")
  void store(multi_ptr<T, Space, DecorateAddress> dst, const T &x) const;

  template <int N, typename T, access::address_space Space,
            access::decorated DecorateAddress>
  __SYCL_DEPRECATED("Use sycl::ext::oneapi::experimental::group_store instead.")
  void store(multi_ptr<T, Space, DecorateAddress> dst,
             const vec<T, N> &x) const;

  __SYCL_DEPRECATED(
      "Sub-group barrier with no arguments is deprecated."
      "Use sycl::group_barrier with the sub-group as the argument instead.")
  void barrier() const;

  __SYCL_DEPRECATED(
      "Sub-group barrier accepting fence_space is deprecated."
      "Use sycl::group_barrier with the sub-group as the argument instead.")
  void barrier(access::fence_space accessSpace) const;

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

} // namespace _V1
} // namespace sycl